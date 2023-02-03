import argparse
import logging
import numpy as np
import time
import torch
import torch.nn as nn
from models.GCN import GCN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import pairwise_distances_argmin
from models.build_graphs import build_graphs
from models.GCN_PCA_GM import Siamese_Gconv

from common.torch import to_numpy
from models.pointnet_util import square_distance, angle_difference
from models.feature_nets import FeatExtractionEarlyFusion
from models.feature_nets import ParameterPredictionNet
# from models.feature_nets import ParameterPredictionNetConstant as ParameterPredictionNet
from common.math_torch import se3

_logger = logging.getLogger(__name__)

_EPS = 1e-5  # To prevent division by zero


def match_features(feat_src, feat_ref, metric='l2'):
    """ Compute pairwise distance between features

    Args:
        feat_src: (B, J, C)
        feat_ref: (B, K, C)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
         in the src agrees with every point in the ref.
    """
    assert feat_src.shape[-1] == feat_ref.shape[-1]

    if metric == 'l2':
        dist_matrix = square_distance(feat_src, feat_ref)
    elif metric == 'angle':
        feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
        feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)

        dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
    else:
        raise NotImplementedError

    return dist_matrix


def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform


class RPMNet(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.add_slack = not args.no_slack
        self.num_sk_iter = args.num_sk_iter
        self.gnn_layer = 5

        #GCN params
        GCN_input_dim = 96
        GCN_hidden_dim = 256
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(GCN_input_dim, GCN_hidden_dim)
            else:
                gnn_layer = Siamese_Gconv(GCN_hidden_dim, GCN_hidden_dim)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(GCN_hidden_dim * 2, GCN_hidden_dim))

    def compute_affinity(self, beta, feat_distance, alpha=0.5):
        """Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
        if isinstance(alpha, float):
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
        else:
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
        return hybrid_affinity

    def forward(self, data, num_iter: int = 1):
        """Forward pass for RPMNet

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, 6)
                    'points_ref': Reference points (B, K, 6)
            num_iter (int): Number of iterations. Recommended to be 2 for training

        Returns:
            transform: Transform to apply to source points such that they align to reference
            src_transformed: Transformed source points
        """
        endpoints = {}

        xyz_ref, norm_ref = data['points_ref'][:, :, :3], data['points_ref'][:, :, 3:6]
        xyz_src, norm_src = data['points_src'][:, :, :3], data['points_src'][:, :, 3:6]
        xyz_src_t, norm_src_t = xyz_src, norm_src
        B, N, d = xyz_src.shape

        transforms = []
        all_gamma, all_perm_matrices, all_weighted_ref = [], [], []
        all_beta, all_alpha = [], []
        for i in range(num_iter):

           # beta, alpha = self.weights_net([xyz_src_t, xyz_ref])
            feat_src = self.feat_extractor(xyz_src_t, norm_src_t)
            feat_ref = self.feat_extractor(xyz_ref, norm_ref)
            # print('feat_src dim : {}'.format(feat_src.shape))
            # print('feat_ref dim : {}'.format(feat_ref.shape))

        #section 1: find time to build graph
            build_graph_strt_time = time.time()
            GCN_input_dim = feat_src.shape[2]
            GCN_hidden_dim = 256
            # print('GCN input dim : {}'.format(GCN_input_dim))
            # print('GCN hidden dim : {}'.format(GCN_hidden_dim))

            g1_feats1_adj_matrices = np.zeros((B, N, N))
            n_edges1_gt = np.zeros((B))
            # graph_list_1 = []
            edge_indices_1 = []

            g2_feats2_adj_matrices = np.zeros((B, N, N))
            n_edges2_gt = np.zeros((B))
            # graph_list_2 = []
            edge_indices_2 = []

            for b in range(B):
                g1_feats1_adj_matrices[b, :, :], n_edges1_gt[b] = build_graphs(feat_src[b, :, :].detach().cpu().numpy(), n = feat_src.shape[1])
                # print(np.array((g1_feats1_adj_matrices[b, :, :] > 0).nonzero()).shape)
                # edge_index = torch.tensor(np.array((g1_feats1_adj_matrices[b, :, :] > 0).nonzero()), dtype = torch.long)
                # if torch.cuda.is_available():
                #     edge_index = edge_index.to("cuda:0")
                # row, col = edge_index
                # edge_weight = g1_feats1_adj_matrices[b, :, :][row, col]
                # edge_indices_1.append(edge_index)
                #convert adj matrix  to edge index format
                # r, c = np.where(g1_feats1_adj_matrices[b, :, :])
                # coo = np.array(list(zip(r, c)))
                # print('jai:{}'.format(coo.shape))
                # print(edge_index.shape)
                # print(np.allclose(coo, edge_index))
                # coo = torch.tensor(np.reshape(coo, (2, -1)), dtype = torch.long)
                # if torch.cuda.is_available():
                #     coo = coo.to("cuda:0")
                # data = Data(x = feat_src[b, :, :].detach().cpu().numpy(), edge_index = coo)
                # graph_list_1.append(data)

                g2_feats2_adj_matrices[b, :, :], n_edges2_gt[b] = build_graphs(feat_ref[b, :, :].detach().cpu().numpy(), n = feat_ref.shape[1])
                # edge_index = torch.tensor(np.array((g2_feats2_adj_matrices[b, :, :] > 0).nonzero()), dtype = torch.long)
                # if torch.cuda.is_available():
                #     edge_index = edge_index.to("cuda:0")
                # row, col = edge_index
                # edge_weight = g1_feats1_adj_matrices[b, :, :][row, col]
                # edge_indices_2.append(edge_index)                
                #convert adj matrix  to edge index format
                # r, c = np.where(g2_feats2_adj_matrices[b, :, :])
                # coo = np.array(list(zip(r, c)))
                # coo = torch.tensor(np.reshape(coo, (2, -1)), dtype = torch.long)
                # if torch.cuda.is_available():
                #     coo = coo.to("cuda:0")
                # data = Data(x = feat_ref[b, :, :].detach().cpu().numpy(), edge_index = coo)
                # graph_list_2.append(data)
            build_graph_end_time = time.time()
            # print('time to build graphs for a batch of {} point clouds:{}'.format(B, build_graph_end_time - build_graph_strt_time))

            # graph_loader_1 = DataLoader(graph_list_1, batch_size = B, shuffle=False)
            # graph_loader_2 = DataLoader(graph_list_2, batch_size = B, shuffle=False)

            # build_graph_end_time = time.time()
            # print('time to build graphs for a batch of {} point clouds:{}'.format(B, build_graph_end_time - build_graph_strt_time))

            # loader = DataLoader(data_list, batch_size=1, shuffle=False)

            if torch.cuda.is_available():
                g1_feats1_adj_matrices = torch.tensor(g1_feats1_adj_matrices).to("cuda:0")
                g2_feats2_adj_matrices = torch.tensor(g2_feats2_adj_matrices).to("cuda:0")
                # feat_src = feat_src.to("cuda:0")
                # feat_ref = feat_ref.to("cuda:0")
                node_embeddings_pc1 = feat_src.to("cuda:0")
                node_embeddings_pc2 = feat_ref.to("cuda:0")
                # print('node_embeddings_pc1 shape: {}'.format(node_embeddings_pc1.shape))
                # print('node_embeddings_pc2 shape : {}'.format(node_embeddings_pc2.shape))

            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                node_embeddings_pc1, node_embeddings_pc2 = gnn_layer([g1_feats1_adj_matrices.float(), node_embeddings_pc1.float()], [g2_feats2_adj_matrices.float(), node_embeddings_pc2.float()])
        # #section 2: find time to generate graph embeddings for all clouds in current batch
        #     generate_graph_embeddings_strt_time = time.time()
        #     # Generate Node Embeddings using GCN
        #     GCN_model =  GCN(GCN_input_dim, GCN_hidden_dim, GCN_hidden_dim, 3, 0.5).to("cuda:0")
        #     node_embeddings_pc1 = torch.tensor(np.zeros((B, N, GCN_hidden_dim)), dtype = torch.float)
        #     node_embeddings_pc2 = torch.tensor(np.zeros((B, N, GCN_hidden_dim)), dtype = torch.float)

        #     P = torch.zeros(B, N, N).to('cuda:0')
        #     # node_embeddings_pc1 = GCN_model(next(iter(graph_loader_1)))
        #     # node_embeddings_pc1 = GCN_model(next(iter(graph_loader_2)))
        #     for b in range(B):
        #         # print(edge_indices_1[b].shape)
        #         node_embeddings_pc1[b, : , :] = GCN_model(feat_src[b, : , :], edge_indices_1[b])
        #         # print(node_embeddings_pc1[b, : , :])
        #         # print('..................................................................................................')
        #         node_embeddings_pc2[b, : , :] = GCN_model(feat_ref[b, : , :], edge_indices_2[b])
        #         # print(node_embeddings_pc2[b, : , :])
			
            # print('node_embeddings_pc1 shape: {}'.format(node_embeddings_pc1.shape))
            # print('node_embeddings_pc2 shape : {}'.format(node_embeddings_pc2.shape))
            if torch.cuda.is_available():
                node_embeddings_pc1 = node_embeddings_pc1.to("cuda:0")
                node_embeddings_pc2 = node_embeddings_pc2.to("cuda:0")
        #     generate_graph_embeddings_end_time = time.time()
            # print('time to generate graph embeddings for a batch of {} point clouds:{}'.format(B, generate_graph_embeddings_end_time - generate_graph_embeddings_strt_time))
            # feat_distance = match_features(feat_src, feat_ref)   #this is only cost matrix (affinity matrix)
            gcn_feat_distance = match_features(node_embeddings_pc1, node_embeddings_pc2)   #this is only cost matrix (affinity matrix)

            # print('model grad:{}'.format(GCN_model.convs[0].edge_weight))

           # affinity = self.compute_affinity(beta, feat_distance, alpha=alpha)

            # Compute weighted coordinates
            # log_perm_matrix = sinkhorn(affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
            log_perm_matrix = sinkhorn(gcn_feat_distance, n_iters=self.num_sk_iter, slack=self.add_slack)
            perm_matrix = torch.exp(log_perm_matrix)
            weighted_ref = perm_matrix @ xyz_ref / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)

            # Compute transform and transform points
            transform = compute_rigid_transform(xyz_src, weighted_ref, weights=torch.sum(perm_matrix, dim=2))
            xyz_src_t, norm_src_t = se3.transform(transform.detach(), xyz_src, norm_src)

            transforms.append(transform)
            all_gamma.append(torch.exp(gcn_feat_distance))
            all_perm_matrices.append(perm_matrix)
            all_weighted_ref.append(weighted_ref)
            # all_beta.append(to_numpy(beta))
            # all_alpha.append(to_numpy(alpha))

        endpoints['perm_matrices_init'] = all_gamma
        endpoints['perm_matrices'] = all_perm_matrices
        endpoints['weighted_ref'] = all_weighted_ref
        # endpoints['beta'] = np.stack(all_beta, axis=0)
        # endpoints['alpha'] = np.stack(all_alpha, axis=0)

        return transforms, endpoints


class RPMNetEarlyFusion(RPMNet):
    """Early fusion implementation of RPMNet, as described in the paper"""
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.weights_net = ParameterPredictionNet(weights_dim=[0])
        self.feat_extractor = FeatExtractionEarlyFusion(
            features=args.features, feature_dim=args.feat_dim,
            radius=args.radius, num_neighbors=args.num_neighbors)


def get_model(args: argparse.Namespace) -> RPMNet:
    return RPMNetEarlyFusion(args)
