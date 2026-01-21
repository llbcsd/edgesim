import os
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import pickle
import math


def edges_to_matrix_dir(node_num, edges, with_diag=True):
    m = np.zeros(shape=(node_num, node_num), dtype=np.uint8)
    if len(edges) != 0:
        m[edges[:, 0], edges[:, 1]] = 1
    if with_diag:
        m[np.arange(node_num), np.arange(node_num)] = 1
    return m


# transfer edge_feat (edge_num, edge_feat) into edge_matrix (node_num, node_num, edge_feat) with direction
def edges_to_feature_matrix_with_dir(node_num, edges, edges_feat):
    em = torch.zeros(node_num, node_num, edges_feat.shape[-1], dtype=torch.float32)
    for idx, e in enumerate(edges):
        em[e[0]][e[1]] = edges_feat[idx]
    return em


# gather graph batch into one graph, return full edge matrix
def collate_fn_dir_edge_with_feat(batch, with_diag=True):
    nodes_list = [b[0] for b in batch]
    nodes = torch.cat(nodes_list, dim=0)

    nodes_lens = np.fromiter(map(lambda l: l.shape[0], nodes_list), dtype=np.int64)
    nodes_inds = np.cumsum(nodes_lens)
    nodes_num = nodes_inds[-1]
    nodes_inds = np.insert(nodes_inds, 0, 0)
    nodes_inds = np.delete(nodes_inds, -1)
    edges_list = [b[1] for b in batch]
    edges_list = [e + i for e, i in zip(edges_list, nodes_inds) if len(e) != 0]
    if len(edges_list) != 0:
        edges = np.concatenate(edges_list, axis=0)
    else:
        edges = []
    m = edges_to_matrix_dir(nodes_num, edges, with_diag)

    batch_mask = [torch.tensor([i] * k, dtype=torch.int32) for i, k in zip(range(len(batch)), nodes_lens)]
    batch_mask = torch.cat(batch_mask, dim=0)

    
    edge_feat_list = [b[2] for b in batch]
    edge_feats = torch.cat(edge_feat_list, dim=0)
    edge_feat_matrix = edges_to_feature_matrix_with_dir(nodes_num, edges, edge_feats)
    return nodes, torch.from_numpy(m).float(), edge_feat_matrix, batch_mask



# combine multiple graphs into one large graph.
def collate_fn_pair_typeD_with_dir(batch):
    cfg1 = [b[0] for b in batch]
    cfg2 = [b[1] for b in batch]

    nodes1, m1, edge_m1, bm1 = collate_fn_dir_edge_with_feat(cfg1, with_diag=False)
    nodes2, m2, edge_m2, bm2 = collate_fn_dir_edge_with_feat(cfg2, with_diag=False)

    if len(batch[0]) == 2:
        return nodes1, nodes2, m1, m2, edge_m1, edge_m2, bm1, bm2

    labels = [b[2] for b in batch]
    return nodes1, nodes2, m1, m2, edge_m1, edge_m2, bm1, bm2, labels

