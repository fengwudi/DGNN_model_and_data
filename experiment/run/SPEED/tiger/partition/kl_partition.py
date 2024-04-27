### This file is unused, please do not use the buildin_kl

import numpy as np
from scipy import sparse as sp
import networkx as nx

from ..data.data_loader import InteractionData


def distribute_node(full_data: InteractionData, train_data: InteractionData, k: int=1):
    parts, cut = partition_data(train_data, full_data.num_node, k=k)
    full_data, train_data = trim_data_from_partition(full_data, train_data, parts)
    return full_data, train_data, cut


def partition_data(data, n_nodes, k=1):
    v1 = data.src
    v2 = data.dst
    adj = sp.coo_matrix((np.ones_like(v1), (v1, v2)),  shape=(n_nodes+1, n_nodes+1))
    adj += adj.T
    graph = nx.from_scipy_sparse_matrix(adj)
    partition = [np.arange(n_nodes)]
    for _ in range(k):
        new_partition = []
        for subset in partition:
            subg = nx.subgraph(graph, subset)
            new_partition.extend(nx.community.kernighan_lin_bisection(subg))
        partition = new_partition
    cut = compute_cut(graph, partition)
    return [np.array(list(p)) for p in partition], cut


def compute_cut(graph, parts):
    c = 0
    for i in range(len(parts)):
        for j in range(i+1, len(parts)):
            c += nx.cut_size(graph, parts[i], parts[j], weight='weight')
    return c


def trim_data_from_partition(full_data, train_data, parts):
    mask = None
    src = train_data.src
    dst = train_data.dst
    train_size = len(train_data)
    for p in parts:
        match = np.isin(src, p) & np.isin(dst, p)
        mask = match if mask is None else (mask | match)
    valid_idx = np.where(mask)[0]
    train_data = train_data.get_subset_by_idx(valid_idx)
    full_idx = np.concatenate([valid_idx, np.arange(train_size, len(full_data))])
    full_data = full_data.get_subset_by_idx(full_idx)
    return full_data, train_data