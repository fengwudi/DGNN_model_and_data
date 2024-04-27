import pathlib
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from ..model.utils import (anonymized_reindex, select_latest_nids,
                           set_anonymized_encoding)
from .data_classes import (ComputationGraph, DummyRestartData, HitData, RestartData,
                           SeqRestartData, StaticRestartData, WalkRestartData)
from .graph import Graph


class ChunkSampler(Sampler):

    def __init__(self, n: int, rank: int, world_size: int, bs: int, seed: int = 0):
        self.n = n
        self.rank = rank
        self.world_size = world_size
        self.bs = bs
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        residual = self.n % (self.world_size * self.bs)
        shift = torch.randint(0, residual + 1, size=(), generator=g)
        l = shift + len(self) * self.rank
        r = l + len(self)
        return iter(range(l, r))

    def __len__(self):
        return self.n // (self.world_size * self.bs) * self.bs

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class GraphCollator:

    def __init__(self, graph: Graph, n_neighbors: int, n_layers: int, *,
                 restarter: str = 'seq',
                 hist_len: Optional[int] = None,
                 n_walks: Optional[int] = None, walk_length: Optional[int] = None,
                 alpha: float = 0.0
                 ):
        self.graph = graph
        self.n_nodes = graph.num_node
        self.n_neighbors = n_neighbors
        self.n_layers = n_layers
        self.restarter = restarter
        self.hist_len = hist_len
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.alpha = alpha

    def check_in_window(self, center_nodes, target_nodes, ts):
        neighbors, *_ = self.graph.sample_temporal_neighbor(
            target_nodes, ts, self.n_neighbors, strategy='recent_edges'
        )  # [bs, n]
        hit = (center_nodes == neighbors.T).T
        hit = torch.from_numpy(hit).float()
        return hit

    def collate_hit_data(self, src_ids, dst_ids, neg_dst_ids, ts):
        src_hits = self.check_in_window(src_ids, dst_ids, ts)
        dst_hits = self.check_in_window(dst_ids, src_ids, ts)
        neg_src_hits = self.check_in_window(src_ids, neg_dst_ids, ts)
        neg_dst_hits = self.check_in_window(neg_dst_ids, src_ids, ts)
        hit_data = HitData(src_hits, dst_hits, neg_src_hits, neg_dst_hits)
        return hit_data

    def __call__(self, batch: List[Tuple[int, int, int, float, int, int]]):
        src_ids, dst_ids, neg_dst_ids, ts, eids, labels = (np.array(x) for x in zip(*batch))
        layers, unique_ids = self.collate_memory_nodes(
            np.concatenate([src_ids, dst_ids, neg_dst_ids]), np.tile(ts, 3)
        )
        restart_data = self.collate_restart_data(
            np.concatenate([src_ids, dst_ids]), np.tile(ts, 2)
        )
        hit_data = self.collate_hit_data(src_ids, dst_ids, neg_dst_ids, ts)

        computation_graph = ComputationGraph([layers, unique_ids], restart_data, hit_data, self.n_nodes)

        src_ids, dst_ids, neg_dst_ids, eids, labels = (
            torch.from_numpy(x).long() for x in (src_ids, dst_ids, neg_dst_ids, eids, labels)
        )
        ts = torch.from_numpy(ts).float()
        return src_ids, dst_ids, neg_dst_ids, ts, eids, labels, computation_graph

    def collate_restart_data(self, nids, ts) -> RestartData:
        if self.restarter == 'seq':
            return self.collate_history(nids, ts)
        elif self.restarter == 'walk':
            return self.collate_walks(nids, ts)
        elif self.restarter == 'static':
            return self.collate_static(nids, ts)
        else:
            return DummyRestartData()

    def collate_memory_nodes(self, nids: np.ndarray, ts: np.ndarray) -> List[Tuple]:
        layers = [None for _ in range(self.n_layers + 1)]
        self._collate_memory_nodes_recursive(nids, ts, self.n_layers, layers)
        layers[0] = [nids, None, None]  # dummy layer storing input node ids
        unique_ids = set()
        for depth in range(len(layers)):
            neigh_nids, neigh_eids, neigh_ts = layers[depth]
            unique_ids.update(neigh_nids.flatten())
            if depth == 0:
                layers[depth] = (torch.from_numpy(neigh_nids).long(), None, None)
            else:
                layers[depth] = (
                    torch.from_numpy(neigh_nids).long(),
                    torch.from_numpy(neigh_eids).long(),
                    torch.from_numpy(neigh_ts).float()
                )
        unique_ids = np.sort(list(unique_ids))  # keep it ndarray
        return layers, unique_ids

    def _collate_memory_nodes_recursive(self, nids: np.ndarray, ts: np.ndarray,
                                        depth: int, layers: List):
        if depth == 0:
            return
        neigh_nids, neigh_eids, neigh_ts, *_ = self.graph.sample_temporal_neighbor(
            nids, ts, self.n_neighbors)  # inputs and outputs are all np.ndarray
        layers[depth] = (neigh_nids, neigh_eids, neigh_ts)
        self._collate_memory_nodes_recursive(neigh_nids.flatten(), neigh_ts.flatten(), depth - 1, layers)

    def collate_history(self, nids: np.ndarray, ts: np.ndarray) -> SeqRestartData:
        # de-duplicate
        unique_nids, index = select_latest_nids(torch.from_numpy(nids), torch.from_numpy(ts))
        unique_nids = unique_nids.numpy()
        ts = ts[index.numpy()]

        hist_nids, hist_eids, hist_ts, hist_dirs = self.graph.get_history(
            unique_nids, ts, self.hist_len)

        anonymized_ids = anonymized_reindex(hist_nids)

        unique_nids = torch.from_numpy(unique_nids).long()
        ts = torch.from_numpy(ts).float()
        hist_nids = torch.from_numpy(hist_nids).long()  # [bs, len]
        anonymized_ids = torch.from_numpy(anonymized_ids).long()
        hist_eids = torch.from_numpy(hist_eids).long()
        hist_ts = torch.from_numpy(hist_ts).float()
        hist_dirs = torch.from_numpy(hist_dirs).long()

        restart_data = SeqRestartData(index, unique_nids, ts, hist_nids,
                                      anonymized_ids, hist_eids, hist_ts, hist_dirs)
        return restart_data

    def collate_static(self, nids: np.ndarray, ts: np.ndarray) -> StaticRestartData:
        unique_nids, index = select_latest_nids(torch.from_numpy(nids), torch.from_numpy(ts))
        unique_nids = unique_nids.numpy()
        ts = ts[index.numpy()]

        _, _, prev_ts, _ = self.graph.get_history(unique_nids, ts, 1)

        unique_nids = torch.from_numpy(unique_nids).long()
        ts = torch.from_numpy(ts).float()
        prev_ts = torch.from_numpy(prev_ts).float()

        restart_data = StaticRestartData(index, unique_nids, ts, prev_ts)
        return restart_data

    def collate_walks(self, nids: np.ndarray, ts: np.ndarray) -> WalkRestartData:
        # de-duplicate
        unique_nids, index = select_latest_nids(torch.from_numpy(nids), torch.from_numpy(ts))
        unique_nids = unique_nids.numpy()
        ts = ts[index.numpy()]

        # NB: Since in restarters we want to estimate h(t'), we should start walks at t'.
        prev_neighbors, prev_eids, prev_ts, prev_dirs = (
            x.squeeze(1) for x in self.graph.get_history(unique_nids, ts, 1))
        walk_nids, walk_eids, walk_ts = self.graph.sample_walks(
            unique_nids, prev_ts, self.n_walks, self.walk_length, self.alpha
        )
        prev_srcs = (1 - prev_dirs) * unique_nids + prev_dirs * prev_neighbors
        prev_dsts = prev_dirs * unique_nids + (1 - prev_dirs) * prev_neighbors
        prev_srcs[prev_neighbors == 0] = 0
        prev_dsts[prev_neighbors == 0] = 0

        walk_anonymized_codes, id2code_dicts = set_anonymized_encoding(walk_nids)

        prev_srcs_codes = np.zeros((len(unique_nids), self.walk_length))
        prev_dsts_codes = np.zeros((len(unique_nids), self.walk_length))
        for i in range(len(unique_nids)):
            prev_srcs_codes[i] = id2code_dicts[i].get(prev_srcs[i], 0)
            prev_dsts_codes[i] = id2code_dicts[i].get(prev_dsts[i], 0)

        unique_nids = torch.from_numpy(unique_nids).long()
        ts = torch.from_numpy(ts).float()
        prev_srcs = torch.from_numpy(prev_srcs).long()
        prev_dsts = torch.from_numpy(prev_dsts).long()
        prev_eids = torch.from_numpy(prev_eids).long()
        walk_nids = torch.from_numpy(walk_nids).long()  # [bs, n_walks, len]
        walk_anonymized_codes = torch.from_numpy(walk_anonymized_codes).float()
        walk_eids = torch.from_numpy(walk_eids).long()
        walk_ts = torch.from_numpy(walk_ts).float()
        prev_srcs_codes = torch.from_numpy(prev_srcs_codes).float()
        prev_dsts_codes = torch.from_numpy(prev_dsts_codes).float()

        restart_data = WalkRestartData(index, unique_nids, ts, prev_srcs, prev_dsts, prev_eids,
                                       walk_nids, walk_anonymized_codes, walk_eids, walk_ts,
                                       prev_srcs_codes, prev_dsts_codes)
        return restart_data


class InteractionData(Dataset):
    """
    Data object storing interactions.
    """

    def __init__(self, src, dst, ts, eids, labels, seed=0, eval=False, neg_dst=None,
                 src_population=None, dst_population=None):
        lengths = [len(x) for x in [src, dst, ts, eids, labels]]
        assert np.all(np.equal(lengths, lengths[0]))
        self.src = src
        self.dst = dst
        self.ts = ts
        self.eids = eids
        self.labels = labels
        self.eval = eval
        self.seed = seed
        self.neg_dst = None
        src_population = src if src_population is None else src_population
        dst_population = dst if dst_population is None else dst_population
        self.neg_dst_sampler = RandEdgeSampler(src_population, dst_population, seed)
        if self.eval:
            if neg_dst is not None:
                self.neg_dst = neg_dst
            else:
                # bs = 200 as TGN and TGAT did
                self.neg_dst = self.neg_dst_sampler.pre_sample_neg_dsts(len(ts), bs=200)
        self.num_node = len(set(self.src).union(self.dst))
        self.reindexer = None

    def get_stats(self):
        return compute_delta_std_and_mean(self.src, self.dst, self.ts)

    def get_subset(self, start, end):
        src = self.src[start:end]
        dst = self.dst[start:end]
        ts = self.ts[start:end]
        eids = self.eids[start:end]
        labels = self.labels[start:end]
        ds = InteractionData(src, dst, ts, eids, labels, self.seed, self.eval, self.neg_dst)
        ds.num_node = self.num_node
        return ds

    def get_subset_by_idx(self, idx):
        src = self.src[idx]
        dst = self.dst[idx]
        ts = self.ts[idx]
        eids = self.eids[idx]
        labels = self.labels[idx]
        ds = InteractionData(src, dst, ts, eids, labels, self.seed, self.eval, self.neg_dst)
        ds.num_node = self.num_node
        return ds

    def get_subset_and_reindex_by_nodes(self, nodes, shared_nodes=None):
        """
        Get a subset of the InteractionData whose edges happened
        within the given node set, and re-index node to 1~N.
        If there are some nodes shared
        across clusters, 
        -----
        nodes: a list of node ids
        shared_nodes: if given, 
        """
        reindexer = ClusterReindexer(nodes, shared_nodes)
        # 若子图没有边则会有问题
        if shared_nodes:
            shared_nodes = set(shared_nodes)
            unique_nodes = list(set(nodes) - shared_nodes)
            shared_nodes = list(shared_nodes)
            all_nodes = shared_nodes + unique_nodes
            idx = np.isin(self.src, all_nodes) & np.isin(self.dst, all_nodes)
        else:
            idx = np.isin(self.src, nodes) & np.isin(self.dst, nodes)
        
        src = reindexer.id_global2cluster(self.src[idx])
        dst = reindexer.id_global2cluster(self.dst[idx])
        ts = self.ts[idx]
        eids = self.eids[idx]
        labels = self.labels[idx]
        # todo: check: 负采样仅在子图中进行
        ds = InteractionData(src, dst, ts, eids, labels, self.seed, self.eval, neg_dst=None)
        # ds = InteractionData(src, dst, ts, eids, labels, self.seed, self.eval, self.neg_dst)
        # todo: ds.num_node 应该是之间自动算出来的？ 如果self.num_node会之间算整个数据集的node数
        # ds.num_node = self.num_node
        ds.reindexer = reindexer
        return ds, reindexer.global_list

    def get_neg_dst_item(self, i) -> int:
        if self.eval:
            return self.neg_dst[i]
        else:
            _, neg_dst = self.neg_dst_sampler.sample(1)
        return neg_dst.item()

    def __getitem__(self, i) -> Tuple[int, int, int, float, int, int]:
        return (self.src[i], self.dst[i], self.get_neg_dst_item(i),
                self.ts[i], self.eids[i], self.labels[i])

    def __len__(self):
        return len(self.ts)

    def __repr__(self):
        m = len(self)
        n = len(set(self.src).union(self.dst))
        tmin = self.ts.min()
        tmax = self.ts.max()
        s = f"Data(#edges={m}, #nodes={n}, trange=({tmin:.1f}, {tmax:.1f}))"
        return s

    __str__ = __repr__

    def summary(self, name=None):
        if name is None:
            msg = ""
        else:
            msg = f"[{name}]\n"
        msg += f"# Interactions: {len(self.src)}\n"
        msg += f"# Src: {len(set(self.src))} # Dst: {len(set(self.dst))} # Total: {len(set(self.src).union(self.dst))}\n"
        msg += f"Src: {min(self.src)} -> {max(self.src)}\n"
        msg += f"Dst: {min(self.dst)} -> {max(self.dst)}\n"
        msg += f"Ts : {min(self.ts):.1f} -> {max(self.ts):.1f}"
        return msg


class ClusterReindexer:

    def __init__(self, nodes, shared_nodes=None):
        shared_nodes = set() if shared_nodes is None else set(shared_nodes)
        unique_nodes = set(nodes) - shared_nodes
        self.global_list = np.concatenate([[0], sorted(shared_nodes), sorted(unique_nodes)])
        self.global2cluster_map = {val: i for i, val in enumerate(self.global_list)}

    def id_cluster2global(self, local_ids) -> np.ndarray:
        return self.global_list[local_ids]

    def id_global2cluster(self, global_ids) -> np.ndarray:
        nodes = np.array([self.global2cluster_map[i] for i in global_ids])
        return nodes


class RandEdgeSampler:
    def __init__(self, src_list: np.ndarray, dst_list: np.ndarray,
                 seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        src_index = self.rng.randint(0, len(self.src_list), size)
        dst_index = self.rng.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.rng = np.random.RandomState(self.seed)

    def pre_sample_neg_dsts(self, n_total: int, bs: int = 200) -> np.ndarray:
        self.reset_random_state()
        residual = n_total
        neg_dsts = []
        while residual > 0:
            if residual >= bs:
                _, negs = self.sample(bs)
                residual -= bs
            else:
                _, negs = self.sample(residual)
                residual -= residual
            neg_dsts.append(negs)
        all_neg_dsts = np.concatenate(neg_dsts)
        assert len(all_neg_dsts) == n_total
        return all_neg_dsts


def load_jodie_data(
        name: str, train_seed: int, *,
        root='.', data_seed=2020, val_p=0.7, test_p=0.85
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], InteractionData, InteractionData,
InteractionData, InteractionData, InteractionData, InteractionData]:
    """
    Load Jodie data.
    """
    root = pathlib.Path(root)
    graph_df = pd.read_csv(root / './data/ml_{}.csv'.format(name))
    if (root / './data/ml_{}.npy'.format(name)).exists():
        efeats = np.load(root / './data/ml_{}.npy'.format(name))
    else:
        efeats = None
    if (root / './data/ml_{}_node.npy'.format(name)).exists():
        nfeats = np.load(root / './data/ml_{}_node.npy'.format(name))
    else:
        nfeats = None

    # default 0.7, 0.85
    val_time, test_time = list(np.quantile(graph_df.ts, [val_p, test_p]))

    src = graph_df.u.values
    dst = graph_df.i.values
    eids = graph_df.idx.values
    labels = graph_df.label.values
    ts = graph_df.ts.values

    full_data = InteractionData(src, dst, ts, eids, labels)

    random.seed(data_seed)  # use seed=2020 as TGAT and TGN did

    node_set = set(src) | set(dst)
    n_total_nodes = len(node_set)

    # nodes that appear at inference time
    inference_node_set = set(src[ts > val_time]).union(set(dst[ts > val_time]))
    # sample (0.1 * n_total_nodes) and remove them from training data
    # these nodes are a part of inductive nodes
    # NB: potential bug: 0 being selected?
    part_inductive_node_set = set(random.sample(
        inference_node_set, int(0.1 * n_total_nodes)))

    # training data: no later than val_time, and contains no inductive nodes
    part_inductive_src_mask = graph_df.u.isin(part_inductive_node_set).values
    part_inductive_dst_mask = graph_df.i.isin(part_inductive_node_set).values
    part_inductive_edge_mask = np.logical_and(
        ~part_inductive_src_mask, ~part_inductive_dst_mask)
    train_data_mask = np.logical_and(ts <= val_time, part_inductive_edge_mask)

    train_data = InteractionData(
        *[x[train_data_mask] for x in [src, dst, ts, eids, labels]],
        seed=train_seed, eval=False
    )

    # inductive nodes: not being observed during training phase
    train_node_set = set(train_data.src) | set(train_data.dst)
    assert len(train_node_set & part_inductive_node_set) == 0
    inductive_node_set = node_set - train_node_set

    # valid/test data mask
    val_data_mask = np.logical_and(ts <= test_time, ts > val_time)
    test_data_mask = ts > test_time

    # valid/test data mask for inductive nodes
    inductive_edge_mask = np.isin(src, list(inductive_node_set)) \
                          | np.isin(dst, list(inductive_node_set))
    inductive_val_data_mask = val_data_mask & inductive_edge_mask
    inductive_test_data_mask = test_data_mask & inductive_edge_mask

    val_data = InteractionData(
        *[x[val_data_mask] for x in [src, dst, ts, eids, labels]],
        seed=0, eval=True,
        src_population=src, dst_population=dst
    )
    test_data = InteractionData(
        *[x[test_data_mask] for x in [src, dst, ts, eids, labels]],
        seed=2, eval=True,
        src_population=src, dst_population=dst
    )
    inductive_val_data = InteractionData(
        *[x[inductive_val_data_mask] for x in [src, dst, ts, eids, labels]],
        seed=1, eval=True
    )
    inductive_test_data = InteractionData(
        *[x[inductive_test_data_mask] for x in [src, dst, ts, eids, labels]],
        seed=3, eval=True
    )

    return (nfeats, efeats, full_data, train_data, val_data, test_data,
            inductive_val_data, inductive_test_data)


def load_jodie_data_for_node_task(
        name: str, train_seed: int, use_validation: bool = False, *,
        root='.', data_seed=2020, val_p=0.7, test_p=0.85
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
InteractionData, InteractionData, InteractionData, InteractionData]:
    """
    Load Jodie data for node classification.
    """
    root = pathlib.Path(root)
    graph_df = pd.read_csv(root / './data/ml_{}.csv'.format(name))
    if (root / './data/ml_{}.npy'.format(name)).exists():
        efeats = np.load(root / './data/ml_{}.npy'.format(name))
    else:
        efeats = None
    if (root / './data/ml_{}_node.npy'.format(name)).exists():
        nfeats = np.load(root / './data/ml_{}_node.npy'.format(name))
    else:
        nfeats = None

    # default 0.7, 0.85
    val_time, test_time = list(np.quantile(graph_df.ts, [val_p, test_p]))

    src = graph_df.u.values
    dst = graph_df.i.values
    eids = graph_df.idx.values
    labels = graph_df.label.values
    ts = graph_df.ts.values

    full_data = InteractionData(src, dst, ts, eids, labels)

    random.seed(data_seed)  # use seed=2020 as TGAT and TGN did

    test_data_mask = ts > test_time
    if use_validation:
        train_data_mask = ts <= val_time
        val_data_mask = np.logical_and(ts <= test_time, ts > val_time)
    else:
        train_data_mask = ts <= test_time
        val_data_mask = test_data_mask

    train_data = InteractionData(
        *[x[train_data_mask] for x in [src, dst, ts, eids, labels]],
        seed=train_seed, eval=False
    )

    val_data = InteractionData(
        *[x[val_data_mask] for x in [src, dst, ts, eids, labels]],
        seed=0, eval=True
    )
    test_data = InteractionData(
        *[x[test_data_mask] for x in [src, dst, ts, eids, labels]],
        seed=2, eval=True
    )

    return nfeats, efeats, full_data, train_data, val_data, test_data,


def compute_delta_std_and_mean(srcs: np.ndarray, dsts: np.ndarray, ts: np.ndarray
                               ) -> Tuple[float, float, float, float, float]:
    """
    Compute the mean and std of differences of times.
    """
    last_ts = dict()
    all_delta_ts = []
    src_delta_ts = []
    dst_delta_ts = []
    for s, d, t in zip(srcs, dsts, ts):
        s_last_t = last_ts.get(s, 0)
        d_last_t = last_ts.get(d, 0)
        all_delta_ts.append(t - s_last_t)
        all_delta_ts.append(t - d_last_t)
        src_delta_ts.append(t - s_last_t)
        dst_delta_ts.append(t - d_last_t)
        last_ts[s] = t
        last_ts[d] = t
    all_mean = np.mean(all_delta_ts)
    all_std = np.std(all_delta_ts)
    src_mean = np.mean(src_delta_ts)
    src_std = np.std(src_delta_ts)
    dst_mean = np.mean(dst_delta_ts)
    dst_std = np.std(dst_delta_ts)
    return all_mean, all_std, src_mean, src_std, dst_mean, dst_std


def is_sorted(x):
    for i in range(len(x) - 1):
        if x[i] > x[i + 1]:
            return False
    return True
