from functools import lru_cache
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:  # to prevent circular import when type hinting
    from .data_loader import InteractionData


class Graph:
    def __init__(self, adj_list, strategy='recent_nodes', seed=None, alpha=0.0):
        """
        Init a graph with 'adj_list'
        Temporal neighbors are extracted according to 'strategy'
        ----
        adj_list: adjacency list
        strategy: ['recent_nodes', 'recent_edges', 'uniform']
        seed: random seed
        """
        self.num_node = len(adj_list)
        self.strategy = strategy
        self.rng = np.random.RandomState(seed)
        self.alpha = alpha

        self.node2neighbors = []
        self.node2eids = []
        self.node2ts = []
        self.node2flags = []  # is src node?

        for edges in adj_list:
            # edges should have been sorted globally (at least for Jodie's datasets)
            sorted_edges = sorted(edges, key=lambda x: x[2])  # sorted by time
            self.node2neighbors.append(np.array([x[0] for x in sorted_edges]))
            self.node2eids.append(np.array([x[1] for x in sorted_edges]))
            self.node2ts.append(np.array([x[2] for x in sorted_edges]))
            self.node2flags.append(np.array([x[3] for x in sorted_edges]))
    
    @classmethod
    def from_data(cls, data: 'InteractionData', strategy='recent_nodes', 
                  seed=None, max_node_id=None):
        adj_list = data2adjlist(data, max_node_id=max_node_id)
        return cls(adj_list, strategy=strategy, seed=seed)
    
    @lru_cache(10000)
    def find_before(self, nid: int, t: float
                   ) -> Tuple[List[int], List[int], List[float], List[int]]:
        """
        Find all edges related to 'nid' before time 't'. (strict '<')
        """
        # If input list is empty, i = 0
        i = np.searchsorted(self.node2ts[nid], t, side='left')
        neighbors, eids, ts, flags = [x[nid][:i] for x in (self.node2neighbors, self.node2eids, self.node2ts, self.node2flags)]
        return neighbors, eids, ts, flags
    
    @lru_cache(10000)
    def find_unique_before(self, nid: int, t: float
                          ) -> Tuple[List[int], List[int], List[float], List[int]]:
        """
        Find all unique nodes related to 'nid' before time 't'. (strict '<')
        """
        neighbors, eids, ts, flags = self.find_before(nid, t)
        _, unique_idx = np.unique(neighbors[::-1], return_index=True)  # find unique nodes' indices
        unique_idx = len(neighbors) - 1 - np.sort(unique_idx)[::-1]  # np.unique sorts unique values
        neighbors, eids, ts, flags = [x[unique_idx] for x in (neighbors, eids, ts, flags)]
        return neighbors, eids, ts, flags

    def sample_temporal_neighbor(self, nids: np.ndarray, ts: np.ndarray, 
                                 n_neighbors: int=20, strategy: Optional[str]=None
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract 'n_neighbors' edges for each (nid, t) pair
        TODO: Bottleneck! Total time >50% 
        ---
        nids: a list of node indices
        ts  : a list of timestamps
        n_neighbors: number of neighbors
        strategy: use this to override the global strategy
        ---
        returns: batch_neighbors, batch_eids, batch_ts, batch_dirs
        batch_neighbors:  [bs, n_neighbors] neighbors' indices
        batch_eids:       [bs, n_neighbors] edge indices
        batch_ts:         [bs, n_neighbors] corresponding timestamps
        dirs:             [bs, n_neighbors] directions (nids as target nodes)
        """
        strategy = self.strategy if strategy is None else strategy
        bs = len(nids)
        assert(len(nids) == len(ts))
        # to be filled
        batch_neighbors = np.zeros([bs, n_neighbors], dtype=int)
        batch_eids = np.zeros([bs, n_neighbors], dtype=int)
        batch_ts = np.zeros([bs, n_neighbors], dtype=np.float32)
        batch_dirs = np.zeros([bs, n_neighbors], dtype=int)

        for i, (nid, t) in enumerate(zip(nids, ts)):
            local_neighbors, local_eids, local_ts, local_dirs = self.find_before(nid, t)

            if len(local_neighbors) == 0:  # no neighbors, use default null values
                continue

            # todo: extracting neighbors separately w.r.t src/dst
            if strategy == 'uniform' or strategy == 'time':
                if strategy == 'uniform' or math.isclose(self.alpha, 0):
                    sampled_idx = self.rng.randint(0, len(local_neighbors), n_neighbors)
                else:
                    time_delta = t - local_ts
                    sampling_weight = np.exp(- self.alpha * time_delta)
                    sampling_weight = sampling_weight / sampling_weight.sum()  # normalize
                    sampled_idx = self.rng.choice(len(local_neighbors), n_neighbors,
                                                  replace=True, p=sampling_weight)
                # sort by time
                sort_idx = local_ts[sampled_idx].argsort()
                batch_neighbors[i, :] = local_neighbors[sampled_idx][sort_idx]
                batch_eids[i, :] = local_eids[sampled_idx][sort_idx]
                batch_ts[i, :] = local_ts[sampled_idx][sort_idx]
                batch_dirs[i, :] = local_dirs[sampled_idx][sort_idx]

            elif strategy == 'recent_edges':
                local_neighbors = local_neighbors[-n_neighbors:]
                local_eids = local_eids[-n_neighbors:]
                local_ts = local_ts[-n_neighbors:]
                local_dirs = local_dirs[-n_neighbors:]
                len_hist = len(local_neighbors)
                # The first (n_neighbors - len_hist) are filled with default null values
                batch_neighbors[i, n_neighbors - len_hist:] = local_neighbors
                batch_eids[i, n_neighbors - len_hist:] = local_eids
                batch_ts[i, n_neighbors - len_hist:] = local_ts
                batch_dirs[i, n_neighbors - len_hist:] = local_dirs

            elif strategy == 'recent_nodes':
                _, unique_idx = np.unique(local_neighbors[::-1], return_index=True)  # find unique nodes' indices
                unique_idx = len(local_neighbors) - 1 - np.sort(unique_idx)[::-1]  # np.unique sorts unique values

                local_neighbors = local_neighbors[unique_idx][-n_neighbors:]
                local_eids = local_eids[unique_idx][-n_neighbors:]
                local_ts = local_ts[unique_idx][-n_neighbors:]
                local_dirs = local_dirs[unique_idx][-n_neighbors:]
                len_hist = len(local_neighbors)

                # The first (n_neighbors - len_hist) are filled with default null values
                batch_neighbors[i, n_neighbors - len_hist:] = local_neighbors
                batch_eids[i, n_neighbors - len_hist:] = local_eids
                batch_ts[i, n_neighbors - len_hist:] = local_ts
                batch_dirs[i, n_neighbors - len_hist:] = local_dirs

            else:
                raise NotImplementedError
        
        return batch_neighbors, batch_eids, batch_ts, batch_dirs
    
    def get_history(self, nids: np.ndarray, ts: np.ndarray, hist_len: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hist_nids, hist_eids, hist_ts, hist_dirs = self.sample_temporal_neighbor(
            nids, ts, n_neighbors=hist_len, strategy='recent_edges'
        )
        return hist_nids, hist_eids, hist_ts, hist_dirs
    
    def find_k_hop(self, k: int, nids: np.ndarray, ts: np.ndarray,
                   n_neighbors_list: List[int], alpha: float
                  ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Sampling the k-hop sub graph in tree struture
        """
        if k == 0:
            return ([], [], [])
        self.alpha = alpha
        bs = len(nids)
        root_nids, root_eids, root_ts, _ = self.sample_temporal_neighbor(
            nids, ts, n_neighbors_list[0], strategy='time')
        nids_layers = [root_nids]
        eids_layers = [root_eids]
        ts_layers = [root_ts]
        for i in range(1, k):
            center_nids, _, center_ts = nids_layers[-1], eids_layers[-1], ts_layers[-1]
            center_nids = center_nids.flatten()
            center_ts = center_ts.flatten()
            neigh_nids, neigh_eids, neigh_ts, _ = self.sample_temporal_neighbor(
                center_nids, center_ts, n_neighbors_list[i], strategy='time')
            neigh_nids = neigh_nids.reshape(bs, -1)
            neigh_eids = neigh_eids.reshape(bs, -1)
            neigh_ts = neigh_ts.reshape(bs, -1)

            nids_layers.append(neigh_nids)
            eids_layers.append(neigh_eids)
            ts_layers.append(neigh_ts)

        return (nids_layers, eids_layers, ts_layers)
    
    def sample_walks(self, nids: np.ndarray, ts: np.ndarray, n: int, length: int, alpha: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample random walks starting from `nids` at `ts`.
        -----
        nids: node ids
        ts: timestamps
        n: number of walks
        length: number of nodes in a walk
        alpha: time decay factor
        -----
        Returns: walk_nids, walk_eids, walk_ts
        walk_nids: [bs, n, length] [:,:,-1] == nids
        walk_eids: [bs, n, length] [:,:,-1] == 0
        walk_ts:   [bs, n, length] [:,:,-1] == ts
        """
        self.alpha = alpha
        bs = len(nids)
        walk_nids = np.zeros([bs, n, length], dtype=int)
        walk_eids = np.zeros([bs, n, length], dtype=int)
        walk_ts = np.zeros([bs, n, length], dtype=np.float32)
        last_nids = np.repeat(nids, n)  # [bs x n]
        last_ts = np.repeat(ts, n)
        walk_nids[:, :, -1] = last_nids.reshape(bs, n)
        walk_ts[:, :, -1] = last_ts.reshape(bs, n)
        for j in range(1, length):
            i = length - j - 1
            # [bs x n, 1]
            neigh_nids, neigh_eids, neigh_ts, _ = self.sample_temporal_neighbor(
                last_nids, last_ts, n_neighbors=1, strategy='time')
            walk_nids[:, :, i] = neigh_nids.reshape(bs, n)
            walk_eids[:, :, i] = neigh_eids.reshape(bs, n)
            walk_ts[:, :, i] = neigh_ts.reshape(bs, n)
            last_nids = neigh_nids.squeeze(-1)
            last_ts = neigh_ts.squeeze(-1)
        return walk_nids, walk_eids, walk_ts


def data2adjlist(data: 'InteractionData', max_node_id: Optional[int]=None
                ) -> List[Tuple[int, int, float, bool]]:
    """
    Transform InteractionData into an adjacency list.
    Each node maintains a list in 'adj_list' storing related edges.
    Nodes without edges are with empty lists.
    Edges are in form '(connected_node, edge_index, timestamp, is_src_flag)'
    """
    if max_node_id is None:
        max_node_id = max(max(data.src), max(data.dst))
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src, dst, _, t, eid, _ in data:
        # flag = 0 for null value
        adj_list[src].append((dst, eid, t, 0))  # 0: the target node is src
        adj_list[dst].append((src, eid, t, 1))  # 1: the target node is dst
    return adj_list
