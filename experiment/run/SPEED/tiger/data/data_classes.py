from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


class RestartData:

    def to(self, device: torch.device):
        raise NotImplementedError

    def pin_memory(self):
        raise NotImplementedError


class DummyRestartData(RestartData):
    def to(self, device: torch.device):
        return

    def pin_memory(self):
        return 
    

@dataclass
class SeqRestartData(RestartData):
    index: Tensor
    nids: Tensor
    ts: Tensor
    hist_nids: Tensor
    anonymized_ids: Tensor
    hist_eids: Tensor
    hist_ts: Tensor
    hist_dirs: Tensor

    def __iter__(self):
        fields = [self.index, self.nids, self.ts, self.hist_nids,
                  self.anonymized_ids, self.hist_eids,
                  self.hist_ts, self.hist_dirs]
        for x in fields:
            yield x
    
    def to(self, device: torch.device):
        self.index = self.index.to(device)
        self.nids = self.nids.to(device)
        self.ts = self.ts.to(device)
        self.hist_nids = self.hist_nids.to(device)
        self.anonymized_ids = self.anonymized_ids.to(device)
        self.hist_eids = self.hist_eids.to(device)
        self.hist_ts = self.hist_ts.to(device)
        self.hist_dirs = self.hist_dirs.to(device)

    def pin_memory(self):
        self.index = self.index.pin_memory()
        self.nids = self.nids.pin_memory()
        self.ts = self.ts.pin_memory()
        self.hist_nids = self.hist_nids.pin_memory()
        self.anonymized_ids = self.anonymized_ids.pin_memory()
        self.hist_eids = self.hist_eids.pin_memory()
        self.hist_ts = self.hist_ts.pin_memory()
        self.hist_dirs = self.hist_dirs.pin_memory()


@dataclass
class WalkRestartData(RestartData):
    index: Tensor
    nids: Tensor
    ts: Tensor
    prev_srcs: Tensor
    prev_dsts: Tensor
    prev_eids: Tensor
    walk_nids: Tensor  # [bs, n_walks, length]
    walk_anonymized_codes: Tensor
    walk_eids: Tensor
    walk_ts: Tensor
    prev_srcs_codes: Tensor
    prev_dsts_codes: Tensor

    def to(self, device: torch.device):
        self.index = self.index.to(device)
        self.nids = self.nids.to(device)
        self.ts = self.ts.to(device)
        self.prev_srcs = self.prev_srcs.to(device)
        self.prev_dsts = self.prev_dsts.to(device)
        self.prev_eids = self.prev_eids.to(device)
        self.walk_nids = self.walk_nids.to(device)
        self.walk_anonymized_codes = self.walk_anonymized_codes.to(device)
        self.walk_eids = self.walk_eids.to(device)
        self.walk_ts = self.walk_ts.to(device)
        self.prev_srcs_codes = self.prev_srcs_codes.to(device)
        self.prev_dsts_codes = self.prev_dsts_codes.to(device)
    
    def pin_memory(self):
        self.index = self.index.pin_memory()
        self.nids = self.nids.pin_memory()
        self.ts = self.ts.pin_memory()
        self.prev_srcs = self.prev_srcs.pin_memory()
        self.prev_dsts = self.prev_dsts.pin_memory()
        self.prev_eids = self.prev_eids.pin_memory()
        self.walk_nids = self.walk_nids.pin_memory()
        self.walk_anonymized_codes = self.walk_anonymized_codes.pin_memory()
        self.walk_eids = self.walk_eids.pin_memory()
        self.walk_ts = self.walk_ts.pin_memory()
        self.prev_srcs_codes = self.prev_srcs_codes.pin_memory()
        self.prev_dsts_codes = self.prev_dsts_codes.pin_memory()


@dataclass
class StaticRestartData(RestartData):
    index: Tensor
    nids: Tensor
    ts: Tensor
    prev_ts: Tensor

    def __iter__(self):
        fields = [self.index, self.nids, self.ts, self.prev_ts]
        for x in fields:
            yield x
    
    def to(self, device: torch.device):
        self.index = self.index.to(device)
        self.nids = self.nids.to(device)
        self.ts = self.ts.to(device)
        self.prev_ts = self.prev_ts.to(device)

    def pin_memory(self):
        self.index = self.index.pin_memory()
        self.nids = self.nids.pin_memory()
        self.ts = self.ts.pin_memory()
        self.prev_ts = self.prev_ts.pin_memory()
    
@dataclass
class HitData:
    src_hits: Tensor
    dst_hits: Tensor
    neg_src_hits: Tensor
    neg_dst_hits: Tensor

    def __iter__(self):
        fields = [self.src_hits, self.dst_hits, self.neg_src_hits, self.neg_dst_hits]
        for x in fields:
            yield x
    
    def to(self, device: torch.device):
        self.src_hits = self.src_hits.to(device)
        self.dst_hits = self.dst_hits.to(device)
        self.neg_src_hits = self.neg_src_hits.to(device)
        self.neg_dst_hits = self.neg_dst_hits.to(device)

    def pin_memory(self):
        self.src_hits = self.src_hits.pin_memory()
        self.dst_hits = self.dst_hits.pin_memory()
        self.neg_src_hits = self.neg_src_hits.pin_memory()
        self.neg_dst_hits = self.neg_dst_hits.pin_memory()


class ComputationGraph:

    def __init__(self, tige_data: Tuple[List[Tuple], Tensor], 
                 restart_data: RestartData,
                 hit_data: HitData,
                 n_nodes: int
                ):
        self.n_nodes = n_nodes
        self.layers = tige_data[0]
        self.np_computation_graph_nodes = tige_data[1]  # np.ndarray
        self.computation_graph_nodes = torch.from_numpy(tige_data[1])
        self.restart_data = restart_data
        self.hit_data = hit_data
        self.local_index = torch.zeros(self.n_nodes, dtype=torch.long)
        self.local_index[self.computation_graph_nodes] = torch.arange(
            len(self.computation_graph_nodes))
    
    @property
    def device(self):
        return self.computation_graph_nodes.device
    
    def to(self, device: torch.device):
        for depth in range(len(self.layers)):
            neigh_nids, neigh_eids, neigh_ts = self.layers[depth]
            neigh_nids = neigh_nids.to(device)
            if depth > 0:
                neigh_eids = neigh_eids.to(device)
                neigh_ts = neigh_ts.to(device)
            self.layers[depth] = (neigh_nids, neigh_eids, neigh_ts)
        self.computation_graph_nodes = self.computation_graph_nodes.to(device)
        self.local_index = self.local_index.to(device)
        self.restart_data.to(device)
        self.hit_data.to(device)
        return self

    def pin_memory(self):
        for depth in range(len(self.layers)):
            neigh_nids, neigh_eids, neigh_ts = self.layers[depth]
            neigh_nids = neigh_nids.pin_memory()
            if depth > 0:
                neigh_eids = neigh_eids.pin_memory()
                neigh_ts = neigh_ts.pin_memory()
            self.layers[depth] = (neigh_nids, neigh_eids, neigh_ts)
        self.computation_graph_nodes = self.computation_graph_nodes.pin_memory()
        self.local_index = self.local_index.pin_memory()
        self.restart_data.pin_memory()
        self.hit_data.pin_memory()
        return self
