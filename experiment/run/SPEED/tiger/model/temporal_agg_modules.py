from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.data_loader import ComputationGraph
from ..data.graph import Graph
from .basic_modules import MergeLayer
from .feature_getter import NumericalFeature
from .time_encoding import TimeEncode
import math


class EmbeddingModule(nn.Module):

    def __init__(self, raw_feat_getter: NumericalFeature, time_encoder: TimeEncode, graph: Graph):
        super().__init__()
        self.raw_feat_getter = raw_feat_getter
        self.time_encoder = time_encoder
        self.graph = graph

    @property
    def device(self):
        return self.raw_feat_getter.device
    
    def compute_embedding_with_computation_graph(
            self, involved_node_reprs: Tensor, center_nids: Tensor, ts: Tensor,
            computation_graph: ComputationGraph, depth: Optional[int] = None
    ) -> Tensor:
        raise NotImplementedError

    def compute_embedding(
            self, all_node_reprs: Tensor, np_center_nids: np.ndarray, np_ts: np.ndarray,
            depth: Optional[int] = None
    ) -> Tensor:
        raise NotImplementedError


class GraphEmbedding(EmbeddingModule):

    def __init__(self, raw_feat_getter: NumericalFeature, time_encoder: TimeEncode, graph: Graph,
                 n_neighbors=20, n_layers=2):
        super().__init__(raw_feat_getter=raw_feat_getter, time_encoder=time_encoder, graph=graph)
        self.n_neighbors = n_neighbors
        self.n_layers = n_layers

    def compute_embedding_with_computation_graph(
            self, involved_node_reprs: Optional[Tensor], center_nids: Tensor, ts: Tensor,
            computation_graph: ComputationGraph, depth: Optional[int] = None
    ) -> Tensor:
        """
        Compute temporal embeddings h(t-) of center nodes at given timestamps
        using h(t'+) of involved nodes with computation graph
        -----
        involved_node_reprs: involved nodes' representations (memory), could be None
        center_nids: an 1d numpy array of center node ids. shape = [size]
        ts: an 1d numpy array of query timestamps. shape = [size]
        computation_graph: computation graph containing necessary info
        depth: the current depth
        -----
        Return: center_node_reprs
        center_node_reprs: a tensor of center node representations. shape = [size, D]
        """
        depth = self.n_layers if depth is None else depth  # current depth, from n_layers to 0
        # global index -> local index of involved nodes
        local_center_nids = computation_graph.local_index[center_nids]
        if involved_node_reprs is None:
            center_node_reprs = self.raw_feat_getter.get_node_embeddings(center_nids)
        else:
            center_node_reprs = involved_node_reprs[local_center_nids] \
                                + self.raw_feat_getter.get_node_embeddings(center_nids)
        if depth == 0:  # exit condition
            return center_node_reprs  # h(t'+) + static_feat

        _, d = center_node_reprs.shape
        # get neighbors directly from the computation graph
        neigh_nids, neigh_eids, neigh_ts = computation_graph.layers[depth]
        n_center, n_neighbors = neigh_nids.shape

        neigh_reprs = self.compute_embedding_with_computation_graph(
            involved_node_reprs=involved_node_reprs,
            center_nids=neigh_nids.flatten(),
            # ts=neigh_ts.flatten(), 
            ts=torch.repeat_interleave(ts, n_neighbors),  # TGN
            computation_graph=computation_graph,
            depth=depth - 1)

        neigh_reprs = neigh_reprs.reshape(n_center, n_neighbors, d)
        edge_feats = self.raw_feat_getter.get_edge_embeddings(neigh_eids)  # 3D
        delta_ts = ts[:, None] - neigh_ts
        delta_ts_reprs = self.time_encoder(delta_ts)  # 3D
        t0_reprs = self.time_encoder(torch.zeros_like(delta_ts[:, 0]))  # 2D

        # multi-head attention
        center_node_reprs = self.aggregate(depth=depth,
                                           center_x=center_node_reprs,
                                           center_tx=t0_reprs,
                                           neigh_x=neigh_reprs,
                                           edge_x=edge_feats,
                                           edge_tx=delta_ts_reprs,
                                           mask=(neigh_nids == 0)
                                           )

        return center_node_reprs

    def compute_embedding(
            self, all_node_reprs: Optional[Tensor], np_center_nids: np.ndarray, np_ts: np.ndarray,
            depth: Optional[int] = None
    ) -> Tensor:
        """
        Compute temporal embeddings of center nodes at given timestamps.
        -----
        all_node_reprs: a tensor containing ALL nodes' representations, could be None
        center_nids: an 1d numpy array of center node ids. shape = [size]
        ts: an 1d numpy array of query timestamps. shape = [size]
        depth: the current depth
        -----
        Return: center_node_reprs
        center_node_reprs: a tensor of center node representations. shape = [size, D]
        """
        depth = self.n_layers if depth is None else depth  # current depth, from n_layers to 0

        center_nids = torch.from_numpy(np_center_nids).long().to(self.device)

        # temporal representations + static (transformed) features 
        if all_node_reprs is None:
            center_node_reprs = self.raw_feat_getter.get_node_embeddings(center_nids)
        else:
            center_node_reprs = all_node_reprs[center_nids] \
                                + self.raw_feat_getter.get_node_embeddings(center_nids)

        if depth == 0:  # exit condition
            return center_node_reprs  # h(t'+) + static_feat

        _, d = center_node_reprs.shape
        np_neigh_nids, np_neigh_eids, np_neigh_ts, *_ = self.graph.sample_temporal_neighbor(
            np_center_nids, np_ts, self.n_neighbors)  # inputs and outputs are all np.ndarray
        # filter and compress?
        if False:
            # remove all-padding columns. at least 1 column is kept to avoid bugs.
            np_neigh_nids, np_neigh_eids, np_neigh_ts = filter_neighbors(np_neigh_nids, np_neigh_eids, np_neigh_ts)

            n_center, n_neighbors = np_neigh_nids.shape  # maybe n_neighbors < self.n_neighbors for reducing computation

            # reduce repeat computation
            unique_neigh_nids, unique_neigh_ts, np_inverse_idx = compress_node_ts_pairs(
                np_neigh_nids.flatten(), np.repeat(np_ts, n_neighbors)
            )
            neigh_reprs = self.compute_embedding(all_node_reprs=all_node_reprs,
                                                 np_center_nids=unique_neigh_nids,
                                                 np_ts=unique_neigh_ts,
                                                 depth=depth - 1)

            neigh_nids = torch.from_numpy(np_neigh_nids).long().to(self.device)
            neigh_eids = torch.from_numpy(np_neigh_eids).long().to(self.device)
            inverse_idx = torch.from_numpy(np_inverse_idx).long().to(self.device)

            neigh_reprs = neigh_reprs[inverse_idx]  # [n x n_neighbors, d]
        else:
            n_center, n_neighbors = np_neigh_nids.shape
            neigh_reprs = self.compute_embedding(all_node_reprs=all_node_reprs,
                                                 np_center_nids=np_neigh_nids.flatten(),
                                                 # TODO: check this !!!!!!
                                                 # np_ts=np_neigh_ts.flatten(),
                                                 np_ts=np.repeat(np_ts, n_neighbors),  # TGN
                                                 depth=depth - 1)

            neigh_nids = torch.from_numpy(np_neigh_nids).long().to(self.device)
            neigh_eids = torch.from_numpy(np_neigh_eids).long().to(self.device)

        neigh_reprs = neigh_reprs.reshape(n_center, n_neighbors, d)
        edge_feats = self.raw_feat_getter.get_edge_embeddings(neigh_eids)  # 3D
        delta_ts = torch.from_numpy(np_ts[:, None] - np_neigh_ts).float().to(self.device)
        delta_ts_reprs = self.time_encoder(delta_ts)  # 3D
        t0_reprs = self.time_encoder(torch.zeros_like(delta_ts[:, 0]))  # 2D

        mask = neigh_nids == 0  # 2D

        center_node_reprs = self.aggregate(depth=depth,
                                           center_x=center_node_reprs,
                                           center_tx=t0_reprs,
                                           neigh_x=neigh_reprs,
                                           edge_x=edge_feats,
                                           edge_tx=delta_ts_reprs,
                                           mask=mask
                                           )

        return center_node_reprs

    def aggregate(self, depth: int, center_x: Tensor, center_tx: Tensor,
                  neigh_x: Tensor, edge_x: Tensor, edge_tx: Tensor,
                  mask: Tensor) -> Tensor:
        raise NotImplementedError


class GraphAttnEmbedding(GraphEmbedding):
    def __init__(self, raw_feat_getter: NumericalFeature, time_encoder: TimeEncode, graph: Graph, n_neighbors=20,
                 n_layers=2, n_head=2, dropout=0.1):
        super().__init__(raw_feat_getter, time_encoder, graph, n_neighbors, n_layers)
        self.n_head = n_head
        self.dropout = dropout
        self.fns = nn.ModuleList([TemporalAttention(
            nfeat_dim=self.raw_feat_getter.nfeat_dim,
            efeat_dim=self.raw_feat_getter.efeat_dim,
            tfeat_dim=self.time_encoder.dim,
            n_head=self.n_head, dropout=dropout
        ) for _ in range(self.n_layers)]
        )

    def aggregate(self, depth: int, center_x: Tensor, center_tx: Tensor,
                  neigh_x: Tensor, edge_x: Tensor, edge_tx: Tensor,
                  mask: Tensor) -> Tensor:
        fn = self.fns[self.n_layers - depth]
        h = fn(qx=center_x, qt=center_tx,
               kx=neigh_x, ky=edge_x, kt=edge_tx,
               padding_mask=mask)
        return h


class IdentityEmbedding(EmbeddingModule):

    def compute_embedding_with_computation_graph(
            self, involved_node_reprs: Tensor, center_nids: Tensor, ts: Tensor,
            computation_graph: ComputationGraph, depth: Optional[int] = None
    ) -> Tensor:
        """
        Compute temporal embeddings h(t-) of center nodes at given timestamps
        with h(t-) = h(t'+).
        -----
        involved_node_reprs: involved nodes' representations
        center_nids: an 1d numpy array of center node ids. shape = [size]
        ts: (not used) an 1d numpy array of query timestamps. shape = [size]
        computation_graph: computation graph containing necessary info
        depth: (not used) the current depth
        -----
        Return: center_node_reprs
        center_node_reprs: a tensor of center node representations. shape = [size, D]
        """
        local_center_nids = computation_graph.local_index[center_nids]
        # temporal representations only
        center_node_reprs = involved_node_reprs[local_center_nids]
        return center_node_reprs
    
    
    def compute_embedding(
            self, all_node_reprs: Tensor, np_center_nids: np.ndarray, np_ts: np.ndarray,
            depth: Optional[int] = None
        ) -> Tensor:
        center_nids = torch.from_numpy(np_center_nids).long().to(self.device)
        # temporal representations only
        center_node_reprs = all_node_reprs[center_nids]
        return center_node_reprs


class TimeEmbedding(EmbeddingModule):
    def __init__(self, raw_feat_getter: NumericalFeature, time_encoder: TimeEncode, graph: Graph,
                 ts_delta_mean: float, ts_delta_std: float):
        super().__init__(raw_feat_getter, time_encoder, graph)
        self.ts_delta_mean = ts_delta_mean
        self.ts_delta_std = ts_delta_std

        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.raw_feat_getter.nfeat_dim)

    def compute_embedding_with_computation_graph(
            self, involved_node_reprs: Tensor, center_nids: Tensor, ts: Tensor,
            computation_graph: ComputationGraph, depth: Optional[int] = None
    ) -> Tensor:
        """
        Compute temporal embeddings h(t-) of center nodes at given timestamps
        with h(t-) = h(t'+) x (1 + Liner(\delta t)).
        -----
        involved_node_reprs: involved nodes' representations
        center_nids: an 1d numpy array of center node ids. shape = [size]
        ts: an 1d numpy array of query timestamps. shape = [size]
        computation_graph: computation graph containing necessary info
        depth: the current depth
        -----
        Return: center_node_reprs
        center_node_reprs: a tensor of center node representations. shape = [size, D]
        """
        local_center_nids = computation_graph.local_index[center_nids]
        _1, _2, neigh_ts = computation_graph.layers[1]
        if (neigh_ts.size(1) != 1) or (len(computation_graph.layers) != 2):
            raise ValueError('Please set n_neighbors=1 and n_layers=1 for JODIE.')
        ts_deltas = ts - neigh_ts[:, -1]
        ts_deltas = (ts_deltas - self.ts_delta_mean) / self.ts_delta_std
        center_node_reprs = involved_node_reprs[local_center_nids] \
                            * (1. + self.embedding_layer(ts_deltas.unsqueeze(1)))
        
        return center_node_reprs

    def compute_embedding(
            self, all_node_reprs: Tensor, np_center_nids: np.ndarray, np_ts: np.ndarray,
            depth: Optional[int] = None
    ) -> Tensor:
        center_nids = torch.from_numpy(np_center_nids).long().to(self.device)
        center_ts = torch.from_numpy(np_ts).long().to(self.device)
        _, _, prev_ts, _ = self.graph.get_history(np_center_nids, np_ts, 1)
        last_ts = torch.LongTensor(prev_ts).to(self.device)
        ts_deltas = center_ts - last_ts
        ts_deltas = (ts_deltas - self.ts_delta_mean) / self.ts_delta_std
        center_node_reprs = all_node_reprs[center_nids] \
                            * (1. + self.embedding_layer(ts_deltas.unsqueeze(1)))
        return center_node_reprs


class TemporalAttention(nn.Module):
    def __init__(self, nfeat_dim, efeat_dim, tfeat_dim, n_head=2, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.dropout = dropout
        self.query_dim = nfeat_dim + tfeat_dim
        self.key_dim = nfeat_dim + efeat_dim + tfeat_dim
        self.merger = MergeLayer(self.query_dim, nfeat_dim, nfeat_dim, nfeat_dim)
        self.mha_fn = nn.MultiheadAttention(embed_dim=self.query_dim,
                                            num_heads=self.n_head,
                                            dropout=self.dropout,
                                            kdim=self.key_dim,
                                            vdim=self.key_dim)

    def forward(self, qx: Tensor, qt: Tensor, kx: Tensor, ky: Tensor, kt: Tensor, padding_mask: Tensor) -> Tensor:
        """
        Temporal Multihead Attention
        -----
        qx: [n, d1] query node features
        qt: [n, d3] query time features
        kx: [n, len, d1] key node features
        ky: [n, len, d2] key edge features
        kt: [n, len, d3] key time features
        padding_mask: [n, len] True indicates its a padding token
        """
        query = torch.cat([qx, qt], 1).unsqueeze(0)  # [1, n, d]
        kv = torch.cat([kx, ky, kt], 2).transpose(0, 1)  # [len, n, d]

        invalid_rows = padding_mask.bool().all(1, keepdims=True)  # [n, 1]
        padding_mask[invalid_rows.squeeze(1), -1] = False  # NB: to avoid NaN

        h, _ = self.mha_fn(query, kv, kv, key_padding_mask=padding_mask)
        h = h.squeeze(0)  # [n, query_dim]

        # if a node has no neighbors, we set its repsentation to zero
        h = h.masked_fill(invalid_rows, 0.)  # fill the entire rows

        # concat two vectors and pass it thru a 2layer MLP
        z = self.merger(h, qx)
        return z


def filter_neighbors(ngh_nids: np.ndarray, ngh_eids: np.ndarray, ngh_ts: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop columns whose elements are all padding tokens.
    """
    col_mask = ~np.all(ngh_nids == 0, 0)  # if entire col is null, drop the col.
    col_mask[-1] = True  # at least have one (null) neighbor to aviod bugs
    ngh_nids = ngh_nids[:, col_mask]
    ngh_eids = ngh_eids[:, col_mask]
    ngh_ts = ngh_ts[:, col_mask]
    return ngh_nids, ngh_eids, ngh_ts


def compress_node_ts_pairs(nids: np.ndarray, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deduplicate node-ts pairs to reduce computations.
    """
    codes = np.stack([nids, ts])  # [2, layer_size]
    _, unique_idx, inverse_idx = np.unique(codes, axis=1, return_index=True, return_inverse=True)
    unique_nids = nids[unique_idx]
    unique_ts = ts[unique_idx]
    return unique_nids, unique_ts, inverse_idx
