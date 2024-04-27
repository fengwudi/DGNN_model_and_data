from typing import Optional, Tuple, Union
import warnings

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.data_loader import ComputationGraph
from ..data.graph import Graph
from .basic_modules import MergeLayer
from .feature_getter import FeatureGetter
from .time_encoding import TimeEncode
from .utils import anonymized_reindex, set_anonymized_encoding


class Restarter(nn.Module):
    def __init__(self, raw_feat_getter: FeatureGetter, graph: Graph):
        super().__init__()
        self.raw_feat_getter = raw_feat_getter
        self.graph = graph

        self.n_nodes = self.raw_feat_getter.n_nodes
        self.nfeat_dim = self.raw_feat_getter.nfeat_dim
        self.efeat_dim = self.raw_feat_getter.efeat_dim

        self.time_encoder = TimeEncode(dim=self.nfeat_dim)
        self.tfeat_dim = self.time_encoder.dim

    def forward(self, nids: Tensor, ts: Tensor,
                computation_graph: Optional[ComputationGraph]=None
               ) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


class SeqRestarter(Restarter):
    def __init__(self, raw_feat_getter: FeatureGetter, graph: Graph,
                 *, hist_len: int=20, n_head=2, dropout=0.1):
        super().__init__(raw_feat_getter, graph)

        self.hist_len = hist_len

        self.anony_emb = nn.Embedding(self.hist_len + 1, self.nfeat_dim)

        self.d_model = self.nfeat_dim * 3 + self.efeat_dim + self.tfeat_dim
        self.mha_fn = nn.MultiheadAttention(self.d_model, n_head, dropout)
        self.out_fn = nn.Linear(self.d_model, self.nfeat_dim)
        self.merger = MergeLayer(self.nfeat_dim, self.d_model - self.tfeat_dim, 
                                 self.nfeat_dim, self.nfeat_dim, dropout=dropout)

    def forward(self, nids: Tensor, ts: Tensor,
                computation_graph: Optional[ComputationGraph]=None
               ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute surrogate representations h(t'-) and h(t'+).
        -----
        nids: node ids
        ts: the current timestamps t
        computation_graph: computation graph containing necessary information
                           This is only given during training.
        -----
        returns: h_prev_left, h_prev_right, prev_ts
        h_prev_left: h(t'-)
        h_prev_right: h(t'+)
        prev_ts: t'
        """
        if computation_graph is None:
            device = nids.device
            hist_nids, hist_eids, hist_ts, hist_dirs = self.graph.get_history(
                nids.cpu().numpy(), ts.cpu().numpy(), self.hist_len)
            anonymized_ids = anonymized_reindex(hist_nids)
            hist_nids = torch.from_numpy(hist_nids).to(device).long()  # [bs, len]
            anonymized_ids = torch.from_numpy(anonymized_ids).to(device).long()
            hist_eids = torch.from_numpy(hist_eids).to(device).long()
            hist_ts = torch.from_numpy(hist_ts).to(device).float()
            hist_dirs = torch.from_numpy(hist_dirs).to(device).long()
        else:
            data = computation_graph.restart_data
            hist_nids = data.hist_nids
            anonymized_ids = data.anonymized_ids
            hist_eids = data.hist_eids
            hist_ts = data.hist_ts
            hist_dirs = data.hist_dirs

        bs, hist_len = hist_nids.shape
        mask = (hist_nids == 0)  # [bs, len]
        mask[:, -1] = False  # to avoid bugs
        invalid_rows = mask.all(1, keepdims=True)  # [n, 1]

        # event reprs = [src, dst, edge, anony, ts]
        # dirs is used to determine if the current node is src or dst
        r_nids = nids.unsqueeze(1).repeat(1, hist_len)
        src_nids = r_nids * hist_dirs + hist_nids * (1-hist_dirs)
        dst_nids = r_nids * (1-hist_dirs) + hist_nids * hist_dirs

        src_vals = self.raw_feat_getter.get_node_embeddings(src_nids)
        dst_vals = self.raw_feat_getter.get_node_embeddings(dst_nids)
        edge_vals = self.raw_feat_getter.get_edge_embeddings(hist_eids)
        anony_vals = self.anony_emb(anonymized_ids)
        ts_vals = self.time_encoder(hist_ts[:, -1].unsqueeze(1) - hist_ts)
        full_vals = torch.cat([src_vals, dst_vals, anony_vals, edge_vals, ts_vals], 2)  # [bs, len, D]

        last_event_feat = full_vals[:, -1, :self.d_model - self.tfeat_dim]
        full_vals[:, -1, :self.d_model - self.tfeat_dim] = 0.  # only keep time feats
        qkv = full_vals.transpose(0, 1)  # [len, bs, D]
        out, _ = self.mha_fn(qkv, qkv, qkv, key_padding_mask=mask)
        # h(t'-)
        h_prev_left = self.out_fn(F.relu(out.mean(0)))  # [bs, D]  mean aggregate
        # h_prev = self.out_fn(F.relu(out[-1, :, :]))  # [bs, D] last
        h_prev_right = self.merger(h_prev_left, last_event_feat)  # h(t'+)
        h_prev_left = h_prev_left.masked_fill(invalid_rows, 0.)
        h_prev_right = h_prev_right.masked_fill(invalid_rows, 0.)
        prev_ts = hist_ts[:, -1]
        return h_prev_left, h_prev_right, prev_ts


class WalkRestarter(Restarter):
    def __init__(self, raw_feat_getter: FeatureGetter, graph: Graph,
                 *, n_walks: int=20, walk_length: int=5, alpha=1e-5,
                 n_head=2, dropout=0.1):
        super().__init__(raw_feat_getter, graph)
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.alpha = alpha
        self.n_head = n_head
        self.dropout = dropout

        self.anony_emb = nn.Sequential(
            nn.Linear(walk_length, self.nfeat_dim),
            nn.ReLU(),
            nn.Linear(self.nfeat_dim, self.nfeat_dim)
        )

        self.d_model = self.nfeat_dim * 2 + self.efeat_dim + self.tfeat_dim
        self.d_last_edge = self.nfeat_dim * 4 + self.efeat_dim
        self.seq_mha_fn = nn.MultiheadAttention(self.d_model, n_head, dropout)
        self.agg_mha_fn = nn.MultiheadAttention(self.d_model, n_head, dropout)
        self.out_fn = nn.Linear(self.d_model, self.nfeat_dim)
        self.merger = MergeLayer(self.nfeat_dim, self.d_last_edge, 
                                 self.nfeat_dim, self.nfeat_dim, dropout=dropout)

    def forward(self, nids: Tensor, ts: Tensor,
                computation_graph: Optional[ComputationGraph]=None
               ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute surrogate representations h(t'-) and h(t'+).
        -----
        nids: node ids
        ts: the current timestamps t
        computation_graph: computation graph containing necessary information
                           This is only given during training.
        -----
        returns: h_prev_left, h_prev_right, prev_ts
        h_prev_left: h(t'-)
        h_prev_right: h(t'+)
        prev_ts: t'
        """
        if computation_graph is None:
            device = nids.device
            np_nids = nids.cpu().numpy()
            np_ts = ts.cpu().numpy()
            prev_neighbors, prev_eids, prev_ts, prev_dirs = (
                x.squeeze(1) for x in self.graph.get_history(np_nids, np_ts, 1))
            walk_nids, walk_eids, walk_ts = self.graph.sample_walks(
                np_nids, prev_ts, self.n_walks, self.walk_length, self.alpha
            )
            prev_srcs = (1-prev_dirs) * np_nids + prev_dirs * prev_neighbors
            prev_dsts = prev_dirs * np_nids + (1-prev_dirs) * prev_neighbors
            prev_srcs[prev_neighbors == 0] = 0
            prev_dsts[prev_neighbors == 0] = 0

            walk_anonymized_codes, id2code_dicts = set_anonymized_encoding(walk_nids)

            prev_srcs_codes = np.zeros((len(nids), self.walk_length))
            prev_dsts_codes = np.zeros((len(nids), self.walk_length))
            for i in range(len(nids)):
                prev_srcs_codes[i] = id2code_dicts[i].get(prev_srcs[i], 0)
                prev_dsts_codes[i] = id2code_dicts[i].get(prev_dsts[i], 0)

            prev_srcs = torch.from_numpy(prev_srcs).long().to(device)
            prev_dsts = torch.from_numpy(prev_dsts).long().to(device)
            prev_eids = torch.from_numpy(prev_eids).long().to(device)
            walk_nids = torch.from_numpy(walk_nids).long().to(device)
            walk_anonymized_codes = torch.from_numpy(walk_anonymized_codes).float().to(device)
            walk_eids = torch.from_numpy(walk_eids).long().to(device)
            walk_ts = torch.from_numpy(walk_ts).float().to(device)
            prev_srcs_codes = torch.from_numpy(prev_srcs_codes).float().to(device)
            prev_dsts_codes = torch.from_numpy(prev_dsts_codes).float().to(device)
        else:
            data = computation_graph.restart_data
            prev_srcs = data.prev_srcs
            prev_dsts = data.prev_dsts
            prev_srcs_codes = data.prev_srcs_codes
            prev_dsts_codes = data.prev_dsts_codes
            prev_eids = data.prev_eids
            walk_nids = data.walk_nids
            walk_anonymized_codes = data.walk_anonymized_codes
            walk_eids = data.walk_eids
            walk_ts = data.walk_ts

        bs, n_walks, walk_length = walk_nids.shape

        prev_ts = walk_ts[:, 0, -1]  # walks of a node have the same starting timestamp, i.e., t-.

        # let n = bs * n_walks
        walk_nids = walk_nids.reshape(bs * n_walks, walk_length)
        walk_eids = walk_eids.reshape(bs * n_walks, walk_length)
        walk_ts = walk_ts.reshape(bs * n_walks, walk_length)
        walk_anonymized_codes = walk_anonymized_codes.reshape(bs * n_walks, walk_length, walk_length)

        node_vals = self.raw_feat_getter.get_node_embeddings(walk_nids)
        edge_vals = self.raw_feat_getter.get_edge_embeddings(walk_eids)
        anony_vals = self.anony_emb(walk_anonymized_codes)
        ts_vals = self.time_encoder(walk_ts[:, -1:] - walk_ts)

        full_vals = torch.cat([node_vals, edge_vals, anony_vals, ts_vals], 2)  # [n, length, D]
        mask = (walk_nids == 0)  # [n, length]
        mask[:, -1] = False  # to avoid bugs

        # encode walks
        qkv = full_vals.transpose(0, 1)  # [length, n, D]
        walk_reprs, _ = self.seq_mha_fn(qkv, qkv, qkv, key_padding_mask=mask)
        # aggregate walks
        walk_reprs = walk_reprs.mean(0).reshape(bs, n_walks, self.d_model).transpose(0, 1)  # [n_walks, bs, D]
        agg_reprs, _ = self.agg_mha_fn(walk_reprs, walk_reprs, walk_reprs)
        agg_reprs = agg_reprs.mean(0)  # [bs, D]

        h_prev_left = self.out_fn(F.relu(agg_reprs))  # [bs, nfeat_dim]

        last_event_feat = self.get_edge_reprs(prev_srcs, prev_dsts,
                                              prev_srcs_codes, prev_dsts_codes, 
                                              prev_eids)

        h_prev_right = self.merger(h_prev_left, last_event_feat)  # h(t'+)
        invalid_rows = (prev_srcs == 0).unsqueeze(1)  # [n, 1]
        h_prev_left = h_prev_left.masked_fill(invalid_rows, 0.)
        h_prev_right = h_prev_right.masked_fill(invalid_rows, 0.)

        return h_prev_left, h_prev_right, prev_ts

    def get_edge_reprs(self, srcs, dsts, srcs_codes, dsts_codes, eids):
        bs = len(srcs)
        nfeats = self.raw_feat_getter.get_node_embeddings(
            torch.stack([srcs, dsts], dim=1)  # [bs, 2]
        ).reshape(bs, 2 * self.nfeat_dim)  # [bs, 2 * d_n]
        efeats = self.raw_feat_getter.get_edge_embeddings(eids)  # [bs, d_e]
        anony_codes = self.anony_emb(
            torch.stack([srcs_codes, dsts_codes], dim=1)  # [bs, 2, length]
        ).reshape(bs, 2 * self.nfeat_dim)
        full_reprs = torch.cat([nfeats, efeats, anony_codes], dim=1)
        return full_reprs


class StaticRestarter(Restarter):
    def __init__(self, raw_feat_getter: FeatureGetter, graph: Graph):
        super().__init__(raw_feat_getter, graph)
        self.left_emb = nn.Embedding(self.n_nodes, self.nfeat_dim)
        self.right_emb = nn.Embedding(self.n_nodes, self.nfeat_dim)
        nn.init.zeros_(self.left_emb.weight)
        nn.init.zeros_(self.right_emb.weight)
    
    def forward(self, nids: Tensor, ts: Tensor, 
                computation_graph: Optional[ComputationGraph]=None
               ) -> Tuple[Tensor, Tensor, Tensor]:
        if computation_graph is None:
            device = nids.device
            _, _, prev_ts, _ = self.graph.get_history(
                nids.cpu().numpy(), ts.cpu().numpy(), 1)
            prev_ts = prev_ts[:, 0]
            prev_ts = torch.from_numpy(prev_ts).to(device).float()
        else:
            data = computation_graph.restart_data
            prev_ts = data.prev_ts
        h_left = self.left_emb(nids)
        h_right = self.right_emb(nids)

        return h_left, h_right, prev_ts
