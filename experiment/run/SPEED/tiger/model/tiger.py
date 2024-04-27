import math
from typing import Optional, Tuple, Union
import warnings

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.data_loader import ComputationGraph
from ..data.graph import Graph
from .basic_modules import MergeLayer
from .feature_getter import FeatureGetter, NumericalFeature
from .memory import (Memory, MessageStore, MessageStoreNoGrad,
                     MessageStoreNoGradLastOnly)
from .message_modules import (IdentityMessageFunction, LastMessageAggregator,
                              LastMessageAggregatorNoGrad,
                              LastMessageAggregatorNoGradLastOnly,
                              LinearMessageFunction, MLPMessageFunction)
from .temporal_agg_modules import GraphAttnEmbedding, IdentityEmbedding, TimeEmbedding
from .time_encoding import TimeEncode
from .update_modules import GRUUpdater, MergeUpdater, RNNUpdater, IdentityUpdater
from .utils import select_latest_nids
from .restarters import Restarter


class TIGE(nn.Module):
    def __init__(self, *, raw_feat_getter: FeatureGetter, graph: Graph,
                 n_neighbors: int = 20, n_layers: int = 2, n_head: int = 2, dropout: float = 0.1,
                 msg_src: str, upd_src: str,
                 msg_tsfm_type: str = 'id', mem_update_type: str = 'gru',
                 tgn_mode: bool = True, msg_last_only: bool = True,
                 hit_type: str = 'none',
                 embedding_type: str = 'att',
                 dyrep: bool = False,
                 no_memory: bool = False,
                 ts_delta_mean: float = 0., ts_delta_std: float = 1.,
                 n_shared_nodes: int = 0, static_shared_nodes: bool = False
                 ):
        """
        Temporal Interaction Graph Embedding Model (without restart)
        -----
        raw_feat_getter: feature getter
        graph: train graph or full graph
        n_neighbors: number of neighbors to sample
        n_layers: number of temporal aggregation layers
        n_head: number of heads in MultiHeadAttention
        dropout: dropout ratio
        msg_src: [left, right] using h(t'-) or h(t''+) to compute memory
        upd_src: [left, right] using h(t'-) or h(t''+) as update input
        msg_tsfm_type: [id, linear, mlp] message transformation function
        mem_update_type: [gru, merge, rnn] memory update function
        tgn_mode: if true, message-related parts cannot receive gradients
                  (except msg_tfsm_fn)
        msg_last_only: if true, only the last messages are kept to obtain speed-up
        hit_type: [vec, bin, none] use hit info when computing edge scores
        embedding_type: [att, id, time] embedding modules described in TGN
        dyrep: if true, use the DyRep mode
               (use h(t'+) as the output, use h(t-) as the part of message for dst nodes)
        no_memory: if true, skip all memory-related modules (TGAT mode)
        ts_delta_mean: mean of differences of timestamps
        ts_delta_std: std of differences of timestamps
        """
        super().__init__()

        self.raw_feat_getter = raw_feat_getter
        self.n_nodes = self.raw_feat_getter.n_nodes
        self.nfeat_dim = self.raw_feat_getter.nfeat_dim
        self.efeat_dim = self.raw_feat_getter.efeat_dim

        self.time_encoder = TimeEncode(dim=self.nfeat_dim)
        # For simplicity, nfeat_dim = tfeat_dim = memory_dim
        self.tfeat_dim = self.time_encoder.dim
        self.memory_dim = self.nfeat_dim
        self.raw_msg_dim = self.memory_dim * 2 + self.efeat_dim + self.tfeat_dim
        self.msg_dim = None  # defined later

        self.n_neighbors = n_neighbors
        self.n_layers = n_layers

        self.msg_src = msg_src
        self.upd_src = upd_src
        self.embedding_type = embedding_type

        # if msg_last_only: True -> tgn_mode: True
        self.tgn_mode = True if msg_last_only else tgn_mode
        self.dyrep = dyrep
        self.no_memory = no_memory

        self.msg_last_only = msg_last_only

        self.n_shared_nodes = n_shared_nodes

        self.static_shared_nodes = static_shared_nodes

        # init memory
        # if self.n_shared_nodes > 0:
        #     # 把0号节点也当作是Embedding计算
        #     self.left_shared_memory = nn.Embedding(self.n_shared_nodes+1, self.memory_dim)
        #     self.right_shared_memory = nn.Embedding(self.n_shared_nodes+1, self.memory_dim)
        # else:
        #     self.left_shared_memory = None
        #     self.right_shared_memory = None
        self.left_memory = Memory(self.n_nodes, self.memory_dim, self.n_shared_nodes, self.static_shared_nodes)
        self.right_memory = Memory(self.n_nodes, self.memory_dim, self.n_shared_nodes, self.static_shared_nodes)

        # Message Store
        if self.msg_last_only:
            # Use this!!! Fastest!!!
            self.msg_store = MessageStoreNoGradLastOnly(self.n_nodes, dim=self.raw_msg_dim)
        elif self.tgn_mode:
            self.msg_store = MessageStoreNoGrad(self.n_nodes)  # TGN Version
        else:
            self.msg_store = MessageStore(self.n_nodes)
        # set up message/update sources
        self.msg_memory = self.left_memory if self.msg_src == 'left' else self.right_memory
        self.upd_memory = self.left_memory if self.upd_src == 'left' else self.right_memory

        # Message Aggregator
        if self.msg_last_only:
            # Use this!!! Fastest!!!
            self.msg_aggregate_fn = LastMessageAggregatorNoGradLastOnly(
                raw_feat_getter=self.raw_feat_getter,
                time_encoder=self.time_encoder,
            )
        elif self.tgn_mode:  # TGN Version
            self.msg_aggregate_fn = LastMessageAggregatorNoGrad(
                raw_feat_getter=self.raw_feat_getter,
                time_encoder=self.time_encoder,
            )
        else:
            self.msg_aggregate_fn = LastMessageAggregator(
                raw_feat_getter=self.raw_feat_getter,
                time_encoder=self.time_encoder,
            )

        # Message Transformation
        if msg_tsfm_type == 'id':
            self.msg_transform_fn = IdentityMessageFunction(raw_msg_dim=self.raw_msg_dim)
        elif msg_tsfm_type == 'linear':
            self.msg_transform_fn = LinearMessageFunction(raw_msg_dim=self.raw_msg_dim)
        elif msg_tsfm_type == 'mlp':
            self.msg_transform_fn = MLPMessageFunction(raw_msg_dim=self.raw_msg_dim)
        else:
            raise NotImplementedError
        self.msg_dim = self.msg_transform_fn.output_size

        # Updater
        if mem_update_type == 'gru':
            self.right_mem_updater = GRUUpdater(self.msg_dim, self.memory_dim)
        elif mem_update_type == 'merge':
            self.right_mem_updater = MergeUpdater(self.msg_dim, self.memory_dim)
        elif mem_update_type == 'rnn':
            self.right_mem_updater = RNNUpdater(self.msg_dim, self.memory_dim)
        elif mem_update_type == 'id':
            self.right_mem_updater = IdentityUpdater(self.msg_dim, self.memory_dim)
        else:
            raise NotImplementedError

        # Temporal embedding
        if self.embedding_type == 'att':
            self.temporal_embedding_fn = GraphAttnEmbedding(
                raw_feat_getter=self.raw_feat_getter,
                time_encoder=self.time_encoder, graph=graph,
                n_neighbors=n_neighbors, n_layers=n_layers, n_head=n_head,
                dropout=dropout
            )
        elif self.embedding_type == 'id':
            self.temporal_embedding_fn = IdentityEmbedding(
                raw_feat_getter=self.raw_feat_getter,
                time_encoder=self.time_encoder, graph=graph,
            )
        elif self.embedding_type == 'time':
            self.temporal_embedding_fn = TimeEmbedding(
                raw_feat_getter=self.raw_feat_getter,
                time_encoder=self.time_encoder, graph=graph,
                ts_delta_mean=ts_delta_mean, ts_delta_std=ts_delta_std
            )
        else:
            raise NotImplementedError

        # Score function
        self.hit_type = hit_type
        if self.hit_type == 'vec':
            merge_dim = self.nfeat_dim + self.n_neighbors
        elif self.hit_type == 'bin':
            self.hit_embedding = nn.Embedding(2, self.nfeat_dim)
            merge_dim = self.nfeat_dim
        elif self.hit_type == 'count':
            self.hit_embedding = nn.Embedding(self.n_neighbors + 1, self.nfeat_dim)
            merge_dim = self.nfeat_dim
        else:
            merge_dim = self.nfeat_dim

        self.score_fn = MergeLayer(merge_dim, merge_dim,
                                   self.nfeat_dim, 1, dropout=dropout)

        # Loss function
        self.contrast_loss_fn = nn.BCEWithLogitsLoss()

        self._sanity_check()

    def _sanity_check(self):
        if self.msg_src not in {'left', 'right'}:
            raise ValueError(f'Invalid msg_src={self.msg_src}')
        if self.upd_src not in {'left', 'right'}:
            raise ValueError(f'Invalid upd_src={self.msg_src}')

    @property
    def graph(self):
        return self.temporal_embedding_fn.graph

    @graph.setter
    def graph(self, new_obj: Graph):
        self.temporal_embedding_fn.graph = new_obj

    @property
    def device(self):
        return self.msg_memory.device

    def contrast_learning(self, src_ids: Tensor, dst_ids: Tensor, neg_dst_ids: Tensor,
                          ts: Tensor, eids: Tensor,
                          computation_graph: ComputationGraph
                          ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward with negative dst nodes.
        -----
        src_ids          : src node ids
        dst_ids          : dst node ids
        neg_dst_ids      : negative dst node ids
        ts               : timestamps of events (t)
        eids             : edge ids
        computation_graph: computation graph containing necessary info
        -----
        returns: contrast_loss, h_return, pos_scores, neg_scores, h_prev_left, h_prev_right
        contrast_loss: contrastive loss
        h_return     : h(t-) of src and dst nodes; if dyrep=True, this would be h(t'+)
        pos_scores   : scores of positive edges (without sigmoid)
        neg_scores   : scores of negative edges (without sigmoid)
        h_prev_left  : h(t'-) of src and dst nodes (for restarter)
        h_prev_right : h(t'+) of src and dst nodes (for restarter)
        """
        if self.no_memory:
            return self.contrast_learning_no_memory(src_ids, dst_ids, neg_dst_ids,
                                                    ts, eids, computation_graph)

        bs = len(src_ids)
        pos_node_ids = torch.cat([src_ids, dst_ids])  # we only update states of these nodes
        batch_node_ids = torch.cat([src_ids, dst_ids, neg_dst_ids])

        # NB: Since any node can be involved in computing temporal embedding
        #     especially with graph attention, we need to update almost all nodes
        #  We can detect involved nodes in advance to reduce redundant computation.
        #  This is useful if 1. temporal embedding module is simple, e.g., identity;
        #  2. some nodes remain untouched for a very long time (dataset-specific).

        # STEP 1: compute h(t'+) for all nodes (involved node) that have pending messages
        #         using h(t'-) and/or h(t''+)
        outdated_nids, msgs, prev_ts = self.compute_messages(
            computation_graph.np_computation_graph_nodes)  # use np to avoid cpu <-> gpu copy

        # STEP 2: Get h(t'+) of involved nodes without actually updating right memory,
        #         since we should only update states of positive nodes.
        involved_node_ids = computation_graph.computation_graph_nodes
        involved_node_reprs = self.right_memory.vals[involved_node_ids].clone()
        if len(outdated_nids):
            h_prev_right_all = self.apply_messages(outdated_nids, msgs, prev_ts)  # h(t'+)
            # |outdated_nids| <= |involved_node_ids|
            # find indices of matching rows
            involved_local_index, outdated_local_index = torch.where(
                involved_node_ids[:, None] == outdated_nids)
            involved_node_reprs[involved_local_index] = h_prev_right_all[outdated_local_index]

        # STEP 3: compute temporal embeddings h(t-) using h(t'+)
        # NB: ts.repeat(3) = torch.cat([ts, ts, ts])
        h_left_with_negs = self.compute_temporal_embedding_with_involved_nodes_only(
            involved_node_reprs, batch_node_ids, ts.repeat(3), computation_graph
        )

        # STEP 4: update right memory in-place (for postive nodes only) (no grad)
        if len(outdated_nids):
            # deduplicate
            unique_pos_ids, _ = select_latest_nids(pos_node_ids, ts.repeat(2))
            # find indices of matching rows
            pos_index, mem_index = torch.where(unique_pos_ids[:, None] == outdated_nids)
            if len(pos_index):
                # some positive nodes may not be outdated
                outdated_pos_ids = unique_pos_ids[pos_index]
                update_vals = h_prev_right_all.detach()[mem_index]
                update_prev_ts = prev_ts[mem_index]
                self.msg_store.clear(outdated_pos_ids)  # the messages are comsumed
                self.update_right_memory(outdated_pos_ids, update_vals, update_prev_ts)

        # STEP 5: save current events (no grad)
        self.store_events(src_ids, dst_ids, ts, eids, h_left_with_negs[:2 * bs])  # storing beforing updating!

        # Side Quest: Save the training targets of the restarter!
        # h(t'-) and h(t'+)
        h_prev_left, _ = self.left_memory.get(pos_node_ids)  # old left memory for restarter
        h_prev_right, _ = self.right_memory.get(pos_node_ids)
        h_prev_left = h_prev_left.clone()
        h_prev_right = h_prev_right.clone()

        # STEP 6: update left memory h(t-) in-place (for positive nodes only)
        h_left = h_left_with_negs[:2 * bs]  # h(t-)
        self.update_left_memory(pos_node_ids, h_left, ts.repeat(2))

        # STEP 7: compute loss
        # compute scores
        if self.dyrep:
            local_center_nids = computation_graph.local_index[batch_node_ids]
            h_prev_right_with_negs = involved_node_reprs[local_center_nids]
            x, y, neg_y = h_prev_right_with_negs.reshape(3, bs, self.nfeat_dim)
        else:
            x, y, neg_y = h_left_with_negs.reshape(3, bs, self.nfeat_dim)
        src_hit, dst_hit, neg_src_hit, neg_dst_hit = computation_graph.hit_data
        if self.hit_type == 'vec':
            x_pos_pair = torch.cat([x, src_hit], 1)  # [bs, dim + n_neigh]
            y_pos_pair = torch.cat([y, dst_hit], 1)  # [bs, dim + n_neigh]
            x_neg_pair = torch.cat([x, neg_src_hit], 1)
            y_neg_pair = torch.cat([neg_y, neg_dst_hit], 1)
        elif self.hit_type == 'bin':
            x_pos_pair = x + self.hit_embedding(src_hit.max(1).values.long())
            y_pos_pair = y + self.hit_embedding(dst_hit.max(1).values.long())
            x_neg_pair = x + self.hit_embedding(neg_src_hit.max(1).values.long())
            y_neg_pair = neg_y + self.hit_embedding(neg_dst_hit.max(1).values.long())
        elif self.hit_type == 'count':
            x_pos_pair = x + self.hit_embedding(src_hit.sum(1).long())
            y_pos_pair = y + self.hit_embedding(dst_hit.sum(1).long())
            x_neg_pair = x + self.hit_embedding(neg_src_hit.sum(1).long())
            y_neg_pair = neg_y + self.hit_embedding(neg_dst_hit.sum(1).long())
        else:
            x_pos_pair = x_neg_pair = x
            y_pos_pair = y
            y_neg_pair = neg_y

        pos_scores = self.score_fn(x_pos_pair, y_pos_pair).squeeze(1)
        neg_scores = self.score_fn(x_neg_pair, y_neg_pair).squeeze(1)
        # compute loss
        label_ones = torch.ones_like(pos_scores)
        label_zeros = torch.zeros_like(neg_scores)
        labels = torch.cat([label_ones, label_zeros], 0)
        contrast_loss = self.contrast_loss_fn(
            torch.cat([pos_scores, neg_scores], 0), labels)

        if self.dyrep:
            h_return = h_prev_right_with_negs[:2 * bs]
        else:
            h_return = h_left
        return contrast_loss, h_return, pos_scores, neg_scores, h_prev_left, h_prev_right

    def contrast_learning_no_memory(
            self, src_ids: Tensor, dst_ids: Tensor, neg_dst_ids: Tensor,
            ts: Tensor, eids: Tensor,
            computation_graph: ComputationGraph
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward with negative dst nodes.
        -----
        src_ids          : src node ids
        dst_ids          : dst node ids
        neg_dst_ids      : negative dst node ids
        ts               : timestamps of events (t)
        eids             : edge ids
        computation_graph: computation graph containing necessary info
        -----
        returns: contrast_loss, h_return, pos_scores, neg_scores, h_prev_left, h_prev_right
        contrast_loss: contrastive loss
        h_return     : h(t-) of src and dst nodes
        pos_scores   : scores of positive edges (without sigmoid)
        neg_scores   : scores of negative edges (without sigmoid)
        h_prev_left  : not used
        h_prev_right : not used
        """
        bs = len(src_ids)
        batch_node_ids = torch.cat([src_ids, dst_ids, neg_dst_ids])

        h_left_with_negs = self.compute_temporal_embedding_with_involved_nodes_only(
            None, batch_node_ids, ts.repeat(3), computation_graph
        )

        # compute scores
        x, y, neg_y = h_left_with_negs.reshape(3, bs, self.nfeat_dim)
        src_hit, dst_hit, neg_src_hit, neg_dst_hit = computation_graph.hit_data
        if self.hit_type == 'vec':
            x_pos_pair = torch.cat([x, src_hit], 1)  # [bs, dim + n_neigh]
            y_pos_pair = torch.cat([y, dst_hit], 1)  # [bs, dim + n_neigh]
            x_neg_pair = torch.cat([x, neg_src_hit], 1)
            y_neg_pair = torch.cat([neg_y, neg_dst_hit], 1)
        elif self.hit_type == 'bin':
            x_pos_pair = x + self.hit_embedding(src_hit.max(1).values.long())
            y_pos_pair = y + self.hit_embedding(dst_hit.max(1).values.long())
            x_neg_pair = x + self.hit_embedding(neg_src_hit.max(1).values.long())
            y_neg_pair = neg_y + self.hit_embedding(neg_dst_hit.max(1).values.long())
        elif self.hit_type == 'count':
            x_pos_pair = x + self.hit_embedding(src_hit.sum(1).long())
            y_pos_pair = y + self.hit_embedding(dst_hit.sum(1).long())
            x_neg_pair = x + self.hit_embedding(neg_src_hit.sum(1).long())
            y_neg_pair = neg_y + self.hit_embedding(neg_dst_hit.sum(1).long())
        else:
            x_pos_pair = x_neg_pair = x
            y_pos_pair = y
            y_neg_pair = neg_y

        pos_scores = self.score_fn(x_pos_pair, y_pos_pair).squeeze(1)
        neg_scores = self.score_fn(x_neg_pair, y_neg_pair).squeeze(1)
        # compute loss
        label_ones = torch.ones_like(pos_scores)
        label_zeros = torch.zeros_like(neg_scores)
        labels = torch.cat([label_ones, label_zeros], 0)
        contrast_loss = self.contrast_loss_fn(
            torch.cat([pos_scores, neg_scores], 0), labels)

        h_return = h_left_with_negs[:2 * bs]  # h(t-)
        return contrast_loss, h_return, pos_scores, neg_scores, None, None

    def compute_messages(self, node_ids: Union[Tensor, np.ndarray, None] = None
                         ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Compute messages from message store where h(t'-) or h(t''+) is stored.
        -----
        node_ids: node ids (optional). It could be used when you
                  1. know the invovled nodes, or
                  2. update only positive nodes.
        -----
        Returns: outdated_nodes, msgs, ts
        outdated_nodes: a tensor of outdated node ids.
        msgs: a tensor of aggregated and transformed messages with len(outdated_nodes) rows
        ts: a tensor of timestamps of (last) messages. In most cases, it would be t'.
        """
        # outdated_nodes is a subset of node_ids
        outdated_nodes = self.msg_store.get_outdated_node_ids(node_ids)
        outdated_nodes = outdated_nodes.to(self.device)  # cpu -> device

        if len(outdated_nodes) == 0:
            return (outdated_nodes, None, None)

        # Aggregate
        # NB: (Training Only)
        #     If msg_src == left, then last_update_ts would be t', otherwise t''.
        #     If last_update_ts = t', which is also ts of stored messages,
        #     then the time encode part would become useless.
        # [v0.2] see the next note
        last_update_ts = self.msg_memory.update_ts[outdated_nodes]

        # NB: when using the LastOnly aggregator and msg store, `last_update_ts` is not used,
        #     as time delta has been computed when storing raw messages.
        raw_msgs, ts = self.msg_aggregate_fn(outdated_nodes, last_update_ts, self.msg_store.node_messages)
        # Sanity check  TODO: to remove
        if self.msg_src == 'left' and not (ts == last_update_ts).all().item():
            raise ValueError("Messages' ts should be equal to last update ts "
                             "when using left memory as msg source.")

        if self.tgn_mode:
            # In TGN, full raw messages are stored.
            # But in our implementation, edge and time parts of raw messages are computed,
            # such that the edge embedding module (if any) and time encoder can be trained.
            # [v0.2] when using the LastOnly aggregator and msg store, `raw_msgs` has no grad.
            raw_msgs = raw_msgs.detach()
        # Then transform
        msgs = self.msg_transform_fn(raw_msgs)
        return outdated_nodes, msgs, ts

    def apply_messages(self, node_ids: Tensor, msgs: Tensor, ts: Tensor) -> Tensor:
        """
        Apply messages to compute updated nodes' representations.
        -----
        node_ids: a tensor of node ids
        msgs: aggregated and transformed messages with len(node_ids) rows
        ts: a tensor of timestamps. This is t' if training.
        -----
        Return: new_vals
        new_vals: a tensor of updated node representations (h(t'+) if training).
        """
        # NB: (Training Only)
        #     last_update_ts could be t' or t'' according to the type of upd_memory
        old_vals, last_update_ts = self.upd_memory.get(node_ids)
        delta_ts = ts - last_update_ts  # Currently, this var is not used.
        # No need to check (delta_ts >= 0) as it will be checked when updating memory
        new_vals = self.right_mem_updater(old_vals, msgs, delta_ts)
        return new_vals

    def compute_temporal_embedding_with_involved_nodes_only(
            self, involved_node_reprs: Optional[Tensor], node_ids: Tensor, ts: Tensor,
            computation_graph: ComputationGraph
    ) -> Tensor:
        """
        Compute temporal embeddings of nodes using h(t'+) of involved nodes.
        -----
        involved_node_reprs: h(t'+) of involved nodes (None for TGAT)
        node_ids: ids of target nodes
        ts: timestamps
        -----
        Return: h
        h: h(t-) of input nodes
        """
        h = self.temporal_embedding_fn.compute_embedding_with_computation_graph(
            involved_node_reprs, node_ids, ts, computation_graph
        )
        return h

    def temporal_embedding(self, memory: Memory, node_ids: Tensor, ts: Tensor) -> Tensor:
        """
        Compute temporal embeddings of nodes using the given right memory.
        -----
        memory: the right memory
        node_ids: ids of target nodes
        ts: timestamps
        -----
        Return: h
        h: h(t-)
        """
        warnings.warn('Please use "compute_embedding_with_computation_graph" instead!',
                      DeprecationWarning)
        # temporal embedding fn requires np.ndarray as inputs
        # TODO: 是否需要修改compute_embedding_with_computation_graph或增加条件
        h = self.temporal_embedding_fn.compute_embedding(all_node_reprs=memory.vals,
                                                         np_center_nids=node_ids.cpu().numpy(),
                                                         np_ts=ts.cpu().numpy())
        return h

    @torch.no_grad()
    def update_right_memory(self, node_ids: Tensor, new_vals: Tensor, ts: Tensor):
        """
        Update right memory inplace.
        NB: node_ids have been de-duplicated when calling this method in TIGE.
        -----
        node_ids: a tensor of node ids
        new_vals: updated representations of nodes
        ts: a tensor of timestamps
        """
        self.right_memory.set(node_ids, new_vals, ts)

    @torch.no_grad()
    def update_left_memory(self, node_ids: Tensor, new_vals: Tensor, ts: Tensor):
        """
        Update left memory INPLACE.
        -----
        node_ids: a tensor of node ids
        new_vals: updated representations of nodes
        ts: a tensor of timestamps
        """
        # There may be duplicates in `ids`
        # So we need to select the lastest index first.
        node_ids, index = select_latest_nids(node_ids, ts)
        self.left_memory.set(node_ids, new_vals[index], ts[index])

    @torch.no_grad()
    def store_events(self, src_ids: Tensor, dst_ids: Tensor, ts: Tensor, eids: Tensor,
                     h_left: Optional[Tensor]):
        """
        Store current events in the batch so that they could act as inputs in the next batch.
        -----
        src_ids: source node ids
        dst_ids: dst node ids
        ts: timestamps
        eids: edge ids
        h_left: h(t-) for DyRep
        """
        # Get nodes' current representations from message memory
        src_vals, src_prev_ts = self.msg_memory.get(src_ids)
        dst_vals, dst_prev_ts = self.msg_memory.get(dst_ids)

        # Sanity check TODO: to remove
        if (src_prev_ts > ts).any().item() or (dst_prev_ts > ts).any().item():
            raise ValueError('Events occur before the udpated memory.')

        if self.dyrep:
            # IN DyRep, raw message = (h(src, t'+), h(dst, t-), edge_feat, time)
            bs = len(src_ids)
            dst_paired_vals, src_paired_vals = h_left.reshape(2, bs, self.nfeat_dim)
            # for src: (src_vals, src_paired_vals) == (h(src, t'+), h(dst, t-))
            self.msg_store.store_events(src_ids, dst_ids, src_prev_ts, dst_prev_ts,
                                        src_vals, dst_vals, eids, ts,
                                        self.raw_feat_getter, self.time_encoder,
                                        src_paired_vals=src_paired_vals,
                                        dst_paired_vals=dst_paired_vals,
                                        )
        else:
            self.msg_store.store_events(src_ids, dst_ids, src_prev_ts, dst_prev_ts,
                                        src_vals, dst_vals, eids, ts,
                                        self.raw_feat_getter, self.time_encoder)

    @torch.no_grad()
    def flush_msg(self):
        """
        Consume all messages and update right memory.
        NB: Use this function before saving the model!
        """
        outdated_nids, msgs, prev_ts = self.compute_messages()
        if len(outdated_nids):
            h_prev_right = self.apply_messages(outdated_nids, msgs, prev_ts)  # h(t'+)
            _ = self.update_right_memory(outdated_nids, h_prev_right, prev_ts)
            # Remove messages of these nodes
            self.msg_store.clear(outdated_nids)

    def reset(self):
        """
        Clear memories and msg store (at the beginning of a new epcoh).
        """
        self.left_memory.clear()
        self.right_memory.clear()
        self.msg_store.clear()

    def save_memory_state(self) -> Tuple[Memory, Memory, MessageStore]:
        """
        Save states of memories and message store.
        """
        left_memory = self.left_memory.clone()
        right_memory = self.right_memory.clone()
        msg_store = self.msg_store.clone()
        data = (left_memory, right_memory, msg_store)
        return data
    
    def backup_to_cpu(self) -> Tuple[Memory, Memory, MessageStore]:
        """
        Backup states of memories and message store to cpu.
        """
        left_memory_ = self.left_memory.backup()
        right_memory_ = self.right_memory.backup()
        msg_store_ = self.msg_store.backup()
        data = (left_memory_, right_memory_, msg_store_)
        return data

    def load_memory_state(self, data: Tuple[Memory, Memory, MessageStore]):
        """
        Load states of memories and message store.
        """
        (left_memory, right_memory, msg_store) = data
        self.left_memory = left_memory
        self.right_memory = right_memory
        self.msg_memory = self.left_memory if self.msg_src == 'left' else self.right_memory
        self.upd_memory = self.left_memory if self.upd_src == 'left' else self.right_memory
        self.msg_store = msg_store

    def _get_computation_graph_nodes(
            self, nids: np.ndarray, ts: np.ndarray, depth: Optional[int] = None
    ) -> np.ndarray:
        """
        Get ids of nodes that are involved in the temporal aggregation.
        """
        warnings.warn('This method is no longer usefule!',
                      DeprecationWarning)
        depth = self.n_layers if depth is None else depth
        if depth == 0:
            return np.unique(nids)
        ngh_nids, _, neigh_ts, *_ = self.graph.sample_temporal_neighbor(nids, ts, self.n_neighbors)
        r_nids = self._get_computation_graph_nodes(ngh_nids.flatten(), neigh_ts.flatten(), depth - 1)
        return np.unique(np.concatenate([nids, r_nids]))


class TIGER(TIGE):

    def __init__(self, *, raw_feat_getter: FeatureGetter, graph: Graph,
                 restarter: Optional[Restarter],
                 n_neighbors: int = 20, n_layers: int = 2,
                 n_head: int = 2, dropout: float = 0.1,
                 msg_src: str, upd_src: str,
                 msg_tsfm_type: str = 'id', mem_update_type: str = 'gru',
                 tgn_mode: bool = True, msg_last_only: bool = True,
                 hit_type: str = 'vec',
                 embedding_type: str = 'att',
                 dyrep: bool = False,
                 no_memory: bool = False,
                 ts_delta_mean: float = 0., ts_delta_std: float = 1.,
                 n_shared_nodes: int = 0, static_shared_nodes: bool = False
                 ):
        """
        Temporal Interaction Graph Embedding Model with Restart
        -----
        raw_feat_getter: feature getter
        graph: train graph or full graph
        restarter: restarter module, could be None
        hist_len: length of history used in surrogate model
        n_neighbors: number of neighbors to sample
        n_layers: number of temporal aggregation layers
        n_head: number of heads in MultiHeadAttention
        dropout: dropout ratio
        msg_src: [left, right] using h(t'-) or h(t''+) to compute memory
        upd_src: [left, right] using h(t'-) or h(t''+) as update input
        msg_tsfm_type: [id, linear, mlp] message transformation function
        mem_update_type: [gru, merge, rnn] memory update function
        tgn_mode: if true, message-related parts cannot receive gradients
                  (except msg_tfsm_fn)
        msg_last_only: if true, only the last messages are kept to obtain speed-up
        hit_type: [vec, bin, none] use hit info when computing edge scores
        embedding_type: [att, id, time] embedding modules described in TGN
        dyrep: if true, use the DyRep mode
               (use h(t'+) as the output, use h(t-) as the part of message for dst nodes)
        no_memory: if true, skip all memory-related modules (TGAT mode)
        ts_delta_mean: mean of differences of timestamps
        ts_delta_std: std of differences of timestamps
        """
        super().__init__(
            raw_feat_getter=raw_feat_getter, graph=graph,
            n_neighbors=n_neighbors, n_layers=n_layers, n_head=n_head, dropout=dropout,
            msg_src=msg_src, upd_src=upd_src,
            msg_tsfm_type=msg_tsfm_type, mem_update_type=mem_update_type,
            tgn_mode=tgn_mode, msg_last_only=msg_last_only,
            hit_type=hit_type,
            embedding_type=embedding_type,
            dyrep=dyrep,
            no_memory=no_memory,
            ts_delta_mean=ts_delta_mean, ts_delta_std=ts_delta_std,
            n_shared_nodes=n_shared_nodes, static_shared_nodes=static_shared_nodes
        )
        self.restarter_fn = restarter
        self.mutual_loss_fn = nn.MSELoss()

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        DDP doesn't support call other methods than forward.
        """
        return self.contrast_and_mutual_learning(*args, **kwargs)

    def contrast_and_mutual_learning(
            self, src_ids: Tensor, dst_ids: Tensor, neg_dst_ids: Tensor,
            ts: Tensor, eids: Tensor, computation_graph: ComputationGraph,
            contrast_only: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute losses of contrastive and mutual learning.
        -----
        src_ids: src node ids
        dst_ids: dst node ids
        neg_dst_ids: negative dst node ids
        ts: timestamps of events (t)
        eids: edge ids
        computation_graph: computation graph containing necessary info
        contrast_only: do not use restarter
        -----
        Returns: contrast_loss, mutual_loss
        contrast_loss: contraistive learning loss
        mutual_loss  : mutual learning loss
        """
        contrast_loss, *_, h_prev_left, h_prev_right = self.contrast_learning(
            src_ids, dst_ids, neg_dst_ids, ts, eids, computation_graph)

        if contrast_only or self.restarter_fn is None:
            mutual_loss = torch.tensor(0, device=contrast_loss.device)
            return contrast_loss, mutual_loss

        # `index` is used to indicates unique ids
        # If there are duplicate nodes, only the lastest one is kept.
        index = computation_graph.restart_data.index
        unique_nids = torch.cat([src_ids, dst_ids])[index]
        unique_ts = ts.repeat(2)[index]

        surrogate_h_prev_left, surrogate_h_prev_right, _ = self.restarter_fn(
            unique_nids, unique_ts, computation_graph)

        targets = torch.cat([h_prev_left[index], h_prev_right[index]], 0)  # [2n, d]
        preds = torch.cat([surrogate_h_prev_left, surrogate_h_prev_right], 0)  # [2n, d]
        valid_rows = torch.where(~(targets == 0).all(1))[0]

        if len(valid_rows):
            mutual_loss = self.mutual_loss_fn(preds[valid_rows], targets[valid_rows].detach())
        else:
            mutual_loss = torch.tensor(0, device=contrast_loss.device)

        return contrast_loss, mutual_loss

    @torch.no_grad()
    def restart(self, nids: Tensor, ts: Tensor, mix: float = 0.):
        """
        Using surrogate representations to fill the memories.
        -----
        nids: node ids
        ts: the current timestamp (NB: we compute representations of node at previous ts)
        """
        if len(nids):
            self.msg_store.clear(nids)
            h_prev_left, h_prev_right, prev_ts = self.restarter_fn(nids, ts)
            if mix > 0:
                h_prev_left = mix * h_prev_left + (1 - mix) * self.left_memory.vals[nids]
                h_prev_right = mix * h_prev_right + (1 - mix) * self.right_memory.vals[nids]
            self.left_memory.set(nids, h_prev_left, prev_ts, skip_check=True)
            self.right_memory.set(nids, h_prev_right, prev_ts, skip_check=True)

    @property
    def graph(self):
        return self.temporal_embedding_fn.graph

    @graph.setter
    def graph(self, new_obj: Graph):
        self.temporal_embedding_fn.graph = new_obj
        if self.restarter_fn is not None:
            self.restarter_fn.graph = new_obj