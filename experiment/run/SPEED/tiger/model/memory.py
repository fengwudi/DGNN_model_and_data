import copy
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from torch import nn, Tensor

from .time_encoding import TimeEncode
from .feature_getter import NumericalFeature
from .utils import select_latest_nids


class Memory(nn.Module):
    def __init__(self, n, dim, n_shared_nodes, static_shared_nodes):
        super().__init__()
        """
        node_ids = [0, share1, share2, ..., shareK, selfK+1, ..., selfN]
        """
        self.n = n
        self.dim = dim

        self.register_buffer('update_ts', torch.zeros(n), persistent=True)
        self.register_buffer('active_mask', torch.zeros(n).bool(), persistent=True)

        self.n_shared_nodes = n_shared_nodes
        if self.n_shared_nodes > 0:
            # 把0号节点也当作是Embedding计算
            self.shared_memory = nn.Embedding(self.n_shared_nodes+1, self.dim)
        else:
            self.shared_memory = None

        if self.n_shared_nodes == 0:
            self.register_buffer('vals', torch.zeros(n, dim), persistent=True)
        else:
            self.register_buffer('non_shared_vals', torch.zeros(n - n_shared_nodes - 1, dim), persistent=True)
            self.register_buffer('vals', torch.zeros(n, dim), persistent=True)

        self.static = static_shared_nodes

    def clone(self):
        if self.n_shared_nodes == 0:
            copy_mem = Memory(self.n, self.dim, self.n_shared_nodes, self.static)
            copy_mem.vals.data = self.vals.data.clone()
            copy_mem.update_ts.data = self.update_ts.data.clone()
        else:
            copy_mem = Memory(self.n, self.dim, self.n_shared_nodes, self.static)
            copy_mem.non_shared_vals.data = self.non_shared_vals.data.clone()

            copy_mem.shared_memory.weight.data = self.shared_memory.weight.data.clone()

            copy_mem.vals.data = self.vals.data.clone()
            copy_mem.update_ts.data = self.update_ts.data.clone()
        return copy_mem
    
    def backup(self):
        if self.n_shared_nodes == 0:
            copy_mem = Memory(self.n, self.dim, self.n_shared_nodes, self.static).to("cpu")
            copy_mem.vals = self.vals.clone().to("cpu")
            copy_mem.update_ts = self.update_ts.clone().to("cpu")
        else:
            shared_memory_copy = self.copy_shared_memory_to_cpu()
            copy_mem = Memory(self.n, self.dim, self.n_shared_nodes, self.static).to("cpu")
            copy_mem.non_shared_vals = self.non_shared_vals.clone().to("cpu")

            copy_mem.shared_memory = shared_memory_copy

            copy_mem.vals = self.vals.clone().to("cpu")
            copy_mem.update_ts = self.update_ts.clone().to("cpu")
            torch.cuda.empty_cache()
        return copy_mem
    
    def copy_shared_memory_to_cpu(self):
        shared_memory_copy = nn.Embedding(self.shared_memory.weight.shape[0], self.shared_memory.weight.shape[1]).cpu()
        shared_memory_copy.weight.data = self.shared_memory.weight.data.clone().to("cpu")
        return shared_memory_copy

    @property
    def device(self):
        return self.vals.device if self.n_shared_nodes == 0 else self.non_shared_vals.device

    def clear(self):
        if self.n_shared_nodes == 0:
            self.vals[...] = 0.
        else:
            self.non_shared_vals[...] = 0.
            if not self.static:
                self.shared_memory.weight.data[...] = 0.
            self.vals[...] = 0.
        self.update_ts[...] = 0.
        self.active_mask[...] = False

    def get(self, ids: Tensor) -> Tuple[Tensor, Tensor]:
        if self.n_shared_nodes == 0:
            v = self.vals[ids]
            ts = self.update_ts[ids]
        else:
            v = self.cat_values(ids)
            ts = self.update_ts[ids]
        return v, ts

    def set(self, ids: Tensor, vals: Tensor, ts: Tensor, skip_check=False):
        if self.n_shared_nodes == 0:
            pass
        else:
            none_shared_node_mask = ids > self.n_shared_nodes
            shared_node_mask = ~none_shared_node_mask
            ids_non_shared = ids[none_shared_node_mask]
            ids_shared = ids[shared_node_mask]
            vals_non_shared = vals[none_shared_node_mask]
            vals_shared = vals[shared_node_mask]
            
        if not skip_check:
            # sanity check
            last_ts = self.update_ts[ids]
            if (last_ts > ts).any().item():
                raise ValueError('You are not allowed to modify past memory.')
            if len(ids) != len(ids.unique()):
                raise ValueError('Duplicate node ids are not allowed.')

        if self.n_shared_nodes == 0:
            self.update_ts.data[ids] = ts
            self.vals.data[ids] = vals
            self.active_mask[ids] = True
        else:
            self.update_ts.data[ids] = ts
            self.active_mask[ids] = True
            self.non_shared_vals.data[ids_non_shared] = vals_non_shared
            if not self.static:
                self.shared_memory.weight.data[ids_shared] = vals_shared
            self.vals.data = torch.cat([self.shared_memory.weight.data, self.non_shared_vals.data], dim=0)


    def cat_values(self, ids):
        assert ids.dim() == 1
        B = ids.size(0)
        value = torch.empty((B, self.dim), dtype=self.non_shared_vals.dtype, device=self.device)
        shared_mask = ids < self.n_shared_nodes + 1
        value[shared_mask, :] = self.shared_memory.weight.data[ids[shared_mask]]
        value[~shared_mask, :] = self.non_shared_vals.data[ids[~shared_mask] - self.n_shared_nodes - 1]
        return value
            

class MessageStoreNoGradLastOnly(nn.Module):
    node_messages: Tuple[Tensor, Tensor]

    def __init__(self, n, dim):
        super().__init__()
        self.n = n
        self.dim = dim
        self.register_buffer('node_msg_vals',
                             torch.zeros((n, dim)).float(),
                             persistent=False)
        self.register_buffer('node_msg_ts',
                             torch.zeros(n).float(),
                             persistent=False)
        self.nodes_with_messages = set()

    @property
    def node_messages(self):
        return (self.node_msg_vals, self.node_msg_ts)

    def clone(self):
        return copy.deepcopy(self)
    
    def backup(self):
        copy_msg = MessageStoreNoGradLastOnly(self.n, self.dim).to("cpu")
        copy_msg.node_msg_vals = self.node_msg_vals.to("cpu")
        copy_msg.node_msg_ts = self.node_msg_ts.to("cpu")
        torch.cuda.empty_cache()
        copy_msg.nodes_with_messages = copy.deepcopy(self.nodes_with_messages)
        return copy_msg

    def store_events(self, src_ids: Tensor, dst_ids: Tensor,
                     src_prev_ts: Tensor, dst_prev_ts: Tensor,
                     src_vals: Tensor, dst_vals: Tensor,
                     eids: Tensor, ts: Tensor,
                     emb_getter: NumericalFeature, time_encoder: TimeEncode,
                     src_paired_vals: Optional[Tensor] = None,
                     dst_paired_vals: Optional[Tensor] = None
                     ):
        """
        Store raw messages of events
        -----
        src_ids: source node ids
        dst_ids: dst node ids
        src_prev_ts: previous timestamps of src nodes
        dst_prev_ts: previous timestamps of dst nodes
        src_vals: values (memory) of src nodes
        dst_vals: values (memory) of dst nodes
        eids: edge ids
        ts: timestamps
        emb_getter: feature getter
        time_encoder: time encoder
        src_paired_vals: (default=dst_vals) use this to simulate DyRep
        dst_paired_vals: (default=src_vals) use this to simulate DyRep
        """
        # sanity check
        # for these nodes, their messages should be used and thus cleared
        # TODO: To remove this check
        for n in torch.concat([src_ids, dst_ids]).cpu().numpy():
            if n in self.nodes_with_messages:
                raise ValueError(f'Node #{n} has unused messages.')

        src_vals += emb_getter.get_node_embeddings(src_ids)  # add static features
        dst_vals += emb_getter.get_node_embeddings(dst_ids)
        edge_vals = emb_getter.get_edge_embeddings(eids)  # [n, d]

        if src_paired_vals is None:
            src_paired_vals = dst_vals
        if dst_paired_vals is None:
            dst_paired_vals = src_vals

        src_full_vals = torch.cat(
            [src_vals, src_paired_vals, edge_vals, time_encoder(ts - src_prev_ts)], 1)  # [n, 4d]
        dst_full_vals = torch.cat(
            [dst_vals, dst_paired_vals, edge_vals, time_encoder(ts - dst_prev_ts)], 1)  # [n, 4d]

        full_nids, index = select_latest_nids(torch.cat([src_ids, dst_ids]), ts.repeat(2))

        full_vals = torch.cat([src_full_vals, dst_full_vals], dim=0)[index]
        full_ts = ts.repeat(2)[index]

        self.nodes_with_messages.update(full_nids.cpu().numpy())

        self.node_msg_vals[full_nids] = full_vals
        self.node_msg_ts[full_nids] = full_ts

    def get_outdated_node_ids(self, node_ids: Union[Tensor, np.ndarray, None]
                              ) -> Tensor:
        """
        Return ids of nodes that store unused messages.
        -----
        node_ids: only return a subset of this tensor
        -----
        Return: outdated_node_ids
        outdated_node_ids: a LongTensor (on CPU)
        """
        if node_ids is None:
            outdated_node_ids = self.nodes_with_messages
        else:
            if isinstance(node_ids, np.ndarray):
                outdated_node_ids = self.nodes_with_messages & set(node_ids)
            else:
                outdated_node_ids = self.nodes_with_messages & set(node_ids.cpu().numpy())
        outdated_node_ids = torch.tensor(list(outdated_node_ids))
        return outdated_node_ids

    def clear(self, nids: Optional[Tensor] = None):
        """
        Clear pending messages.
        """
        if nids is None:
            nids = torch.tensor(list(self.nodes_with_messages),
                                device=self.node_msg_vals.device).long()

        self.nodes_with_messages.difference_update(nids.cpu().numpy())
        self.node_msg_vals[nids].fill_(0)
        self.node_msg_ts[nids].fill_(0)


class MessageStore:
    node_messages: Dict[int, List[Tuple[Tensor, int, float, bool]]]

    def __init__(self, n):
        self.n = n
        self.node_messages = {i: [] for i in range(n)}
        self.nodes_with_messages = set()

    def clone(self):
        return copy.deepcopy(self)

    def backup(self):
        copy_msg = MessageStore(self.n)
        copy_msg.node_messages = copy.deepcopy(self.node_messages)
        copy_msg.nodes_with_messages = copy.deepcopy(self.nodes_with_messages)
        return copy_msg

    def store_events(self, src_ids: Tensor, dst_ids: Tensor,
                     src_prev_ts: Tensor, dst_prev_ts: Tensor,
                     src_vals: Tensor, dst_vals: Tensor,
                     eids: Tensor, ts: Tensor,
                     emb_getter: NumericalFeature, time_encoder: TimeEncode,
                     src_paired_vals: Optional[Tensor] = None,
                     dst_paired_vals: Optional[Tensor] = None
                     ):
        if (src_paired_vals is not None) or (dst_paired_vals is not None):
            raise NotImplementedError('"MessageStore" does not support DyRep!')

        src_ids, dst_ids, ts, eids = (x.cpu().numpy() for x in (src_ids, dst_ids, ts, eids))
        # sanity check
        # for these nodes, their messages should be used and thus cleared
        # TODO: To remove this check
        for n in np.concatenate([src_ids, dst_ids]):
            if len(self.node_messages[n]):
                raise ValueError(f'Node #{n} has {len(self.node_messages[n])} unused messages.')

        vals = torch.stack([src_vals, dst_vals], 1)  # [n, 2, d]
        for src, dst, val, eid, t in zip(src_ids, dst_ids, vals, eids, ts):
            self.node_messages[src].append((val.flatten(), eid, t, True))
            self.node_messages[dst].append((val.roll(shifts=1, dims=0).flatten(), eid, t, False))
            self.nodes_with_messages.add(src)
            self.nodes_with_messages.add(dst)

    def get_outdated_node_ids(self, node_ids: Union[Tensor, np.ndarray, None]
                              ) -> Tensor:
        """
        Return ids of nodes that store unused messages.
        -----
        node_ids: only return a subset of this tensor
        -----
        Return: node_ids
        outdated_node_ids: a LongTensor (on CPU)
        """
        if node_ids is None:
            outdated_node_ids = self.nodes_with_messages
        else:
            if isinstance(node_ids, np.ndarray):
                outdated_node_ids = self.nodes_with_messages & set(node_ids)
            else:
                outdated_node_ids = self.nodes_with_messages & set(node_ids.cpu().numpy())
        outdated_node_ids = torch.tensor(list(outdated_node_ids))
        return outdated_node_ids

    def clear(self, nids: Optional[Tensor] = None):
        """
        Clear stashed messages.
        """
        if nids is not None:
            for n in nids:
                n = n.item()
                self.node_messages[n].clear()
                self.nodes_with_messages.remove(n)
        else:
            for n in self.nodes_with_messages:
                self.node_messages[n].clear()
            self.nodes_with_messages.clear()


class MessageStoreNoGrad(MessageStore):
    node_messages: Dict[int, List[Tuple[Tensor, float]]]

    def store_events(self, src_ids: Tensor, dst_ids: Tensor,
                     src_prev_ts: Tensor, dst_prev_ts: Tensor,
                     src_vals: Tensor, dst_vals: Tensor,
                     eids: Tensor, ts: Tensor,
                     emb_getter: NumericalFeature, time_encoder: TimeEncode,
                     src_paired_vals: Optional[Tensor] = None,
                     dst_paired_vals: Optional[Tensor] = None
                     ):
        # sanity check
        # for these nodes, their messages should be used and thus cleared
        # TODO: To remove this check
        for n in torch.concat([src_ids, dst_ids]).cpu().numpy():
            if len(self.node_messages[n]):
                raise ValueError(f'Node #{n} has {len(self.node_messages[n])} unused messages.')

        src_vals += emb_getter.get_node_embeddings(src_ids)  # add static features
        dst_vals += emb_getter.get_node_embeddings(dst_ids)

        edge_vals = emb_getter.get_edge_embeddings(eids)  # [n, d]

        if src_paired_vals is None:
            src_paired_vals = dst_vals
        if dst_paired_vals is None:
            dst_paired_vals = src_vals

        src_full_vals = torch.cat([src_vals, src_paired_vals, edge_vals, time_encoder(ts - src_prev_ts)], 1)  # [n, 4d]
        dst_full_vals = torch.cat([dst_vals, dst_paired_vals, edge_vals, time_encoder(ts - dst_prev_ts)], 1)  # [n, 4d]

        src_ids = src_ids.cpu().numpy()
        dst_ids = dst_ids.cpu().numpy()
        ts = ts.cpu().numpy()
        for src, dst, sval, dval, t in zip(src_ids, dst_ids, src_full_vals, dst_full_vals, ts):
            self.node_messages[src].append((sval, t))
            self.node_messages[dst].append((dval, t))
            self.nodes_with_messages.add(src)
            self.nodes_with_messages.add(dst)
