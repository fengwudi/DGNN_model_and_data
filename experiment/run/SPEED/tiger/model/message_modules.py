from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor

from .feature_getter import FeatureGetter
from .time_encoding import TimeEncode


class MessageFunction(nn.Module):

    def __init__(self, raw_msg_dim: int, out_msg_dim: Optional[int]=None):
        super().__init__()
        self.input_size = raw_msg_dim
        self.output_size = out_msg_dim

    def forward(self, raw_messages: Tensor) -> Tensor:
        raise NotImplementedError


class IdentityMessageFunction(MessageFunction):

    def __init__(self, raw_msg_dim: int, *args, **kwargs):
        super().__init__(raw_msg_dim, raw_msg_dim)

    def forward(self, raw_messages: Tensor) -> Tensor:
        return raw_messages


class MLPMessageFunction(MessageFunction):

    def __init__(self, raw_msg_dim: int, out_msg_dim: Optional[int]=None, dropout: float=0.0):
        out_msg_dim = raw_msg_dim if out_msg_dim is None else out_msg_dim
        super().__init__(raw_msg_dim, out_msg_dim)
        self.hidden_size = self.output_size // 2
        self.fn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(raw_msg_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, raw_messages: Tensor) -> Tensor:
        return self.fn(raw_messages)


class LinearMessageFunction(MessageFunction):

    def __init__(self, raw_msg_dim: int, out_msg_dim: Optional[int]=None, dropout: float=0.0):
        out_msg_dim = raw_msg_dim if out_msg_dim is None else out_msg_dim
        super().__init__(raw_msg_dim, out_msg_dim)
        self.fn = nn.Sequential(nn.Dropout(dropout), nn.Linear(raw_msg_dim, out_msg_dim))

    def forward(self, raw_messages: Tensor) -> Tensor:
        return self.fn(raw_messages)


class MessageAggregatorNoGrad(nn.Module):

    def __init__(self, raw_feat_getter: FeatureGetter, time_encoder: TimeEncode):
        super().__init__()
        self.raw_feat_getter = raw_feat_getter
        self.time_encoder = time_encoder
    
    def forward(self, node_ids: Tensor, prev_ts: Tensor,
                node_msg_dict: Dict[int, List[Tuple[Tensor, float, bool]]]
                ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class MessageAggregator(nn.Module):

    def __init__(self, raw_feat_getter: FeatureGetter, time_encoder: TimeEncode):
        super().__init__()
        self.raw_feat_getter = raw_feat_getter
        self.time_encoder = time_encoder
    
    def forward(self, node_ids: Tensor, prev_ts: Tensor,
                node_msg_dict: Dict[int, List[Tuple[Tensor, int, float, bool]]]
                ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class LastMessageAggregator(MessageAggregator):

    def forward(self, node_ids: Tensor, prev_ts: Tensor,
                node_msg_dict: Dict[int, List[Tuple[Tensor, int, float, bool]]]
                ) -> Tuple[Tensor, Tensor]:
        # Sanity check
        for node in node_ids:
            assert len(node_msg_dict[node.item()])
            
        nodes_tensors = []
        eids = []
        ts = []
        # dirs = []
        for node in node_ids:
            msgs = node_msg_dict[node.item()]
            node_vals, eid, timestamp, dir = msgs[-1]
            nodes_tensors.append(node_vals)
            eids.append(eid)
            ts.append(timestamp)
            # dirs.append(dir)
        nodes_tensors = torch.stack(nodes_tensors, 0)
        device = nodes_tensors.device  # this can help us figure out the using device

        eids = torch.tensor(eids).long().to(device)
        ts = torch.tensor(ts).float().to(device)
        # dirs = torch.tensor(dirs).float().to(device)

        # sanity check
        if (prev_ts > ts).any().item():
            raise ValueError('Messages happened later than memory updating.')

        edges_vals = self.raw_feat_getter.get_edge_embeddings(eids)
        delta_ts_vals = self.time_encoder(ts - prev_ts)
        full_msgs = torch.cat([nodes_tensors, edges_vals, delta_ts_vals], dim=1)
        return full_msgs, ts
        
        
class LastMessageAggregatorNoGrad(MessageAggregatorNoGrad):

    def forward(self, node_ids: Tensor, prev_ts: Tensor,
                node_msg_dict: Dict[int, List[Tuple[Tensor, float]]]
                ) -> Tuple[Tensor, Tensor]:
        # Sanity check
        for node in node_ids:
            assert len(node_msg_dict[node.item()])
            
        nodes_tensors = []
        ts = []
        for node in node_ids:
            msgs = node_msg_dict[node.item()]
            node_vals, timestamp = msgs[-1]
            nodes_tensors.append(node_vals)
            ts.append(timestamp)
        full_msgs = torch.stack(nodes_tensors, 0)
        device = full_msgs.device  # this can help us figure out the using device
        ts = torch.tensor(ts).float().to(device)

        # sanity check
        if (prev_ts > ts).any().item():
            raise ValueError('Messages happened later than memory updating.')

        return full_msgs, ts
        
        
class LastMessageAggregatorNoGradLastOnly(MessageAggregatorNoGrad):

    def forward(self, node_ids: Tensor, prev_ts: Tensor,
                node_msg: Tuple[Tensor, Tensor]
                ) -> Tuple[Tensor, Tensor]:
        full_msgs = node_msg[0][node_ids]
        ts = node_msg[1][node_ids]
        # sanity check
        if (prev_ts > ts).any().item():
            raise ValueError('Messages happened later than memory updating.')
        return full_msgs, ts
