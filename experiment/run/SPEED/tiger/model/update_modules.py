import torch
from torch import nn, Tensor

from .basic_modules import MergeLayer


class UpdateModule(nn.Module):
    def __init__(self, msg_dim, memory_dim):
        super().__init__()
        self.msg_dim = msg_dim
        self.memory_dim = memory_dim

    def forward(self, mem: Tensor, msg: Tensor, delta_ts: Tensor) -> Tensor:
        """
        mem: old memory values with shape [n, mem_dim]
        msg: messages with shape [n, msg_dim]
        delta_ts: elapsed time since last update [n]
        -----
        Returns: updated_mem
        updated_mem: updated memory (*) with shape [n, mem_dim]
        -----
        Note:
        The return value may not be directly used as new memory.
        TGN : h(t'+) <- Update{ h(t''+), m(t') }
        Ours: h(t'+) <- Update{ h(t'-),  m(t') }
        """
        raise NotImplementedError


class GRUUpdater(UpdateModule):
    def __init__(self, msg_dim, memory_dim):
        super().__init__(msg_dim, memory_dim)
        self.cell = nn.GRUCell(input_size=self.msg_dim, hidden_size=self.memory_dim)
    
    def forward(self, mem: Tensor, msg: Tensor, delta_ts: Tensor) -> Tensor:
        h = self.cell(msg, mem)
        return h


class MergeUpdater(UpdateModule):
    def __init__(self, msg_dim, memory_dim):
        super().__init__(msg_dim, memory_dim)
        self.fn = MergeLayer(msg_dim, memory_dim, memory_dim, memory_dim)
    
    def forward(self, mem: Tensor, msg: Tensor, delta_ts: Tensor) -> Tensor:
        h = self.fn(msg, mem)
        return h


class IdentityUpdater(UpdateModule):
    def forward(self, mem: Tensor, msg: Tensor, delta_ts: Tensor) -> Tensor:
        return mem


class RNNUpdater(UpdateModule):
    def __init__(self, msg_dim, memory_dim):
        super().__init__(msg_dim, memory_dim)
        self.cell = nn.RNNCell(input_size=self.msg_dim, hidden_size=self.memory_dim)

    def forward(self, mem: Tensor, msg: Tensor, delta_ts: Tensor) -> Tensor:
        h = self.cell(msg, mem)
        return h
