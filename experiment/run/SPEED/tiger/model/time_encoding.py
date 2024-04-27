import numpy as np
import torch
from torch import Tensor, nn


class TimeEncode(nn.Module):
    """
    Time Encoding proposed by TGAT
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(dim).float())

    def forward(self, ts: Tensor) -> Tensor:
        """
        ts: [batch_size] or [batch_size, seq_len]
        ---
        return: harmonic

        harmonic: [batch_size, dim] or [batch_size, seq_len, dim]
        """
        ts = ts.unsqueeze(dim=-1)  # [bs, 1] or [bs, seq_len, 1]
        # auto broad-cast
        harmonic = torch.cos(ts * self.basis_freq + self.phase)
        return harmonic
