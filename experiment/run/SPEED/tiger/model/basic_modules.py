import torch
from torch import nn


class MergeLayer(nn.Module):
    def __init__(self, dim1, dim2, hidden_size, out_size, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        h = self.dropout(self.act(self.fc1(x)))
        return self.fc2(h)


class MLP(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, 80), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(80, 10), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        h = self.fn(x).squeeze(dim=-1)
        return h