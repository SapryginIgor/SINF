from torch import nn


class Reparametrization(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        fc1 = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc1(x)