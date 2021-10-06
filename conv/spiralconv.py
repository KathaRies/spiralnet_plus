import torch
import torch.nn as nn
import torch.nn.functional as F


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1, activation=F.elu):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.activation = activation

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        x = self.activation(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


class GatedSpiralConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, indices, dim=1, activation=F.elu
    ):
        super(GatedSpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.mask_layer = nn.Linear(
            in_channels * self.seq_length, out_channels
        )
        self.reset_parameters()

        self.activation = activation
        self.gating = torch.nn.Sigmoid()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        out = self.layer(x)
        out = self.activation(out)
        gated = self.gating(self.mask_layer(x))
        return out*gated

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)
