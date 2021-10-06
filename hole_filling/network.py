from reconstruction.network import AE, SpiralDeblock, SpiralEnblock, Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from conv import SpiralConv, GatedSpiralConv


class SkipAE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform, conv=SpiralConv):
        super(SkipAE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(SpiralEnblock(
                    in_channels, out_channels[idx],
                    self.spiral_indices[idx],
                    conv=conv
                ))
            else:
                self.en_layers.append(SpiralEnblock(
                    out_channels[idx - 1], out_channels[idx],
                    self.spiral_indices[idx],
                    conv=conv
                ))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1]+out_channels[idx+1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1], conv=conv))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx]+out_channels[idx-1], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1], conv=conv))
        self.de_layers.append(
            conv(out_channels[0]+in_channels, 3, self.spiral_indices[0]))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, *indices):
        # 0->3 [-3]
        # 1->2 [-2]
        # 2->1 [-1]
        original = x

        enc_out = []
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
                enc_out.append(x)
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
                # enc_out.append(x)

        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                input = torch.cat((x, enc_out[-i]), -1)
                x = layer(input, self.up_transform[num_features - i])
            else:
                x = layer(torch.cat((x, original), -1))

        return x


class CNN(nn.Module):
    # Better use correspondences.Net
    def __init__(
        self, in_channels, out_channels, latent_channels,
        spiral_indices, down_transform, up_transform
    ):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices

        # encoder
        self.layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.layers.append(
                    SpiralConv(in_channels, out_channels[idx],
                               self.spiral_indices[idx]))
            else:
                self.layers.append(
                    SpiralConv(out_channels[idx - 1], out_channels[idx],
                               self.spiral_indices[idx]))
        self.layers.append(
            nn.Linear(out_channels[-1], latent_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            layer.reset_parameters()
        # for name, param in self.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0)
        #     else:
        #         nn.init.xavier_uniform_(param)

    def forward(self, x, *indices):
        for i, layer in enumerate(self.layers):
            # x = x.view(-1, layer.weight.size(1))
            x = layer(x,)
        return x
