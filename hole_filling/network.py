from enum import Enum
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
            nn.Linear(out_channels[0]+in_channels, 3)
        )

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
        spiral_indices, down_transform, up_transform, conv=SpiralConv
    ):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spiral_index = spiral_indices[0]

        # encoder
        self.layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.layers.append(
                    conv(in_channels, out_channels[idx],
                         self.spiral_index))
            else:
                self.layers.append(
                    conv(out_channels[idx - 1], out_channels[idx],
                         self.spiral_index))
        self.layers.append(
            nn.Linear(out_channels[-1], 3))

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


class SkipCNN(nn.Module):
    def __init__(
        self, in_channels, out_channels, latent_channels,
        spiral_indices, down_transform, up_transform, conv=SpiralConv
    ):
        super(SkipCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spiral_index = spiral_indices[0]

        # encoder
        self.layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.layers.append(
                    conv(in_channels, out_channels[idx],
                         self.spiral_index))
            else:
                self.layers.append(
                    conv(out_channels[idx - 1], out_channels[idx],
                         self.spiral_index))
        self.layers.append(
            nn.Linear(out_channels[-1]+in_channels, 3))

        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            layer.reset_parameters()

    def forward(self, x, *indices):
        original = x
        for i, layer in enumerate(self.layers[:-1]):
            # x = x.view(-1, layer.weight.size(1))
            x = layer(x)
        input = torch.cat((x, original), -1)
        x = self.layers[-1](input)
        return x


class NN(nn.Module):
    def __init__(
        self, in_channels, out_channels, latent_channels,
        spiral_indices, down_transform, up_transform, conv=None
    ):
        super(NN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spiral_index = spiral_indices[0]

        # encoder
        self.layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.layers.append(
                    nn.Linear(in_channels, out_channels[idx])
                )
            else:
                self.layers.append(
                    nn.Linear(out_channels[idx - 1], out_channels[idx])
                )
        self.layers.append(
            nn.Linear(out_channels[-1], 3))

        self.reset_parameters()

    def reset_parameters(self):
        # for i, layer in enumerate(self.layers):
        #     layer.reset_parameters()
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, *indices):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.elu(layer(x))
        x = self.layers[-1](x)
        return x


class Architecture(Enum):
    GatedSkipAE = "GatedSkipAE"
    GatedSkipCNN = "GatedSkipCNN"
    GatedAE = "GatedAE"
    GatedCNN = "GatedCNN"
    SkipAE = "SkipAE"
    SkipCNN = "SkipCNN"
    AE = "AE"
    CNN = "CNN"
    NN = "NN"

    def get_model_class(self):
        if self == Architecture.GatedAE:
            return AE, GatedSpiralConv
        elif self == Architecture.GatedCNN:
            return CNN, GatedSpiralConv
        elif self == Architecture.GatedSkipAE:
            return SkipAE, GatedSpiralConv
        elif self == Architecture.GatedSkipCNN:
            return SkipCNN, GatedSpiralConv
        elif self == Architecture.SkipAE:
            return SkipAE, SpiralConv
        elif self == Architecture.SkipCNN:
            return SkipCNN, SpiralConv
        elif self == Architecture.CNN:
            return CNN, SpiralConv
        elif self == Architecture.AE:
            return AE, SpiralConv
        elif self == Architecture.NN:
            return NN, None
        else:
            raise ValueError("Unknown Architecture")
