import openmesh as om
import csv
from typing import List, Tuple
import matplotlib.pyplot as plt
import argparse
import os.path as osp
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
# from torch_geometric.data import DataLoader
from torchsummary import summary

from psbody.mesh import Mesh
from conv.spiralconv import GatedSpiralConv, SpiralConv
from hole_filling.network import Architecture
from hole_filling.train_eval import Loss, c1_loss

from utils import utils, mesh_sampling, Writer, DataLoader
from datasets import BEZIER, MeshData

from hole_filling import run, SkipAE, eval_error, CNN, AE
from correspondence.network import Net

# Settings
parser = argparse.ArgumentParser(description='hole_filling')
parser.add_argument('--dataset', type=str, default='500_0_2x2_4x4_0')
# -1 cpu, else gpu idx
parser.add_argument('--device_idx', type=int, default=0)
parser.add_argument('--n_threads', type=int, default=4)

# network hyperparameters
parser.add_argument('--architecture', type=str, default=Architecture.SkipAE)
parser.add_argument('--loss', type=str, default=Loss.L1)
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[32, 64],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=32)
parser.add_argument('--in_channels', type=int, default=4)
parser.add_argument('--seq_length', type=int, default=[3, 3], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1], nargs='+')
parser.add_argument('--pooling', type=int, default=[2, 2], nargs='+')

# Dimensions e.g 2x2_4x4
# input: 81,3 vertices, 238 edges, 97 faces (tris)
# conv: k-ring, k E [0,args.seq_length]
# pool: downsampled by pooling: 2,2

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=500)

# others
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

args.data_fp = osp.join(
    '/home/katha/Documents/Uni/Thesis/BezierGAN/datasets/', args.dataset
)

args.work_dir = osp.dirname(osp.realpath(__file__))
args.out_dir = osp.join(args.work_dir, 'out')
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')


def _get_transform(
    transform_fp: str, template_fp: str, device, vertices_per_face
) -> Tuple[List[List[float]]]:
    if not osp.exists(transform_fp):
        print('Generating transform matrices...')
        if vertices_per_face == 4:
            raise NotImplemented("QuadMesh cant be sampeled properly")

            class QuadMesh():
                def __init__(self, path: str):
                    mesh = om.read_polymesh(template_fp)
                    self.f = mesh.face_vertex_indices()
                    self.v = mesh.points()

            mesh = QuadMesh(path=template_fp)
        elif vertices_per_face == 3:
            mesh = Mesh(filename=template_fp)
        else:
            raise ValueError("Only supporting Tri and Quad Meshes")
        _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
            mesh, args.pooling
        )
        tmp = {
            'vertices': V,
            'face': F,
            'adj': A,
            'down_transform': D,
            'up_transform': U
        }

        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))
    else:
        with open(transform_fp, 'rb') as f:
            tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        utils.preprocess_spiral(
            tmp['face'][idx], args.seq_length[idx],
            tmp['vertices'][idx],
            args.dilation[idx]
        ).to(device)
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]

    return spiral_indices_list, down_transform_list, up_transform_list


def train_network(args: argparse.Namespace):
    architecture, conv = args.architecture.get_model_class()

    model = architecture(
        args.in_channels, args.out_channels, args.latent_channels,
        spiral_indices_list, down_transform_list,
        up_transform_list, conv=conv
    ).to(device)
    # model = Net(3, 3, spiral_indices_list[0]).to(device) horrible results (5 liloss score)
    print('Number of parameters: {}'.format(utils.count_parameters(model)))
    print(model)
    print(d.size())
    # summary(model, input_size=(81, 3))
    # x = torch.zeros((1, 81, 3)).to(device)  # d.x.to(device).unsqueeze(0)
    # writer.add_graph(model, x)

    # train
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        args.decay_step,
        gamma=args.lr_decay
    )

    print(f"training for {args.epochs} epochs")

    loss = args.loss.get_loss()

    info = run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        device=device,
        use_mask=True,
        loss=loss
    )
    eval_error(model, test_loader, device, meshdata,
               args.out_dir, use_mask=True)

    return info


# get data


device = torch.device(
    'cuda:{}'.format(args.device_idx)
) if args.device_idx else torch.device('cpu')
torch.set_num_threads(args.n_threads)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = Writer(args)

# Data
template_fp = osp.join(args.data_fp, 'raw', 'train',
                       'data', 'data_0.obj')
meshdata = MeshData(
    root=args.data_fp,
    template_fp=template_fp,
    datast=BEZIER,
    dataset_kwargs={}
)
train_dataset = meshdata.train_dataset  # BEZIER(args.data_fp, True)
test_dataset = meshdata.test_dataset  # BEZIER(args.data_fp, False)
train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
# d = train_dataset[0]
d = next(iter(test_loader))
# spiral_indices = preprocess_spiral(d.face.T, args.seq_length).to(device)
print(d.y.size())
plt.scatter(d.y[0, :, 1], d.y[0, :, 2])
# plt.show()

print(c1_loss(d.y.to(device), None))
# Network
# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform.pkl')
spiral_indices_list, down_transform_list, up_transform_list = _get_transform(
    transform_fp, template_fp, device, 3
)

first = True

with open('architectures_l1_comparision_500.csv', 'w') as f:
    w = csv.writer(f, delimiter=',')
    # for l in Loss:
    for arch in Architecture:
        # for n_layers in range(1, 4):
        for length in range(1, 9, 2):
            # for pool in range(1, 4):
            args.architecture = arch
            args.loss = Loss.L1
            args.seq_length = [length]*2
            # args.pooling = [pool]*n_layers
            # args.dilation = [dil]*n_layers
            print(args)
            info = train_network(args)
            temp = vars(args)
            temp.update(info)
            print(temp)
            if first:
                first = False
                w.writerow(temp.keys())
            w.writerow(temp.values())
