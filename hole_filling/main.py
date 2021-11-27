import os
import openmesh as om
import csv
from typing import List, Tuple
import matplotlib.pyplot as plt
import argparse
import os.path as osp
import pickle
import numpy as np
import numpy.core.multiarray

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
from hole_filling.network import Architecture, HoleSkipAE, HoleAE
from hole_filling.train_eval import Loss, c1_eval, c1_loss

from utils import utils, mesh_sampling, Writer, DataLoader
from datasets import BEZIER, MeshData

from hole_filling import run, SkipAE, eval_error, CNN, AE
from correspondence.network import Net
from utils.read import read_mesh

# Settings
parser = argparse.ArgumentParser(description='hole_filling')
parser.add_argument('--dataset', type=str,
                    default='100_0_2x2_4x4_0_norm')
# -1 cpu, else gpu idx
parser.add_argument('--device_idx', type=int, default=0)
parser.add_argument('--n_threads', type=int, default=4)

# network hyperparameters
parser.add_argument('--architecture', type=str, default=Architecture.SkipAE)
parser.add_argument('--loss', type=str, default=Loss.L1)
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[16, 32],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=32)
parser.add_argument('--in_channels', type=int, default=4)
parser.add_argument('--seq_length', type=int, default=[3, 3], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1], nargs='+')
parser.add_argument('--pooling', type=int, default=[2, 2], nargs='+')

# Dimensions e.g 2x2_4x4
# input: 81,3 vertices, 238 edges, 97 faces (tris)
# conv: v_i E k-ring, i E [0,args.seq_length]
# pool: downsampled by pooling: 2,2

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.5)  # 0.99
parser.add_argument('--decay_step', type=int, default=25)  # 1
parser.add_argument('--weight_decay', type=float, default=0)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=500)

# others
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


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
            print(template_fp)
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
    train_dataset = meshdata.train_dataset
    test_dataset = meshdata.test_dataset
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    d = next(iter(test_loader))
    d.mask[0]
    input_size = d.x.shape[1]
    #hole_size = input_size-d.mask[0].count_nonzero()

    architecture, conv = args.architecture.get_model_class()

    model = architecture(
        args.in_channels, args.out_channels, args.latent_channels,
        spiral_indices_list, down_transform_list,
        up_transform_list, conv=conv, hole_size=args.hole_size, input_size=input_size
    ).to(device)
    # model = Net(3, 3, spiral_indices_list[0]).to(device) horrible results (5 liloss score)
    print('Number of parameters: {}'.format(utils.count_parameters(model)))
    print(model)

    # train
    writer = Writer(args)

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
    # print(loss)
    # print(c1_loss(d.y, d.y))
    # exit()

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
    meshes = eval_error(model, test_loader, device, meshdata,
                        args.out_dir, use_mask=True)

    # eval_error(
    #     model, flat, device, meshdata, args.out_dir, use_mask=True
    # )

    return info, meshes


def preload(args):
    # get data
    args.data_fp = osp.join(
        '/home/katha/Documents/Uni/Thesis/BezierGAN/datasets/', args.dataset
    )

    # Data
    template_fp = osp.join(args.data_fp, 'raw', 'test',
                           'label', 'label_1.obj')
    meshdata = MeshData(
        root=args.data_fp,
        template_fp=template_fp,
        datast=BEZIER,
        dataset_kwargs={}
    )

    # Network
    # generate/load transform matrices
    transform_fp = osp.join(args.data_fp, 'transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list = _get_transform(
        transform_fp, template_fp, device, 3
    )
    return meshdata, device, spiral_indices_list, down_transform_list, up_transform_list


args.epochs = 50
args.batch_size = 8
args.loss = Loss.COMBI
#args.architecture = Architecture.HoleAE
#degree = 2
#dim = 4
#args.dataset = f"10000_0_{degree}x{degree}_{dim}x{dim}"
args.seq_length = [3, 3]
args.out_channels = [32, 64]
args.latent_channels = 128
# meshdata, device, spiral_indices_list, down_transform_list, up_transform_list = preload(
#     args
# )
# info, meshes = train_network(args)
args.work_dir = osp.dirname(osp.realpath(__file__))
args.out_dir = osp.join(args.work_dir, 'evaluation')
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')

device = torch.device(
    'cuda:{}'.format(args.device_idx)
) if args.device_idx else torch.device('cpu')
torch.set_num_threads(args.n_threads)

utils.makedirs(args.out_dir)

first = True
with open(osp.join(args.out_dir, 'cubic_redo_results.csv'), 'w') as f:
    w = csv.writer(f, delimiter=',')
    for degree in [3]:  # 2
        for dataset in ["4", "8"]:  # "4", "8",m8
            if dataset == "4":
                patch_size = 4
                dataset_path = f"10000_0_{degree}x{degree}_{patch_size}x{patch_size}"
                args.hole_size = (2*degree-1)**2
                # model_path = "/home/katha/Documents/Uni/Thesis/spiralnet_plus/hole_filling/evaluation/GatedSkipHoleAE/checkpoints/checkpoint_50_2021-11-24-11-00-36.pt"
                #model_path = "/home/katha/Documents/Uni/Thesis/spiralnet_plus/hole_filling/evaluation/HoleAE/checkpoints/checkpoint_50_2021-11-24-11-25-17.pt"
            elif dataset == "8":
                patch_size = 8
                dataset_path = f"10000_0_{degree}x{degree}_{patch_size}x{patch_size}"
                args.hole_size = (6*degree-1)**2
                # model_path = "/home/katha/Documents/Uni/Thesis/spiralnet_plus/hole_filling/evaluation/GatedSkipHoleAE/checkpoints/checkpoint_50_2021-11-25-00-12-57.pt"
                #model_path = "/home/katha/Documents/Uni/Thesis/spiralnet_plus/hole_filling/evaluation/HoleAE/checkpoints/checkpoint_50_2021-11-25-07-10-21.pt"
            elif dataset == "m8":
                patch_size = 8
                dataset_path = f"10000_0_{degree}x{degree}_{patch_size}x{patch_size}_moving"
                args.hole_size = (2*degree-1)**2
            args.dataset = dataset_path

            meshdata, device, spiral_indices_list, down_transform_list, up_transform_list = preload(
                args
            )
            # # model from checkpoint
            # # model = HoleSkipAE(
            # #     args.in_channels, args.out_channels, args.latent_channels,
            # #     spiral_indices_list, down_transform_list,
            # #     up_transform_list, conv=GatedSpiralConv, hole_size=args.hole_size,
            # #     input_size=(patch_size*degree+1)**2
            # # ).to(device)
            # model = HoleAE(
            #     args.in_channels, args.out_channels, args.latent_channels,
            #     spiral_indices_list, down_transform_list,
            #     up_transform_list, conv=SpiralConv, hole_size=args.hole_size,
            #     input_size=(patch_size*degree+1)**2
            # ).to(device)
            # model.load_state_dict(torch.load(model_path)["model_state_dict"])
            # test_loader = DataLoader(
            #     meshdata.test_dataset, batch_size=args.batch_size)
            # print(c1_eval(model, test_loader, True, device))
            # continue
            for architecture in [Architecture.GatedSkipHoleAE, Architecture.HoleAE]:
                args.out_dir = osp.join(
                    args.work_dir, 'evaluation', architecture.value)
                os.makedirs(osp.join(args.out_dir, "meshes"), exist_ok=True)
                args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
                utils.makedirs(args.checkpoints_dir)

                args.architecture = architecture
                print(args)
                info, meshes = train_network(args)
                temp = vars(args)
                temp.update(info)
                print(temp)
                if first:
                    first = False
                    w.writerow(temp.keys())
                w.writerow(temp.values())
                for m in meshes:
                    om.write_mesh(
                        mesh=meshes[m], filename=osp.join(
                            args.out_dir, "meshes", f"{args.dataset}_{args.architecture.value}_test_{m}.obj"
                        )
                    )
        continue
exit()

with open('architectures_l1_layers_500_seq_len.csv', 'w') as f:
    w = csv.writer(f, delimiter=',')
    # for l in Loss:
    # for arch in Architecture:
    # for n_layers in range(2, 4):
    n_layers = 2
    # for length in range(1, 9, 2):
    for length in range(1, 25, 1):
        for latent in range(2, 8, 1):
            for outs in range(2, 6, 1):
                # for pool in range(1, 4):
                args.architecture = Architecture.GatedSkipAE  # arch
                args.loss = Loss.L1
                args.seq_length = [length]*n_layers
                args.out_channels = [2**(outs+n) for n in range(n_layers)]
                args.pooling = [2]*n_layers
                args.dilation = [1]*n_layers
                args.latent_channels = 2**latent
                print(args)
                info = train_network(args)
                temp = vars(args)
                temp.update(info)
                print(temp)
                if first:
                    first = False
                    w.writerow(temp.keys())
                w.writerow(temp.values())
