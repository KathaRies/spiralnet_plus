import argparse
import os.path as osp
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
#from torch_geometric.data import DataLoader
from torchsummary import summary

from psbody.mesh import Mesh

from utils import utils, mesh_sampling, writer, DataLoader
from datasets import BEZIER, meshdata

from hole_filling import run, SkipAE, eval_error

# Settings
parser = argparse.ArgumentParser(description='hole_filling')
parser.add_argument('--dataset', type=str, default='100_0_2x2_4x4')
# -1 cpu, else gpu idx
parser.add_argument('--device_idx', type=int, default=0)
parser.add_argument('--n_threads', type=int, default=4)

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[32, 64],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=32)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[5, 5], nargs='+')
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
parser.add_argument('--epochs', type=int, default=100)

# others
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

args.data_fp = osp.join(
    '/home/katha/Documents/Uni/Thesis/BezierGAN/datasets/', args.dataset
)
device = torch.device(
    'cuda:{}'.format(args.device_idx)
) if args.device_idx else torch.device('cpu')
torch.set_num_threads(args.n_threads)

args.work_dir = osp.dirname(osp.realpath(__file__))
args.out_dir = osp.join(args.work_dir, 'out')
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)

# Data
template_fp = osp.join(args.data_fp, 'raw', 'train',
                       'label', 'label_0.obj')
meshdata = meshdata.MeshData(
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
d = train_dataset[0]
# spiral_indices = preprocess_spiral(d.face.T, args.seq_length).to(device)
print(d)

# Network
# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
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

model = SkipAE(
    args.in_channels, args.out_channels, args.latent_channels,
    spiral_indices_list, down_transform_list,
    up_transform_list
).to(device)
print('Number of parameters: {}'.format(utils.count_parameters(model)))
print(model)
print(d.size())
#summary(model, input_size=(81, 3))
# x = torch.zeros((1, 81, 3)).to(device)  # d.x.to(device).unsqueeze(0)
#writer.add_graph(model, x)

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

run(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=args.epochs,
    optimizer=optimizer,
    scheduler=scheduler,
    writer=writer,
    device=device
)
eval_error(model, test_loader, device, meshdata, args.out_dir)
