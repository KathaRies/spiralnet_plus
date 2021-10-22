import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import openmesh as om
import numpy as np


def read_mesh(path):
    mesh = om.read_trimesh(path)
    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
    x = torch.tensor(mesh.points().astype('float32'))
    # x = torch.tensor(np.unique(mesh.points(), axis=0).astype('float32')) # correct dimensions
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index, face=face)
