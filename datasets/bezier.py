import os
import os.path as osp
import shutil
from typing import List

import torch
from torch_geometric.data import InMemoryDataset, extract_zip

from utils import read_mesh

# data hireachy
# -/home/katha/Documents/Uni/Thesis/BezierGAN/datasets/
#   - 100_0_2x2_4x4
#       - raw
#          - train
#          - test
#              - data
#              - label


class BEZIER(InMemoryDataset):

    # url = 'http://faust.is.tue.mpg.de/'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 mask_size=(9, 3)):
        self.mask_size = mask_size
        super(BEZIER, self).__init__(
            root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'train'

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            f'Dataset not found. Please be sure that {self.raw_file_names} is at {self.raw_dir}'
        )

    def process(self):
        # extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        mask = torch.ones((self.mask_size[0], self.mask_size[0], 1))
        m = self.mask_size[1]
        mask[m:-m, m:-m] = 0

        data_path = osp.join(self.raw_dir, 'train', 'data')
        label_path = osp.join(self.raw_dir, 'train', 'label')
        train_list = self._get_set_meshes(
            data_path, label_path, mask=mask.flatten(end_dim=-2)
        )

        data_path = osp.join(self.raw_dir, 'test', 'data')
        label_path = osp.join(self.raw_dir, 'test', 'label')
        test_list = self._get_set_meshes(
            data_path, label_path, mask.flatten(end_dim=-2))

        torch.save(self.collate(train_list), self.processed_paths[0])
        torch.save(self.collate(test_list), self.processed_paths[1])

        # shutil.rmtree(osp.join(self.raw_dir))

    def _get_set_meshes(self, data_path: str, label_path, mask) -> List:
        mesh_list = []
        path_list = os.listdir(data_path)
        for d_path in path_list:
            data = read_mesh(osp.join(data_path, d_path))
            i = d_path.split(".")[0].split("_")[-1]
            #data.x = torch.transpose(data.x, 0, 1)
            l_path = osp.join(label_path, 'label_{}.obj')
            label = read_mesh(l_path.format(i))
            data.y = label.x  # torch.transpose(label.x, 0, 1)
            mask = torch.sum(
                data.x, dim=(-1), keepdim=True).type(torch.bool).type(torch.float)[:, 0:1]
            data.mask = mask
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            mesh_list.append(data)
        return mesh_list
