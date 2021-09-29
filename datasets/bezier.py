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
                 pre_filter=None):
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

        data_path = osp.join(self.raw_dir, 'train', 'data', 'data_{}.obj')
        label_path = osp.join(self.raw_dir, 'train', 'label', 'label_{}.obj')
        # TODO data with correct topology
        train_list = self._get_set_meshes(label_path, label_path)

        data_path = osp.join(self.raw_dir, 'test', 'data', 'data_{}.obj')
        label_path = osp.join(self.raw_dir, 'test', 'label', 'label_{}.obj')
        test_list = self._get_set_meshes(label_path, label_path)

        torch.save(self.collate(train_list), self.processed_paths[0])
        torch.save(self.collate(test_list), self.processed_paths[1])

        # shutil.rmtree(osp.join(self.raw_dir))

    def _get_set_meshes(self, data_path: str, label_path) -> List:
        mesh_list = []
        for i in range(100):
            data = read_mesh(data_path.format(i))
            #data.x = torch.transpose(data.x, 0, 1)
            label = read_mesh(label_path.format(i))
            data.y = label.x  # torch.transpose(label.x, 0, 1)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            mesh_list.append(data)
        return mesh_list