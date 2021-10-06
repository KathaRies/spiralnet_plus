from typing import Type
import openmesh as om
from torch_geometric.data.dataset import Dataset
from datasets import CoMA


class MeshData(object):
    def __init__(
        self,
        root,
        template_fp,
        transform=None,
        pre_transform=None,
        datast: Dataset = CoMA,
        dataset_kwargs={"split": 'interpolation',
                        "test_exp": 'bareteeth'}
    ):
        self.root = root
        self.template_fp = template_fp
        self.set = datast
        self.dataset_kwargs = dataset_kwargs
        self.transform = transform
        self.pre_transform = pre_transform
        self.train_dataset = None
        self.test_dataste = None
        self.template_points = None
        self.template_face = None
        self.mean = None
        self.std = None
        self.num_nodes = None

        self.load()

    def load(self):
        self.train_dataset = self.set(
            self.root,
            train=True,
            transform=self.transform,
            pre_transform=self.pre_transform,
            **self.dataset_kwargs
        )
        self.test_dataset = self.set(
            self.root,
            train=False,
            transform=self.transform,
            pre_transform=self.pre_transform,
            **self.dataset_kwargs
        )

        tmp_mesh = om.read_trimesh(self.template_fp)
        self.template_points = tmp_mesh.points()
        self.template_face = tmp_mesh.face_vertex_indices()
        self.num_nodes = self.train_dataset[0].num_nodes

        self.num_train_graph = len(self.train_dataset)
        self.num_test_graph = len(self.test_dataset)
        self.mean = self.train_dataset.data.x.view(
            self.num_train_graph, -1, 3
        ).mean(dim=0)
        self.std = self.train_dataset.data.x.view(
            self.num_train_graph, -1, 3
        ).std(dim=0)
        # self.normalize()

    def normalize(self):
        print('Normalizing...')
        self.train_dataset.data.x = (
            (self.train_dataset.data.x.view(self.num_train_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)
        self.test_dataset.data.x = (
            (self.test_dataset.data.x.view(self.num_test_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)
        print('Done!')

    def save_mesh(self, fp, x):
        x = x * self.std + self.mean
        om.write_mesh(fp, om.TriMesh(x.numpy(), self.template_face))
