import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR10, CIFAR100

from openselfsup.utils import build_from_cfg
from openselfsup.datasets.data_sources.cifar import Cifar
from openselfsup.datasets.registry import DATASETS, PIPELINES, DATASOURCES
from openselfsup.datasets.builder import build_datasource
from openselfsup.datasets.utils import to_numpy


@DATASETS.register_module
class StoragedBYOLDataset(Dataset):
    """Dataset for fix augmentation searching, towards population center."""

    def __init__(self, data_source, pipeline1, pipeline2, prefetch=False):
        self.data_source = build_datasource(data_source)
        pipeline1 = [build_from_cfg(p, PIPELINES) for p in pipeline1]
        self.pipeline1 = Compose(pipeline1)
        pipeline2 = [build_from_cfg(p, PIPELINES) for p in pipeline2]
        self.pipeline2 = Compose(pipeline2)
        self.storage = {}
        self.prefetch = prefetch

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        if idx not in self.storage:
            img = self.data_source.get_sample(idx)
            img1 = self.pipeline1(img)
            img2 = self.pipeline2(img)
            if self.prefetch:
                img1 = torch.from_numpy(to_numpy(img1))
                img2 = torch.from_numpy(to_numpy(img2))

            img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
            self.storage[idx] = img_cat
        else:
            img_cat = self.storage[idx]
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented


@DATASETS.register_module
class MultiAugBYOLDataset(Dataset):
    """Dataset for multi-augmentation ensemble bootstrapping."""

    def __init__(self, data_source, pipeline1, pipeline2, prefetch=False, num_pairs=4):
        self.num_pairs = num_pairs
        self.data_source = build_datasource(data_source)
        pipeline1 = [build_from_cfg(p, PIPELINES) for p in pipeline1]
        self.pipeline1 = Compose(pipeline1)
        pipeline2 = [build_from_cfg(p, PIPELINES) for p in pipeline2]
        self.pipeline2 = Compose(pipeline2)
        self.prefetch = prefetch

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        img1, img2 = [], []
        for i in range(self.num_pairs):
            img1.append(self.pipeline1(img).unsqueeze(0))
            img2.append(self.pipeline2(img).unsqueeze(0))
            if self.prefetch:
                img1 = torch.from_numpy(to_numpy(img1))
                img2 = torch.from_numpy(to_numpy(img2))

        img_cat = torch.cat((*img1, *img2), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented


@DATASOURCES.register_module
class NATSCifar10(Cifar):

    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ]

    def __init__(self, root, split, return_label=True):
        super().__init__(root, split, return_label)

    def set_cifar(self):
        try:
            self.cifar = CIFAR10(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download CIFAR10 manually from"
                            " https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4.")


@DATASOURCES.register_module
class NATSCifar100(Cifar):

    def __init__(self, root, split, return_label=True):
        super().__init__(root, split, return_label)

    def set_cifar(self):
        try:
            self.cifar = CIFAR100(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download CIFAR100 manually from"
                            " https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4.")
