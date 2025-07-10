import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data.breakingbad import DatasetBreakingBad
from data.fantasticbreaks import DatasetFantasticBreaks
class GADataset:

    @classmethod
    def initialize(cls, datapath, data_category, sub_category, min_part, max_part, n_pts, scale):
        cls.datapath = datapath
        cls.data_category = data_category
        cls.sub_category = sub_category
        cls.n_pts = n_pts
        cls.scale = scale
        cls.min_part = min_part
        cls.max_part = max_part

    @classmethod
    def build_dataloader(cls, batch_size, nworker, split, visualize=False):
        training = split == 'train'
        shuffle = training
        if cls.data_category == 'fantastic':
            dataset = DatasetFantasticBreaks(cls.datapath, cls.n_pts, visualize)
        else:
            dataset = DatasetBreakingBad(cls.datapath, cls.data_category, cls.sub_category, cls.min_part, cls.max_part, cls.n_pts, split, cls.scale, visualize)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nworker, pin_memory=False)

        return dataloader