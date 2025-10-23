import torch
from torch.utils.data import DataLoader

from data.breakingbad import DatasetBreakingBad

# [TODO] DATASET IS NOT CHECKED YET
from data.fantasticbreaks import DatasetFantasticBreaks
from data.ambiguous import DatasetAmbiguous

class GADataset:

    @classmethod
    def initialize(cls, datapath, data_category, sub_category, min_part, max_part, n_pts, scale, multiplicity, CMorigin_mode=False):
        cls.datapath = datapath
        cls.data_category = data_category
        cls.sub_category = sub_category
        cls.min_part = min_part
        cls.max_part = max_part
        cls.multiplicity = multiplicity
        cls.n_pts = n_pts
        cls.scale = scale
        cls.CMorigin_mode = CMorigin_mode


    @classmethod
    def build_dataloader(cls, batch_size, nworker, split, visualize=False):
        training = split == 'train'
        shuffle = training
        if cls.data_category == 'fantastic':
            dataset = DatasetFantasticBreaks(cls.datapath, cls.n_pts, visualize)
        elif cls.data_category == 'ambiguous':
            dataset = DatasetAmbiguous(cls.datapath, cls.data_category, cls.sub_category, cls.min_part, cls.max_part, cls.n_pts, split, cls.scale, visualize)
        else:
            dataset = DatasetBreakingBad(cls.datapath, cls.data_category, cls.sub_category, cls.min_part, cls.max_part, cls.n_pts, split, cls.scale, cls.multiplicity, cls.CMorigin_mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nworker, pin_memory=False)

        return dataloader