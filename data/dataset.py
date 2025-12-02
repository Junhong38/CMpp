from torch.utils.data import DataLoader
from data.breakingbad import DatasetBreakingBad, collate_fn


class GADataset:

    @classmethod
    def initialize(cls, datapath, data_category, sub_category, scale, multiplicity, min_part, max_part, min_n_pts, n_pts, overlap_radius):
        cls.datapath = datapath
        cls.data_category = data_category
        cls.sub_category = sub_category
        cls.scale = scale
        cls.multiplicity = multiplicity
        cls.min_part = min_part
        cls.max_part = max_part
        cls.min_n_pts = min_n_pts
        cls.n_pts = n_pts
        cls.overlap_radius = overlap_radius


    @classmethod
    def build_dataloader(cls, batch_size, nworker, split):
        shuffle = split == 'train'
        dataset = DatasetBreakingBad(cls.datapath, cls.data_category, cls.sub_category, split, cls.scale, cls.multiplicity, cls.min_part, cls.max_part, cls.min_n_pts, cls.n_pts, cls.overlap_radius)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nworker, pin_memory=False, collate_fn=collate_fn)
        return dataloader