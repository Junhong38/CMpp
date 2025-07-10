from typing import Literal, Optional, List, Dict
import lightning as L
from torch.utils.data import DataLoader

from . import BreakingBad2PcsWeighted

class BreakingBad2PcsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str = 'data/data_shared_surface/data_pairwise_surface.hdf5',
        categories: List[str] = ['everyday', 'artifact'],
        num_points_to_sample: int = 8192,
        mesh_sample_strategy: Literal['uniform', 'poisson'] = 'uniform',
        sample_method: Literal['weighted'] = 'weighted',
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_root = data_root
        self.categories = categories
        self.num_points_to_sample = num_points_to_sample
        self.mesh_sample_strategy = mesh_sample_strategy
        self.sample_method = sample_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        if sample_method == 'weighted':
            self.dataset_cls = BreakingBad2PcsWeighted
        else:
            raise ValueError('Only weighted sampling is supported for 2pcs dataset')

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            datasets = []
            for cat in self.categories:
                ds = self.dataset_cls(
                    split='train',
                    data_root=self.data_root,
                    category=cat,
                    num_points_to_sample=self.num_points_to_sample,
                    mesh_sample_strategy=self.mesh_sample_strategy,
                )
                datasets.append(ds)
            # for two pieces, no need to concat across categories necessarily, but we will
            from torch.utils.data import ConcatDataset
            self.train_dataset = ConcatDataset(datasets)

            datasets = []
            for cat in self.categories:
                ds = self.dataset_cls(
                    split='val',
                    data_root=self.data_root,
                    category=cat,
                    num_points_to_sample=self.num_points_to_sample,
                    mesh_sample_strategy=self.mesh_sample_strategy,
                )
                datasets.append(ds)
            self.val_dataset = ConcatDataset(datasets)

        # Also prepare the val_dataset for test and predict stages
        if stage == 'test' or stage == 'predict':
            datasets = []
            for cat in self.categories:
                ds = self.dataset_cls(
                    split='val',
                    data_root=self.data_root,
                    category=cat,
                    num_points_to_sample=self.num_points_to_sample,
                    mesh_sample_strategy=self.mesh_sample_strategy,
                )
                datasets.append(ds)
            from torch.utils.data import ConcatDataset
            self.val_dataset = ConcatDataset(datasets)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return self.val_dataloader() 