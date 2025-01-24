from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict
from functools import partial

from pytorch_lightning import LightningDataModule
# from torch.utils.data import DataLoader 
from torch_geometric.loader import DataLoader
from torch.utils.data import default_collate

from t2p_copy.dataset.t2p_dataset import T2PDataset


class jrdb_DataModule(LightningDataModule):
    def __init__(
        self,
        train_args: Dict,
        val_args: Dict,
        test_args: Dict,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super(jrdb_DataModule, self).__init__()
        self.train_args, self.val_args, self.test_args = {}, {}, {}
        for _arg in train_args: self.train_args.update(_arg)
        for _arg in val_args: self.val_args.update(_arg)
        for _arg in test_args: self.test_args.update(_arg)

        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = T2PDataset(self.train_args['dataset'], mode=0, input_time=self.train_args['input_time'])
        self.val_dataset = T2PDataset(self.val_args['dataset'], mode=1, input_time=self.val_args['input_time'])
        self.test_dataset = T2PDataset(self.test_args['dataset'], mode=1, input_time=self.test_args['input_time'])

    def custom_collate_fn(batch):
        # Move tensors to the default device
        import pdb;pdb.set_trace()
        batch = [(item[0].to('cuda'), item[1].to('cuda')) for item in batch]
        return default_collate(batch)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_args['bs'],
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            # collate_fn=self.custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_args['bs'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            # collate_fn=self.custom_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_args['bs'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            # collate_fn=self.custom_collate_fn
        )