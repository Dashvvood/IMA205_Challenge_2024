import sys
import os
import motti
motti.append_current_dir(__file__)
motti.append_parent_dir(__file__)

from os import PathLike
import torch
from torch.utils.data import Dataset, DataLoader 
from common import BaseDatasetMixin
from typing import Any
from dataclasses import dataclass

@dataclass
class ISIC2019Item:
    pixel_value: torch.FloatTensor = None
    label: int = None
    meta: object = None

@dataclass
class ISIC2019Batch:
    pixel_values: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    
class ISIC2019Dataset(BaseDatasetMixin, Dataset):
    def __init__(self, metadata: str | PathLike, img_root: str | PathLike, processor, in_memory=False) -> None:
        super().__init__(metadata, img_root)
        self.processor = processor
        
        # TODO: save all data in memory
        if in_memory:
            pass
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index) -> Any:
        meta = self.df.iloc[index]
        img = self._get_img_by_id(meta.ID)
        outputs = self.processor(img, return_tensors="pt")
        # TODO: label must substract 1, because in csv we start from 1, use -1 indicating the miss of label
        label = meta.CLASS - 1 if "CLASS" in meta else -1
        return ISIC2019Item(pixel_value=outputs["pixel_values"], label=label, meta=meta)

    @staticmethod
    def collate_fn(batch):
        return ISIC2019Batch(
            pixel_values=torch.cat([x.pixel_value for x in batch]), 
            labels=torch.tensor([x.label for x in batch])
        )
        

    def split_train_val(self):
        df = self.df
        val_df = df.groupby("CLASS").head(10)
        train_df = df[~df.index.isin(val_df.index)]

        train_set = ISIC2019Dataset(metadata=train_df, img_root=self.img_root, processor=self.processor)
        val_set = ISIC2019Dataset(metadata=val_df, img_root=self.img_root, processor=self.processor)

        return train_set, val_set
    