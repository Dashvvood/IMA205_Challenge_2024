from typing import Optional, List, Union, Any, Dict
import os
import pandas as pd
from PIL import Image
import numpy as np

class BaseDatasetMixin:
    def __init__(
        self,
        metadata: Union[str, os.PathLike, pd.DataFrame],
        img_root: Union[str, os.PathLike]
    ) -> None:
        if  not os.path.exists(img_root):
            raise FileNotFoundError
        
        self.metadada = metadata
        
        if isinstance(metadata, pd.DataFrame):
            self.df = metadata
        elif os.path.exists(metadata):
            self.df = pd.read_csv(metadata)
        else:
            raise ValueError
        
        self.img_root = img_root
        
    def _get_img_by_id(self, ID: str) -> Image.Image:
        path = os.path.join(self.img_root, ID+".jpg")
        img = Image.open(path)
        return img
    
    def _get_meta_by_id(self, ID: str) -> pd.Series:
        sample = self.df[self.df.ID == ID].iloc[0]
        return sample
    
    
    def _get_img_by_index(self, index) -> Image.Image:
        sample = self.df.iloc[index]
        img = self._get_img_by_id(sample.ID)
        return img
    
    