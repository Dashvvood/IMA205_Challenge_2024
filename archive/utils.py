from typing import Optional, List, Union, Any, Dict
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dataclasses import dataclass

from constant import (
    INDEX2LABEL,
    LABEL2INDEX,
    WEIGHT,
    HEIGHT,
)

    
def __plt_imread(path) -> np.ndarray:
    img = plt.imread(path)
    img = img / 255 if img.dtype == "uint8" else img  
    return img


@dataclass
class KaggleChallengeDataOutput:
    ID: str 
    image: np.ndarray
    metadata: Dict


class KaggleChallengeData(object):
    def __init__(
        self,
        metadata: Union[str, os.PathLike],
        img_root: Union[str, os.PathLike]
    ) -> None:
        if (not os.path.exists(metadata)) or not (os.path.exists(img_root)):
            raise FileNotFoundError
        
        self.metadada = metadata
        self.df = pd.read_csv(metadata)
        self.img_root = img_root
        
        
    def _get_img_by_id(self, ID: str) -> np.ndarray:
        path = os.path.join(self.img_root, ID+".jpg")
        img = Image.open(path)
        img = img.resize(size=(WEIGHT, HEIGHT))  # (weight, height)
        return np.array(img, dtype=np.float32) / 255.0

    def get_item_by_id(self, ID: str):
        sample = self.df[self.df.ID == ID].iloc[0]
        img = self._get_img_by_id(sample.ID)
        return KaggleChallengeDataOutput(
            ID=sample.ID, image=img, metadata=dict(sample)
        )
    
    def __getitem__(self, key):
        sample = self.df.iloc[key]
        img = self._get_img_by_id(sample.ID)
        return KaggleChallengeDataOutput(
            ID=sample.ID, image=img, metadata=dict(sample)
        )
    
    
import torch
import torchvision.transforms as T

class Augment:
   """
   A stochastic data augmentation module
   Transforms any given data example randomly
   resulting in two correlated views of the same example,
   denoted x ̃i and x ̃j, which we consider as a positive pair.
   """

   def __init__(self, img_size, s=1):
       color_jitter = T.ColorJitter(
           0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
       )
       # 10% of the image
       blur = T.GaussianBlur((3, 3), (0.1, 2.0))

       self.train_transform = torch.nn.Sequential(
           T.RandomResizedCrop(size=img_size),
           T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
           T.RandomApply([color_jitter], p=0.8),
           T.RandomApply([blur], p=0.5),
           T.RandomGrayscale(p=0.2),
           # imagenet stats
           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       )

   def __call__(self, x):
       return self.train_transform(x), self.train_transform(x)
   