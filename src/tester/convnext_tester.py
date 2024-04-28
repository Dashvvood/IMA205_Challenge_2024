import sys
import os
import motti
import numpy as np
import pandas as pd
from collections import OrderedDict

motti.append_parent_dir(__file__)
motti.append_current_dir(__file__)


from constant import PROJECT_ROOT

from args import opts
os.makedirs(opts.ckpt_dir, exist_ok=True)
os.makedirs(opts.log_dir, exist_ok=True)

from motti import load_json
import logging

import torch
from torch.utils.data import DataLoader
from dataset.ISIC2019 import ISIC2019Dataset
from model.ConvNeXtClassifier import LitConvNeXtClassifier


from transformers import (
    ConvNextConfig,
    ConvNextImageProcessor,
    ConvNextForImageClassification
)

import lightning as L


o_d = motti.o_d()
