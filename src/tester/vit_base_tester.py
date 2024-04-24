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
if not os.path.exists(opts.ckpt_dir):
    os.makedirs(opts.ckpt_dir, exist_ok=True)
if not os.path.exists(opts.log_dir):
    os.makedirs(opts.log_dir, exist_ok=True)
    
from motti import load_json
import logging

import torch
from torch.utils.data import DataLoader
from dataset.ISIC2019 import ISIC2019Dataset
from model.ViTClassifier import LitViTClassifier

from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification, 
    ViTConfig
)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

o_d = motti.o_d()

processor = ViTImageProcessor(**load_json(opts.processor_config))
vit_classifer = ViTForImageClassification(ViTConfig(**load_json(opts.vit_config)))

dataset = ISIC2019Dataset(
    metadata=opts.metadata, 
    img_root=opts.img_root,
    processor=processor
)

dataloader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=False, num_workers=16, collate_fn=ISIC2019Dataset.collate_fn)

model = LitViTClassifier(
    model=vit_classifer,
    criterion=torch.nn.CrossEntropyLoss(),
    lr = float(opts.lr)
)


if opts.ckpt != "" and os.path.exists(opts.ckpt):
    model = LitViTClassifier.load_from_checkpoint(opts.ckpt, model=vit_classifer, map_location=torch.device("cpu"))


trainer = L.Trainer(
    accelerator="gpu",
    devices=opts.device_num,
    fast_dev_run=opts.fast,
    max_epochs=opts.max_epochs,
    accumulate_grad_batches=opts.accumulate_grad_batches,
    log_every_n_steps=10,
)

predictions = trainer.predict(
    model=model,
    dataloaders=dataloader,
)

results = np.concatenate([x.numpy() for x in predictions])
L = len(results)

new_df = pd.DataFrame(data={"ID": dataset.df["ID"][:L], "CLASS": results})
summary = load_json("../data/Submission/summary.json")
summary = OrderedDict() if summary is  None else OrderedDict(summary)
summary[o_d] = opts

new_df.to_csv(f"../data/Submission/{o_d}.csv", index=False)

logging.info("Done")
