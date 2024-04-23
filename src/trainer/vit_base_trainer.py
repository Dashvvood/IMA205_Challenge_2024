import sys
import os
import motti

motti.append_parent_dir(__file__)

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

vit_config = ViTConfig(**load_json(opts.vit_config))
vit_classifer = ViTForImageClassification(vit_config)

dataset = ISIC2019Dataset(
    metadata=opts.metadata, 
    img_root=opts.img_root,
    processor=processor
)

train_set, val_set = dataset.split_train_val()

train_dataloader = DataLoader(dataset=train_set, batch_size=opts.batch_size, collate_fn=ISIC2019Dataset.collate_fn, num_workers=16)
val_dataloader = DataLoader(dataset=val_set, batch_size=8, collate_fn=ISIC2019Dataset.collate_fn, num_workers=16)


criterion = torch.nn.CrossEntropyLoss()
wandblogger = WandbLogger(
    name=o_d + "vit_base", 
    save_dir=opts.log_dir, 
    project="isic2019",
)


    
checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy", 
    dirpath=opts.ckpt_dir,
    save_last=True,
    save_top_k=1,
    mode="max",
)

model = LitViTClassifier(
    model=vit_classifer,
    criterion=torch.nn.CrossEntropyLoss(),
    lr = float(opts.lr)
)

if opts.ckpt != "" and os.path.exists(opts.ckpt):
    model.load_from_checkpoint(opts.ckpt)
    
trainer = L.Trainer(
    accelerator="gpu",
    devices=opts.device_num,
    fast_dev_run=opts.fast,
    max_epochs=opts.max_epochs,
    logger=wandblogger,
    accumulate_grad_batches=opts.accumulate_grad_batches,
    log_every_n_steps=10,
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader
)
