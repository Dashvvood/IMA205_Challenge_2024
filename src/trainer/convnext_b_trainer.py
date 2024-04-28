import sys
import os
import motti
from motti import load_json
thisfile = os.path.basename(__file__).split(".")[0]
motti.append_parent_dir(__file__)

from constant import PROJECT_ROOT, CLS_NUM_LIST, CLS_WEIGHT
from args import opts

os.makedirs(opts.ckpt_dir, exist_ok=True)
os.makedirs(opts.log_dir, exist_ok=True)
    
import logging
import torch
from torch.utils.data import DataLoader
from dataset.ISIC2019 import ISIC2019Dataset

import numpy as np

from transformers import (
    ConvNextConfig,
    ConvNextImageProcessor,
    ConvNextForImageClassification
)

from model.ConvNeXtClassifier import LitConvNeXtClassifier

from losses import LMFLoss

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

o_d = motti.o_d()

convnext_config = ConvNextConfig(**load_json(opts.model_config))
convnext = ConvNextForImageClassification(config=convnext_config)
processor = ConvNextImageProcessor(**load_json(opts.processor_config))


dataset = ISIC2019Dataset(
    metadata=opts.metadata, 
    img_root=opts.img_root,
    processor=processor
)

train_set, val_set = dataset.split_train_val()

train_dataloader = DataLoader(dataset=train_set, shuffle=True, batch_size=opts.batch_size, collate_fn=ISIC2019Dataset.collate_fn, num_workers=16)
val_dataloader = DataLoader(dataset=val_set, batch_size=8, collate_fn=ISIC2019Dataset.collate_fn, num_workers=16)

# criterion = torch.nn.CrossEntropyLoss()

criterion = LMFLoss(cls_num_list=CLS_NUM_LIST, weight=CLS_WEIGHT, gamma=1.5)

wandblogger = WandbLogger(
    name=f"{o_d}_{thisfile}_{opts.commit}", 
    save_dir=opts.log_dir, 
    project="isic2019",
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", 
    dirpath=os.path.join(opts.ckpt_dir, o_d),
    save_last=True,
    save_top_k=3,
    mode="min",
)

if opts.ckpt != "" and os.path.exists(opts.ckpt):
    model = LitConvNeXtClassifier.load_from_checkpoint(
        opts.ckpt, 
        model=convnext,
        criterion=torch.nn.CrossEntropyLoss(),
        lr = float(opts.lr),
        map_location=torch.device("cpu"),
    )
else:
    model = LitConvNeXtClassifier(
        model=convnext,
        criterion=torch.nn.CrossEntropyLoss(),
        lr = float(opts.lr)
    )
    
trainer = L.Trainer(
    accelerator="gpu",
    devices=opts.device_num,
    fast_dev_run=opts.fast,
    max_epochs=opts.max_epochs,
    logger=wandblogger,
    accumulate_grad_batches=opts.accumulate_grad_batches,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback],
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader
)

