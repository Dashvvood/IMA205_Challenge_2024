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
import numpy as np

from transformers import (
    ConvNextConfig,
    ConvNextImageProcessor,
    ConvNextForImageClassification
)

import lightning as L


o_d = motti.o_d()
convnext_config = ConvNextConfig(**load_json(opts.model_config))
convnext = ConvNextForImageClassification(config=convnext_config)
processor = ConvNextImageProcessor(**load_json(opts.processor_config))
dataset = ISIC2019Dataset(
    metadata=opts.metadata, 
    img_root=opts.img_root,
    processor=processor
)
dataloader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=False, num_workers=16, collate_fn=ISIC2019Dataset.collate_fn)

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
    accumulate_grad_batches=opts.accumulate_grad_batches,
    log_every_n_steps=10,
)

predictions = trainer.predict(
    model=model,
    dataloaders=dataloader,
)

results = np.concatenate([x.numpy() for x in predictions]) + 1
L = len(results)

new_df = pd.DataFrame(data={"ID": dataset.df["ID"][:L], "CLASS": results})
summary = load_json("../data/Submission/summary.json")
summary = OrderedDict() if summary is  None else OrderedDict(summary)
summary[o_d] = vars(opts)
motti.dump_json(summary, "../data/Submission/summary.json")
new_df.to_csv(f"../data/Submission/{o_d}.csv", index=False)

logging.info("Done")

