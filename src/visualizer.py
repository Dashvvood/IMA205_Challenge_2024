import pandas as pd
import numpy as np
from constant import (
    INDEX2LABEL,
    LABEL2INDEX
)
import matplotlib
import matplotlib.pyplot as plt

from archive.utils import KaggleChallengeData


def viz_data_overview():
    D = KaggleChallengeData(
        metadata="../data/metadataTrain.csv",
        img_root="../data/Train/Train/"
    )

    fig, axs = plt.subplots(8, 8, figsize=(10, 10), constrained_layout=True)
    axsf = axs.flatten()

    for i, ind in enumerate(INDEX2LABEL):
        sub = D.df[D.df.CLASS == ind]
        IDs = sub.iloc[0:8].ID.values
        for j, ID in enumerate(IDs):
            sample = D.get_item_by_id(ID)
            ax = axs[i,j]
            ax.imshow(sample.image)
            ax.axis("off")
            
            if j == 0:
                ax.set_title(f"{sample.class_idx}: {sample.class_name}", loc='left')
                
    fig.suptitle("Dataset Overview", y=1)
    fig.save("Dataset_Overview.jpg")
    

def viz_testset():
    D = KaggleChallengeData(
        metadata="../data/metadataTest.csv",
        img_root="../data/Test/Test/"
    )
    fig, axs = plt.subplots(8, 8, figsize=(10, 10), constrained_layout=True)
    axsf = axs.flatten()

    IDs = D.df.iloc[:64].ID.values
    for j, ID in enumerate(IDs):
        sample = D.get_item_by_id(ID)
        ax = axsf[j]
        ax.imshow(sample.image)
        ax.axis("off")
        
    fig.suptitle("Test Set Overview", y=1.02)
    
    fig.savefig("testset.png")
    
if __name__ == 'main':
    viz_data_overview()
    