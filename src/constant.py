import os
import sys
import motti
motti.append_current_dir(__file__)
PROJECT_ROOT = motti.append_parent_dir(__file__)

from collections import OrderedDict
INDEX2LABEL= OrderedDict({
    1: "Melanoma",
    2: "Melanocytic nevus",
    3: "Basal cell carcinoma",
    4: "Actinic keratosis",
    5: "Benign keratosis",
    6: "Dermatofibroma",
    7: "Vascular lesion",
    8: "Squamous cell carcinoma",
})

NUM_LABEL = len(INDEX2LABEL)

LABEL2INDEX = OrderedDict()
for k, v in INDEX2LABEL.items():
    LABEL2INDEX[v] = k
    
INDEX2ABBR= OrderedDict({
    1: "MEL",
    2: "NV",
    3: "BCC",
    4: "AK",
    5: "BKL",
    6: "DF",
    7: "VASC",
    8: "SCC",
})

INDEX2WEIGHT = OrderedDict({
    1: 0.7005531,
    2: 0.24592265,
    3: 0.95261733,
    4: 3.64804147,
    5: 1.20674543,
    6: 13.19375,
    7: 12.56547619,
    8: 5.04219745,
})

CLS_NUM_LIST = [
    3391, # 1 Melanoma 3391
    9657, # 2 Melanocytic nevus 9657
    2492, # 3 Basal cell carcinoma 2492
    650, # 4 Actinic keratosis 650
    1968, # 5 Benign keratosis 1968
    179, # 6 Dermatofibroma 179
    190, # 7 Vascular lesion 190
    471, # 8 Squamous cell carcinoma 471   
]

WEIGHT = 512
HEIGHT = 512

HF_CACHE_DIR = "/media/lupin/svp/lab/cache/hf/"
