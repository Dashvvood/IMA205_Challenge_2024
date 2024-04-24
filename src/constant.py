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

WEIGHT = 512
HEIGHT = 512

HF_CACHE_DIR = "/media/lupin/svp/lab/cache/hf/"
