import time
import os
import csv
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from tqdm import tqdm

from keras.preprocessing.image import load_img

import pyjet.data as pyjet
from data import FullCarDataset
import rle
from models import simple_model as model_func
from models import MODEL_FILE

# File paths
TRAIN_DIR = "../input/train/"
METADATA_PATH = "../input/metadata.csv"
SAVE_DIR = "../input/train_pred_masks/"
MODEL_FILE = MODEL_FILE.format(model_func.__name__)

# Script stuff
ORIG_IMGSIZE = (1918, 1280)
IMGSIZE = (128, 128)
BATCH_SIZE = 32
SEED = 1234
np.random.seed(SEED)

# Make the dataset
car_dataset = FullCarDataset(base_dir=BASE_DIR, metadata_path=METADATA_PATH,
                             img_size=IMGSIZE)

print("Complete Dataset: ", len(car_dataset), " samples")
