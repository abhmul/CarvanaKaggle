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
from models import u_net_512 as model_func
from models import MODEL_FILE
from pyjet.training import GeneratorEnqueuer

# File paths
TEST_DIR = "../input/test/"
METADATA_PATH = "../input/metadata.csv"
SUBMISSION_PATH = "../submissions/"
MODEL_FILE = MODEL_FILE.format(model_func.__name__)
# MODEL_FILE = "../models/{name}-{loss}.h5".format(
#     name=model_func.__name__, loss="augmentation")

# Script stuff
ORIG_IMGSIZE = (1918, 1280)
IMGSIZE = (512, 512)
RESIZE = True
BATCH_SIZE = 16
SEED = 1234
np.random.seed(SEED)

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt
    from data import construct_img_path
    from debug_utils import examine_prediction


def infer_padding(img_size, orig_imgsize):
    space_x, space_y = img_size[1] - orig_imgsize[0], img_size[0] - orig_imgsize[1]
    left, top = int(space_x // 2), int(space_y // 2)
    right, bottom = space_x - left, space_y - top
    return (left, right, top, bottom)


def create_submission(model, datagen, orig_imgsize=(1918, 1280), resize=False, sort=True):
    # Create the rle
    timestr = time.strftime("%Y%m%d-%H%M%S")
    submission_name = os.path.join(SUBMISSION_PATH, 'carvana-sub-%s-%s.csv' %
                                   (model_func.__name__, timestr))
    print("Saving Submission file to: %s" % submission_name)
    # Open and memory efficiently write to the file
    with open(submission_name, 'w') as csvfile:
        # Initialize the header and writer
        fieldnames = ['img', 'rle_mask']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Loop through all the test samples and encode them
        for step in tqdm(range(datagen._generator.steps_per_epoch)):
            batch_img, batch_long_ids = next(datagen)
            batch_preds = model.predict_on_batch(batch_img)

            if resize:
                # Resize predictions
                zoom_factor = (1., orig_imgsize[1] / batch_preds.shape[1],
                               orig_imgsize[0] / batch_preds.shape[2])
                resized_batch_preds = ndi.interpolation.zoom(
                    batch_preds, zoom_factor, order=0, mode='nearest')
            else:
                # We need to remove the padding
                padding = infer_padding(batch_preds.shape[1:3], orig_imgsize)
                if padding != (0, 0, 0, 0):
                    left, right, top, bottom = padding
                    resized_batch_preds = batch_preds[:, top:batch_preds.shape[2] -
                                                      bottom, left:batch_preds.shape[2] - right]
                else:
                    resized_batch_preds = batch_preds
            # Threshold the image
            resized_batch_preds = np.rint(resized_batch_preds)

            assert resized_batch_preds.shape[1:] == (
                orig_imgsize[1], orig_imgsize[0]), "Resized images have wrong size of %s" % resized_batch_preds.shape

            for i in range(resized_batch_preds.shape[0]):
                sample_ind = step * BATCH_SIZE + i

                if DEBUG:
                    print("Index: ", sample_ind)
                    print("Expected Long ID: ", datagen._generator.dataset.long_ids[sample_ind])
                    print("Actual Long ID: ", batch_long_ids[i])
                    print("Resized: ", resize)
                    examine_prediction(datagen._generator.dataset.long_ids[sample_ind], batch_img[i],
                                       batch_preds[i], resized_batch_preds[i],
                                       orig_imgsize=orig_imgsize)

                assert datagen._generator.dataset.long_ids[sample_ind] == batch_long_ids[i], "Long IDs don't match! %s and %s" % (
                    datagen._generator.dataset.long_ids[sample_ind], batch_long_ids[i])
                # Write the row into csvfile
                writer.writerow({'img': datagen._generator.dataset.long_ids[sample_ind] + '.jpg', 'rle_mask': rle.rle_to_string(
                    rle.rle_encode(resized_batch_preds[i]))})
    if sort:
        print("Sorting CSVfile")
        df = pd.read_csv(submission_name)
        df = df.sort_values(by=['img'])
        df.to_csv(submission_name, index=False)
    print("Submission created at: %s" % submission_name)


# Load the test Dataset
test_dataset = FullCarDataset(
    base_dir=TEST_DIR, metadata_path=METADATA_PATH, img_size=IMGSIZE, resize=RESIZE)
print("Test Dataset: ", len(test_dataset), " samples")
# Turn it into a generator
test_gen = pyjet.DatasetGenerator(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Test Steps: ", test_gen.steps_per_epoch, " steps")
test_gen = GeneratorEnqueuer(test_gen)
test_gen.start(max_q_size=3)

# Create the model
model = model_func(test_dataset.img_size, train=False)
model.load_weights(MODEL_FILE)

# Create the submission
create_submission(model, test_gen, orig_imgsize=ORIG_IMGSIZE, resize=RESIZE)
