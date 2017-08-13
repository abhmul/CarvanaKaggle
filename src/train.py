import numpy as np
from pyjet.preprocessing.image import ImageDataGenerator
import pyjet.data as pyjet

from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

from plotter_callback import Plotter
from data import CarDataset
from models import u_net_full_img as model_func
from models import MODEL_FILE

# File path stuff
BASE_DIR = "../input/train/"
MASK_DIR = "../input/train_masks/"
METADATA_PATH = "../input/metadata.csv"
ORIGINAL_MODEL_FILE = MODEL_FILE.format(model_func.__name__)
MODEL_FILE = "../models/{name}-{loss}.h5".format(
    name=model_func.__name__, loss="binary_crossentropy")

# Script Settings
IMGSIZE = (1280, 1920)
BATCH_SIZE = 4
SPLIT = 0.2
SEED = 1234
np.random.seed(SEED)
LEARNING_RATE = 1e-4
OPTIMIZER = 'sgd'
MOMENTUM = 0.0
NESTEROV = False

DEBUG = True

# Load the data
car_dataset = CarDataset(base_dir=BASE_DIR, mask_dir=MASK_DIR, metadata_path=METADATA_PATH,
                         img_size=IMGSIZE, resize=False)
# Split the data
train_dataset, val_dataset = car_dataset.validation_split(split=SPLIT, shuffle=True,
                                                          seed=np.random.randint(10000))
# Validate over the entire Dataset
# val_dataset = val_dataset.to_full()

print("Complete Dataset: ", len(car_dataset), " samples")
print("Train Dataset: ", len(train_dataset), " samples")
print("Val Dataset: ", len(val_dataset), " samples")

# Create the data generators
train_gen = pyjet.DatasetGenerator(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   seed=np.random.randint(10000))
val_gen = pyjet.DatasetGenerator(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 seed=np.random.randint(10000))
# Set up the training augmentation
train_gen = ImageDataGenerator(train_gen, labels=True, augment_masks=True,
                               width_shift_range=0.3,
                               height_shift_range=0.3,
                               rotation_range=30,
                               horizontal_flip=True)

if DEBUG:
    from debug_utils import plot_img_mask_tiled
    for x, y in train_gen:
        plot_img_mask_tiled(x, y, (1, 2), ion=False)

print("Train Steps per Epoch: ", train_gen.steps_per_epoch, " steps")
print("Val Steps per Epoch: ", val_gen.steps_per_epoch, " steps")

if OPTIMIZER == 'sgd':
    optimizer = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=NESTEROV)
if OPTIMIZER == 'adam':
    optimizer = Adam(lr=LEARNING_RATE)
print("Optimizer: ", OPTIMIZER)
# This will save the best scoring model weights to the parent directory
best_model = ModelCheckpoint(MODEL_FILE, monitor='val_dice_coef', mode='max', verbose=1,
                             save_best_only=True, save_weights_only=True)
# This will plot the losses while training
plotter = Plotter(scale='log')
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1)
callbacks = [best_model, plotter]

# Create the model and fit it
model = model_func(car_dataset.img_size, optimizer=optimizer, loss='binary_crossentropy')
model.load_weights(ORIGINAL_MODEL_FILE)
fit = model.fit_generator(train_gen, steps_per_epoch=train_gen.steps_per_epoch,
                          epochs=3000, verbose=1, callbacks=callbacks,
                          validation_data=val_gen, validation_steps=val_gen.steps_per_epoch,
                          max_q_size=3)
