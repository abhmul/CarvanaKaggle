import argparse
import numpy as np
from pyjet.preprocessing.image import ImageDataGenerator
import pyjet.data as pyjet

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from data import CarDataset, FullCarDataset
from models import u_net_1024 as model_func

# CommandLine Flags
# TODO Add the model func choosing to the ArgumentParser
parser = argparse.ArgumentParser(
    description='Train on a model for Kaggle Carvana Image masking competition.')
# File path stuff
parser.add_argument(
    '--base_dir', help="The directory where the train data is stored", default="../input/train/")
parser.add_argument('--mask_dir', help="The directory where the train mask data is stored",
                    default="../input/train_masks/")
parser.add_argument('--metadata_path', help="The file path to the metadata csv file",
                    default="../input/metadata.csv")
parser.add_argument('--model_file', help="The save path for the model to train",
                    default="../models/{name}.h5".format(name=model_func.__name__))
parser.add_argument('-w', '--load_weights', help="Weights to initialize model with")
# Script Settings
parser.add_argument('-f', '--full', action='store_true',
                    help="Will split the full dataset instead of by ids.")
parser.add_argument('-i', '--img_size', help="The size of the input image to the model as "
                                             "(nCOL, nROW)",
                    type=tuple, default=(1024, 1024))
parser.add_argument('-p', '--pad', action="store_true",
                    help="Will pad the image instead of resizing it to the input image size.")
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help="The batch size to train the model with.")
parser.add_argument('--split', type=float, default=0.2,
                    help="What portion of the data to seperate for validation.")
parser.add_argument('--unshuffle', action='store_true',
                    help="Will not shuffle before splitting into the train and validation "
                    "datasets.")
parser.add_argument('-s', '--seed', type=int, default=6379,
                    help="Seed for the random number generator (used for reproducibility)")
parser.add_argument('--plotter', action='store_true',
                    help="Will plot the loss and val loss during training.")
# Augmentation Settings
parser.add_argument('--no_augmentation', action='store_true',
                    help="Turns of input augmentation during training")
parser.add_argument('--width_shift_range', type=float, default=0.3,
                    help="Maximum random shift of input image and mask along x-axis")
parser.add_argument('--height_shift_range', type=float, default=0.3,
                    help="Maximum random shift of input image and mask along y-axis")
parser.add_argument('--rotation_range', type=float, default=45.0,
                    help="Maximum random degrees of rotation allowed")
parser.add_argument('--no_horizontal_flip', action='store_true',
                    help="Will not randomly flip the image over the y-axis")
parser.add_argument('--zoom_range', type=float, default=0.2,
                    help="Will randomly rescale the image in interval (1-z, 1+z)")
parser.add_argument('--fill_mode', default='constant',
                    help="Decides how pixels beyod image boundary are filled.")
# Optimizer settings
parser.add_argument('-l', '--learning_rate', type=float, default=0.01,
                    help="Learning rate for SGD optimizer")
parser.add_argument('-m', '--momentum', type=float, default=0.9, help="Momentum for SGD optimizer")
parser.add_argument('-n', '--nesterov', action='store_true',
                    help="Whether or not to use Nesterov Accelerated Gradients for SGD optimizer")
# Training settings
parser.add_argument('-e', '--epochs', type=int, default=9999,
                    help="Number of epochs to train for, defualts to a very large number.")
parser.add_argument('--train_verbosity', type=int, default=1,
                    choices=[0, 1, 2], help="Verbosity to use for training the model.")
parser.add_argument('-q', '--max_q_size', type=int, default=3,
                    help="Size of data generator queue for asynchronous generation.")
parser.add_argument('--initial_epoch', type=int, default=0, help="Epoch to start training from.")
# Debug settings
parser.add_argument('-d', '--debug', action='store_true', help="Runs the script in debug mode")

args = parser.parse_args()

if __name__ == '__main__':
    # Load the data
    if args.full:
        car_dataset = FullCarDataset(base_dir=args.base_dir, mask_dir=args.mask_dir,
                                     metadata_path=args.metadata_path, img_size=args.img_size,
                                     resize=(not args.pad))
    else:
        car_dataset = CarDataset(base_dir=args.base_dir, mask_dir=args.mask_dir,
                                 metadata_path=args.metadata_path, img_size=args.img_size,
                                 resize=(not args.pad))
    # Split the data
    train_dataset, val_dataset = car_dataset.validation_split(
        split=args.split, shuffle=(not args.unshuffle), seed=np.random.randint(10000))
    if not args.full:
        # Train over the entire Dataset
        train_dataset = train_dataset.to_full()
        # Validate over the entire Dataset
        val_dataset = val_dataset.to_full()

    print("Complete Dataset: ", len(car_dataset), " samples")
    print("Train Dataset: ", len(train_dataset), " samples")
    print("Val Dataset: ", len(val_dataset), " samples")

    # Create the data generators
    train_gen = pyjet.DatasetGenerator(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       seed=np.random.randint(10000))
    val_gen = pyjet.DatasetGenerator(val_dataset, batch_size=args.batch_size, shuffle=True,
                                     seed=np.random.randint(10000))
    # Set up the training augmentation
    if not args.no_augmentation:
        train_gen = ImageDataGenerator(train_gen, labels=True, augment_masks=True,
                                       width_shift_range=args.width_shift_range,
                                       height_shift_range=args.height_shift_range,
                                       rotation_range=args.rotation_range,
                                       horizontal_flip=(not args.no_horizontal_flip),
                                       zoom_range=args.zoom_range,
                                       fill_mode=args.fill_mode)

    if args.debug:
        from debug_utils import plot_img_mask_tiled
        for i, (x, y) in enumerate(train_gen):
            plot_img_mask_tiled(x, y, ion=False)
            if i >= 5:
                break
        for i, (x, y) in enumerate(val_gen):
            plot_img_mask_tiled(x, y, ion=False)
            if i >= 5:
                break
        train_gen.restart()
        val_gen.restart()

    print("Train Steps per Epoch: ", train_gen.steps_per_epoch, " steps")
    print("Val Steps per Epoch: ", val_gen.steps_per_epoch, " steps")

    optimizer = SGD(lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov)
    # This will save the best scoring model weights to the parent directory
    best_model = ModelCheckpoint(args.model_file, monitor='val_dice_loss', mode='max', verbose=1,
                                 save_best_only=True, save_weights_only=True)
    # This will decrease the learning rate everytime we stop learning
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_loss', factor=0.1,
                                  patience=4, verbose=1, epsilon=1e-4, mode='max')
    callbacks = [best_model, reduce_lr]
    if args.plotter:
        from plotter_callback import Plotter
        # This will plot the losses while training
        plotter = Plotter(scale='log')
        callbacks.append(plotter)

    # Create the model and fit it
    model = model_func(car_dataset.img_size, optimizer=optimizer)
    # Load the initialization weights if given
    if args.load_weights:
        model.load_weights(args.load_weights)
    # Train the model
    fit = model.fit_generator(train_gen, steps_per_epoch=train_gen.steps_per_epoch,
                              epochs=args.epochs, verbose=args.train_verbosity, callbacks=callbacks,
                              validation_data=val_gen, validation_steps=val_gen.steps_per_epoch,
                              max_q_size=args.max_q_size, initial_epoch=args.initial_epoch)
