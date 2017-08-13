import os
from glob import glob

import pyjet.data as pyjet
import pandas as pd
import numpy as np

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

ORIG_IMGSIZE = (1918, 1280)


def construct_img_path(long_id, dirpath, mask=False):
    fname = long_id + "_mask.gif" if mask else long_id + ".jpg"
    return os.path.join(dirpath, fname)


def extract_long_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def extract_id_from_long_id(long_id):
    return long_id.split("_")[0]


class FullCarDataset(pyjet.Dataset):

    def __init__(self, base_dir="", mask_dir="", long_ids=None, metadata_path=None, metadata=None, img_size=(256, 256), resize=False):
        super(FullCarDataset, self).__init__()

        # Initialize file stuff
        self.base_dir = base_dir
        self.mask_dir = mask_dir
        self.long_ids = np.array([])
        self.unique_ids = np.array([])
        self.metadata = metadata
        self.img_size = img_size
        self.resize = resize

        # If provided load the file stuff
        if long_ids is not None:
            self.long_ids = long_ids
        elif base_dir:
            # No IDs but have a base path, then infer
            self.long_ids = np.array([extract_long_id(fp)
                                      for fp in glob(os.path.join(base_dir, "*.jpg"))])
        else:
            raise ValueError("Must provide an array of long ids and/or a base_dir")
        self.unique_ids = np.unique([extract_id_from_long_id(long_id) for long_id in self.long_ids])

        if metadata is None:
            if metadata_path is not None:
                metadata = pd.read_csv(metadata_path)
                metadata.index = metadata['id']
                self.metadata = metadata.loc[self.unique_ids, :]
        else:
            self.metadata = metadata

    def __len__(self):
        return len(self.long_ids)

    def _construct_path(self, long_id, mask=False):
        dirpath = self.mask_dir if mask else self.base_dir
        return construct_img_path(long_id, dirpath, mask=mask)

    def load_img(self, long_id, mask=False):
        """Loads an image into PIL format.
        # Arguments
            long_id: id and angle of image
            mask: Boolean, whether to load the image as grayscale.
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.
        # Returns
            A PIL Image instance.
        # Raises
            ImportError: if PIL is not available.
        """
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')

        img = pil_image.open(self._construct_path(long_id, mask=mask))
        if mask:
            if img.mode != 'L':
                img = img.convert('L')
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
        if self.resize and self.img_size:
            hw_tuple = (self.img_size[1], self.img_size[0])
            if img.size != hw_tuple:
                img = img.resize(hw_tuple)
        return img

    @staticmethod
    def img_to_array(img, padding=(0, 0, 0, 0), mode='nearest'):
        """padding is (left, right, top, bottom)"""
        # print("Padding: ", padding)
        tmp = np.asarray(img)
        if tmp.ndim == 2:
            tmp = tmp[..., np.newaxis]
        elif tmp.ndim == 3:
            pass
        else:
            raise ValueError("Incorrect number of image dimensions: %s" % npimg.ndim)
        if padding != (0, 0, 0, 0):
            new_shape = (tmp.shape[0] + padding[2] + padding[3],
                         tmp.shape[1] + padding[0] + padding[1], tmp.shape[2])
            npimg = np.zeros(new_shape, dtype=np.uint8)
            npimg[padding[2]:npimg.shape[0] - padding[3],
                  padding[0]:npimg.shape[1] - padding[1]] = tmp
            # Clear the memory
            tmp = None
            # Fill the border if we need to
            if mode == 'nearest':
                npimg[:padding[2]] = npimg[padding[2]:padding[2] + 1]
                npimg[npimg.shape[0] - padding[3]:] = npimg[-padding[3] - 1:npimg.shape[0] - padding[3]]
                npimg[:, :padding[0]] = npimg[:, padding[0]:padding[0] + 1]
                npimg[:, npimg.shape[1] - padding[1]:] = npimg[:,
                                                               npimg.shape[1] - padding[1] - 1:-padding[1]]
            elif mode != 'constant':
                raise ValueError("Invalid mode")
        else:
            npimg = tmp
        return npimg

    def create_batch(self, batch_indicies):
        # Grab the batch ids
        batch_long_ids = self.long_ids[batch_indicies]
        return self._create_batch_from_ids(batch_long_ids)

    def _create_batch_from_ids(self, batch_long_ids):
        # Create the image batch
        batch_img = np.empty((len(batch_long_ids),) + self.img_size + (3,))
        if not self.resize and self.img_size:
            space_x, space_y = self.img_size[1] - \
                ORIG_IMGSIZE[0], self.img_size[0] - ORIG_IMGSIZE[1]
            left, top = int(space_x // 2), int(space_y // 2)
            right, bottom = space_x - left, space_y - top
            padding = (left, right, top, bottom)
        else:
            padding = (0, 0, 0, 0)
        for i, long_id in enumerate(batch_long_ids):
            batch_img[i] = self.img_to_array(self.load_img(long_id, mask=False), padding=padding)
            batch_img[i] /= 255.

        if self.mask_dir:
            # Create the mask batch
            batch_mask = np.empty((len(batch_long_ids),) + self.img_size)
            for i, long_id in enumerate(batch_long_ids):
                batch_mask[i] = self.img_to_array(self.load_img(
                    long_id, mask=True), padding=padding)[..., 0]
            # Threshold the masks
            batch_mask[batch_mask > 0.] = 1.
            return batch_img, batch_mask
        return batch_img, batch_long_ids


class CarDataset(FullCarDataset):

    def __init__(self, base_dir="", mask_dir="", long_ids=None, ids=None, metadata_path=None, metadata=None, img_size=(256, 256), resize=True):
        super(CarDataset, self).__init__(base_dir=base_dir, mask_dir=mask_dir, long_ids=long_ids,
                                         metadata_path=metadata_path, metadata=metadata,
                                         img_size=img_size, resize=resize)
        # Create the id mapping
        if ids is None:
            self.ids = {extract_id_from_long_id(long_id): [] for long_id in self.long_ids}
            for long_id in self.long_ids:
                self.ids[extract_id_from_long_id(long_id)].append(long_id)
        else:
            self.ids = ids
        assert np.all(np.sort(list(self.ids.keys())) ==
                      self.unique_ids), "ID mapping and unique ids don't match."

    def __len__(self):
        return len(self.ids)

    def to_full(self):
        return FullCarDataset(base_dir=self.base_dir, mask_dir=self.mask_dir,
                              long_ids=self.long_ids, metadata=self.metadata,
                              img_size=self.img_size, resize=self.resize)

    def create_batch(self, batch_indicies):
        # Grab the batch ids
        batch_ids = self.unique_ids[batch_indicies]
        # Choose the files for the batch
        batch_long_ids = [np.random.choice(self.ids[id_name]) for id_name in batch_ids]
        return self._create_batch_from_ids(batch_long_ids)

    def validation_split(self, split=0.2, shuffle=True, seed=None):
        split_ind = int(split * len(self))
        val_split = slice(split_ind)
        train_split = slice(split_ind, None)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            indicies = np.random.permutation(len(self))
            train_split = indicies[train_split]
            val_split = indicies[val_split]

        # Create the split datasets
        train_ids = {id_name: self.ids[id_name] for id_name in self.unique_ids[train_split]}
        train_long_ids = np.array([self.ids[id_name]
                                   for id_name in self.unique_ids[train_split]]).flatten()
        train_data = CarDataset(base_dir=self.base_dir, mask_dir=self.mask_dir, ids=train_ids,
                                long_ids=train_long_ids, metadata=self.metadata, img_size=self.img_size, resize=self.resize)
        val_ids = {id_name: self.ids[id_name] for id_name in self.unique_ids[val_split]}
        val_long_ids = np.array([self.ids[id_name]
                                 for id_name in self.unique_ids[val_split]]).flatten()
        val_data = CarDataset(base_dir=self.base_dir, mask_dir=self.mask_dir, ids=val_ids,
                              long_ids=val_long_ids, metadata=self.metadata, img_size=self.img_size, resize=self.resize)

        return train_data, val_data
