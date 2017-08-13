import time

import numpy as np
import pandas as pd
from scipy import ndimage

def rle_encode(mask_image, resize_factor=1.):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[:-1:2]
    return runs

def vect_rle_encode(mask_images):
    return [rle_encode(mask_images[i]) for i in range(mask_images.shape[0])]

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

### TESTS ###
TRAIN_MASKS_PATH = '../input/train_masks/'
TRAIN_MASKS_CSV_PATH = '../input/train_masks.csv'
def read_mask_image(car_code, angle_code):
    mask_img_path = TRAIN_MASKS_PATH + '/' + car_code + '_' + angle_code + '_mask.gif';
    mask_img = ndimage.imread(mask_img_path, mode = 'L')
    mask_img[mask_img <= 127] = 0
    mask_img[mask_img > 127] = 1
    return mask_img

def test_rle_encode_basic():
    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
    assert rle_to_string(rle_encode(test_mask)) == '7 2 11 2'

def test_rle_encode_train_data():
    train_masks = pd.read_csv(TRAIN_MASKS_CSV_PATH)
    num_masks = len(train_masks['img'])
    print('Verfiying RLE encoding on', num_masks, 'masks ...')
    time_read = 0.0 # seconds
    time_rle = 0.0 # seconds
    time_stringify = 0.0 # seconds
    for mask_idx in range(num_masks):
        img_file_name = train_masks.loc[mask_idx, 'img']
        car_code, angle_code = img_file_name.split('.')[0].split('_')
        t0 = time.clock()
        mask_image = read_mask_image(car_code, angle_code)
        time_read += time.clock() - t0
        t0 = time.clock()
        rle_truth_str = train_masks.loc[mask_idx, 'rle_mask']
        rle = rle_encode(mask_image)
        time_rle += time.clock() - t0
        t0 = time.clock()
        rle_str = rle_to_string(rle)
        time_stringify += time.clock() - t0
        assert rle_str == rle_truth_str
        if mask_idx and (mask_idx % 500) == 0:
            print('  ..', mask_idx, 'tested ..')
    print('Time spent reading mask images:', time_read, 's, =>', \
            1000*(time_read/num_masks), 'ms per mask.')
    print('Time spent RLE encoding masks:', time_rle, 's, =>', \
            1000*(time_rle/num_masks), 'ms per mask.')
    print('Time spent stringifying RLEs:', time_stringify, 's, =>', \
            1000*(time_stringify/num_masks), 'ms per mask.')
