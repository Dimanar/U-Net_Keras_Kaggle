import os
from os.path import join, exists
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import load
from numpy import save

PATH_TRAIN = 'data/train/'
PATH_TEST = 'data/test/'

def get_images(path, test=False, mask=False):
    if test:
        if exists("data/test_x.npy"):
            test = load("data/test_x.npy", allow_pickle=True)
        else:
            test = []
            for item in tqdm(os.listdir(path)):
                path_img = path + item + '/images/'
                img_ = cv2.imread(join(path_img, item + '.png'), 1)
                test.append(img_)
            test = np.array(test)
            save("data/test_x.npy", test)
        return test
    else:
        if exists("data/x_train.npy") and exists("data/y_train.npy"):
            X_train = load("data/x_train.npy", allow_pickle=True)
            y_train = load("data/y_train.npy", allow_pickle=True)
        else:
            X_train, y_train = [], []
            for item in tqdm(os.listdir(path)[:-1]):

                path_img = path + item + '/images/'
                img_ = cv2.imread(join(path_img, item + '.png'), 1)
                h, w, d = img_.shape
                X_train.append(img_)

                if mask:
                    msk_ = np.zeros((h, w, 1))
                    path_msk = path + item + '/masks/'
                    for file in next(os.walk(path_msk))[2]:
                        msk = cv2.imread(join(path_msk, file), -1)
                        msk = np.expand_dims(msk, axis=-1)
                        msk_ = np.maximum(msk_, msk)
                    y_train.append(msk_)

        X_train, y_train = np.array(X_train), np.array(y_train)
        save("data/x_train.npy", X_train)
        save("data/y_train.npy", y_train)
        if mask:
            return (X_train, y_train)
        else:
            return X_train

def get_shapes(test_folders, data):
    shapes = []
    for name in test_folders:
        select = data[data['ImageId'] == name][['Height', 'Width']]
        h, w = select.iloc[0, :]
        shapes.append((h, w))
    return shapes

def get_mask(array):
    array = list(map(lambda x: int(x), array.split(' ')))
    pixel, pixel_count = [], []
    [pixel.append(array[i]) if i % 2 == 0 else pixel_count.append(array[i]) for i in range(0, len(array))]
    pixels = [list(range(pixel[i], pixel[i] + pixel_count[i])) for i in range(0, len(pixel))]
    mask_pixels = sum(pixels, [])
    return mask_pixels

def get_mask_test(shapes, pictures):
    if exists('data/test_y.npy'):
        return np.load('data/test_y.npy', allow_pickle=True)
    else:
        images = []
        shapes = iter(shapes)
        for i, pct in enumerate(pictures):
            h, w = next(shapes)
            img_ = np.zeros((h*w, 1), dtype=int)
            mask_ = get_mask(pct)
            try:
                img_[mask_] = 255
            except IndexError as e:
                mask_.remove(max(mask_))
                img_[mask_] = 255
            img_ = np.reshape(img_, (w, h)).T
            img_ = img_.astype(np.float32)
            images.append(img_)
        np.save('data/test_y.npy', np.asarray(images))
        return np.array(images)

if __name__ == '__main__':
    (x_train, y_train) = get_images(PATH_TRAIN, mask=True)
    x_test = get_images(PATH_TEST, test=True)

    data = pd.read_csv('data/stage1_solution.csv')
    test_ids = next(os.walk(PATH_TEST))[1]
    picture = data.groupby('ImageId')['EncodedPixels'].apply(lambda pixels: ' '.join(pixels))
    shapes = get_shapes(test_ids, data)
    y_test = get_mask_test(shapes, picture)
