import skimage.transform as skt
import cv2
from tqdm import tqdm
import numpy as np

def resize_picture(array, h, w, mask=False):
    result = []
    for picture in tqdm(array, total=array.shape[0]):
        if not mask:
            img_float = np.float32(picture)
            img_float = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
            img_float = skt.resize(img_float, (h, w, 1), mode='constant', preserve_range=True)
        else:
            img_float = np.float32(picture)
            img_float = skt.resize(img_float, (h, w, 1), mode='constant', preserve_range=True)
        result.append(img_float)
    return np.array(result)

def normalization(array):
    result = (array / 255.0) * 0.99 + 0.01
    return np.array(result)

