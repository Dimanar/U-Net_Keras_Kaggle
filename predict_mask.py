from __future__ import print_function as pf

import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.transform import resize
import keras.backend as K

from models.metric import dice_coef_loss, dice_coef
from models.preprocessing import resize_picture, normalization

np.random.seed(42)
path = 'data/test/'

test_x = np.load('data/test_x.npy', allow_pickle=True)
test_x = resize_picture(test_x, 256, 256)
test_x = normalization(test_x)
test_x = np.asarray(test_x).astype(np.float)

json_file = open('output/lighter_model_with_dropouts_10_epochs.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('output\lighter_model_with_dropouts_10_epochs.h5')
loaded_model.compile(loss=dice_coef_loss, optimizer=Adam(1e-4),
                    metrics=[dice_coef])


result = loaded_model.predict(test_x, verbose=2)
np.save('data/predicted_mask.npy', result)


test_y = np.load('data/test_y.npy', allow_pickle=True)
def reshape(images, h, w):
    result = []
    for picture in images:
        img_float = np.float32(picture)
        img_float = resize(img_float, (h, w, 1), mode='constant', preserve_range=True)
        result.append(img_float)
    return np.array(result)

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join([str(x) for x in run_lengths])

test_y = reshape(test_y, 256, 256)
test_y = normalization(test_y)
test_y = np.asarray(test_y).astype(np.float32)

data = pd.DataFrame(columns=['IndexId', 'dice_coef', 'encoded_pixels'])
name = next(os.walk('data/test/'))[1]
name = iter(name)
for i, pred in enumerate(result):
    data.loc[i] = next(name), K.get_value(dice_coef(test_y[i], pred)), rle_encoding(pred)

print(data['dice_coef'].describe())
data.to_csv("output/test_first.csv")