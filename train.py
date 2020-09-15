from __future__ import print_function
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from models.model import U_Net, Image_Generator
from models.preprocessing import normalization, resize_picture
from models.metric import dice_coef_loss, dice_coef

os.environ["KERAS_BACKEND"] = "tensorflow"

batch_size = 8
epochs = 10
verbose = 1

X = np.load('data/x_train.npy', allow_pickle=True)
Y = np.load('data/y_train.npy', allow_pickle=True)

X = normalization(resize_picture(X, 256, 256))
Y = normalization(resize_picture(Y, 256, 256, mask=True))

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=47
)

x_train = np.asarray(x_train).astype(np.float)
y_train = np.asarray(y_train).astype(np.float)


model_ = U_Net.build(256, 256, 1)
generator = Image_Generator.build(x_train, y_train)
model_.compile(loss=dice_coef_loss,
               optimizer=Adam(1e-4),
               metrics=[dice_coef])

history = model_.fit(generator, validation_data=(x_test, y_test),
                validation_steps=batch_size / 4,
                steps_per_epoch=len(x_train) / 2,
                epochs=epochs, verbose=1)

print(history.history.keys())
print(history.history.values())

def show_history(coef, val, label):
    plt.plot(coef)
    plt.plot(val)
    plt.title('model dice_coef')
    plt.ylabel(label)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('output/{}.png'.format(label))
    plt.show()


show_history(history.history['dice_coef'], history.history['val_dice_coef'],'dice_coef')

show_history(history.history['loss'], history.history['val_loss'], 'dice_coef_loss')

plot_model(model_, to_file='output/lighter_model_with_dropouts.png', show_shapes=True, show_layer_names=True)

model_json = model_.to_json()
with open("output/lighter_model_with_dropouts_10_epochs.json", "w") as json_file:
    json_file.write(model_json)

model_.save_weights("output/lighter_model_with_dropouts_10_epochs.h5")
print("Saved model to disk")
