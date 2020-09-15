from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, \
    Conv2DTranspose, concatenate, Input, UpSampling2D, Conv1D
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.random.set_seed(47)

class U_Net:
    @staticmethod
    def build(width, height, depth):


        inputs = Input((height, width, depth))
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        drop1 = Dropout(0.1)(conv1)
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(drop1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        drop2 = Dropout(0.5)(conv2)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(drop2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        drop3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(drop3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        drop4 = Dropout(0.5)(conv4)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        drop5 = Dropout(0.1)(conv5)
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(drop5)

        up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv5))
        merge6 = concatenate([drop4, up6])
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        drop6 = Dropout(0.15)(conv6)
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(drop6)

        up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7])
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        drop7 = Dropout(0.2)(conv7)
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(drop7)

        up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8])
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        drop8 = Dropout(0.1)(conv8)
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(drop8)

        up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9])
        conv9 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        drop9 = Dropout(0.2)(conv9)
        conv9 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(drop9)
        conv10 = Conv2D(1, (1, 1), activation='sigmoid',padding='same')(conv9)

        # inputs = Input((height, width, depth))
        # conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        # conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        # conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        # conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        # drop4 = Dropout(0.5)(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        #
        # conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        # conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        # drop5 = Dropout(0.5)(conv5)
        #
        # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(drop5))
        # merge6 = concatenate([drop4, up6])
        # conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        # conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        #
        # up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv6))
        # merge7 = concatenate([conv3, up7])
        # conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        # conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        #
        # up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv7))
        # merge8 = concatenate([conv2, up8])
        # conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        # conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        #
        # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     UpSampling2D(size=(2, 2))(conv8))
        # merge9 = concatenate([conv1, up9])
        # conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        # conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv10 = Conv2D(1, (1, 1), activation='sigmoid',padding='same')(conv9)

        model = Model(inputs, conv10)

        model.summary()

        return model

class Image_Generator:
    @staticmethod
    def build(x_train, y_train, batch_size=4, seed=47):

        data_gen_args = dict(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            rotation_range=15,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=False)

        X_datagen = ImageDataGenerator(**data_gen_args)
        Y_datagen = ImageDataGenerator(**data_gen_args)
        X_datagen.fit(x_train, augment=True, seed=seed)
        Y_datagen.fit(y_train, augment=True, seed=seed)
        X_train_augmented = X_datagen.flow(x_train, batch_size=batch_size, shuffle=True, seed=seed)
        Y_train_augmented = Y_datagen.flow(y_train, batch_size=batch_size, shuffle=True, seed=seed)

        # combine generators into one which yields image and masks
        train_generator = zip(X_train_augmented, Y_train_augmented)

        return train_generator


