# -*- coding: utf-8 -*-
import os

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam

from metrics import dice_coef, dice_coef_loss, jacard_coef
from params import Params


class UNetModel:
    def __init__(self, params: Params):
        self.params = params
        self.name = 'UNet'
        self.model = self._build_model()

    def _build_model(self) -> Model:
        IMG_WIDTH = None
        IMG_HEIGHT = None
        IMG_CHANNELS = 1
        OUTPUT_MASK_CHANNELS = 1
        f = 16

        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x / 255)(inputs)

        c1 = Conv2D(f, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(f, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(f * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(f * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(f * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(f * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(f * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(f * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(f * 16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(f * 16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = Conv2DTranspose(f * 8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(f * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(f * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(f * 4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(f * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(f * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(f * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(f * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(f * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(f, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(f, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(c9)
        outputs = BatchNormalization(axis=3)(outputs)
        outputs = Activation('sigmoid')(outputs)

        model = Model(name=self.name, inputs=[inputs], outputs=[outputs])
        model.summary()
        return model

    def train(self, train_gen, validation_gen):
        self.model.compile(optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics=[jacard_coef, dice_coef])

        return self.model.fit_generator(
            generator=train_gen,
            steps_per_epoch=self.params.steps_per_epoch,
            epochs=self.params.epochs,
            validation_data=validation_gen,
            validation_steps=self.params.validation_steps_per_epoch,
            callbacks=[TensorBoard(
                log_dir=self.params.tensorboard_dir,
                batch_size=self.params.batch_size
            ), ModelCheckpoint(
                os.path.join(self.params.chekpoints_path,
                             "weights-improvement-{epoch:02d}-{val_dice_coef:.2f}.hdf5"),
                monitor='val_dice_coef',
                verbose=1,
                save_best_only=False
            )])
