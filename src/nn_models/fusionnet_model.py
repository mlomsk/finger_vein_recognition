# -*- coding: utf-8 -*-
import os

from keras import backend as K
from keras import layers
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Layer
from keras.layers import Input, BatchNormalization, Activation, regularizers, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Lambda
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

from metrics import mean_iou, dice_coef
from params import Params


class FusionNetModel:
    """
    https://arxiv.org/pdf/1612.05360.pdf
    """

    def __init__(self, params: Params):
        self.params = params
        self.name = 'FusionNet'
        self.model = self._build_model()

    def _triple_conv(self, input_tensor, filters, stage, activation: str):
        with K.name_scope(name='triple_{}'.format(stage)):
            x = Conv2D(filters=filters,
                       kernel_size=(3, 3),
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(l=0.0001))(input_tensor)
            x = BatchNormalization()(x)
            if activation == 'LeakyReLU':
                x = LeakyReLU(alpha=0.2)(x)
            else:
                x = Activation('relu')(x)

            x = Conv2D(filters=filters,
                       kernel_size=(3, 3),
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(l=0.0001))(x)
            x = BatchNormalization()(x)
            if activation == 'LeakyReLU':
                x = LeakyReLU(alpha=0.2)(x)
            else:
                x = Activation('relu')(x)

            x = Conv2D(filters=filters,
                       kernel_size=(3, 3),
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(l=0.0001))(x)
            x = BatchNormalization()(x)
            return x

    def _conv(self, input_tensor, filters, activation) -> Layer:
        x = Conv2D(filters=filters,
                   kernel_size=(3, 3),
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001))(input_tensor)
        x = BatchNormalization()(x)
        if activation == 'LeakyReLU':
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = Activation('relu')(x)
        return x

    def _conv_trans(self, input_tensor, filters, activation: str) -> Layer:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding='same')(input_tensor)
        x = BatchNormalization()(x)
        if activation == 'LeakyReLU':
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = Activation('relu')(x)
        return x

    def _residual(self, input_tensor, filters, activation, stage) -> Layer:
        with K.name_scope(name='residual_{}'.format(stage)):
            conv_1 = self._conv(input_tensor, filters, activation)
            conv_2 = self._triple_conv(conv_1, filters, stage, activation)
            x = layers.add([conv_1, conv_2])
            conv_3 = self._conv(x, filters, activation)
            return conv_3

    def _build_model(self) -> Model:
        IMG_WIDTH = 256
        IMG_HEIGHT = 256
        IMG_CHANNELS = 3
        OUTPUT_MASK_CHANNELS = 1
        filters = 16

        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x / 255)(inputs)

        c1 = self._residual(s, filters, 'LeakyReLU', 1)
        p1 = MaxPooling2D((2, 2))(c1)
        c2 = self._residual(p1, filters * 2, 'LeakyReLU', 2)
        p2 = MaxPooling2D((2, 2))(c2)
        c3 = self._residual(p2, filters * 4, 'LeakyReLU', 3)
        p3 = MaxPooling2D((2, 2))(c3)
        c4 = self._residual(p3, filters * 8, 'LeakyReLU', 4)
        p4 = MaxPooling2D((2, 2))(c4)

        c5 = self._residual(p4, filters * 16, 'LeakyReLU', 5)

        u6 = self._conv_trans(c5, filters * 8, 'relu', )
        u6 = concatenate([u6, c4], axis=3)
        c6 = self._residual(u6, filters * 8, 'relu', 6)
        u7 = self._conv_trans(c6, filters * 4, 'relu')
        u7 = concatenate([u7, c3], axis=3)
        c7 = self._residual(u7, filters * 4, 'relu', 7)
        u8 = self._conv_trans(c7, filters * 2, 'relu')
        u8 = concatenate([u8, c2], axis=3)
        c8 = self._residual(u8, filters * 2, 'relu', 8)
        u9 = self._conv_trans(c8, filters, 'relu')
        u9 = concatenate([u9, c1], axis=3)
        c9 = self._residual(u9, filters, 'relu', 9)

        outputs = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(c9)
        outputs = BatchNormalization(axis=3)(outputs)

        outputs = Activation('sigmoid')(outputs)

        model = Model(name=self.name, inputs=[inputs], outputs=[outputs])
        model.summary()
        return model

    def train(self, train_gen, validation_gen):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou, dice_coef])

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
                             "weights-improvement-{epoch:02d}-{val_mean_iou:.2f}.hdf5"),
                monitor='val_mean_iou',
                verbose=1,
                save_best_only=False,
                mode='max'
            )])
