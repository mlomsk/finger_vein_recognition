# -*- coding: utf-8 -*-
import os
from glob import glob

from sklearn.model_selection import train_test_split

from data_loader.readers import MasksReader, ImagesReader
from params import Params


def make_train_generator(params: Params):
    from data_loader.simple_images_reader import SimpleImagesReader
    from data_loader.simple_masks_reader import SimpleMasksReader

    return _make_train_generator(params, SimpleImagesReader(params.height, params.width),
                                 SimpleMasksReader(params.height, params.width))


def _make_train_generator(params: Params, images_reader: ImagesReader, masks_reader: MasksReader):
    """
    Find, read and build train and validation generators
    :param params:
    :return:
    """
    print("Loading data")
    X_train, Y_train = make_train_df(params)
    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=params.validation_size)
    train_generator, val_generator = generator(xtr, xval, ytr, yval, params.batch_size, images_reader, masks_reader)
    print("Data loaded")
    return train_generator, val_generator


def make_train_df(params: Params):
    """
    Load paths for train data set
    :param params:
    :return: list of pairs of image and list of masks for this image
    """
    train_root = params.train_path

    train_ids = list(get_image_ids(train_root))
    print("Find {} train_ids".format(len(train_ids)))

    X_train = [_img_path(i, train_root) for i in train_ids]
    Y_train = [_mask_paths(i, train_root) for i in train_ids]

    if params.sample:
        return X_train[:params.sample], Y_train[:params.sample]
    else:
        return X_train, Y_train


def make_test_df(params: Params):
    test_root = params.test_path
    test_ids = list(get_image_ids(test_root))
    print("Find {} test_ids".format(len(test_ids)))
    return [_img_path(i, test_root) for i in test_ids]


def generator(X_train, X_test, Y_train, Y_test, batch_size,
              images_reader: ImagesReader, masks_reader: MasksReader):
    from keras.preprocessing.image import ImageDataGenerator

    X_train, _ = images_reader.read(X_train)
    X_test, _ = images_reader.read(X_test)
    Y_train = masks_reader.read(Y_train)
    Y_test = masks_reader.read(Y_test)

    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=20.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1,
                         fill_mode='reflect')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(X_train)
    mask_datagen.fit(Y_train)
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=7)
    mask_generator = mask_datagen.flow(Y_train, batch_size=batch_size, seed=7)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(X_test)
    mask_datagen_val.fit(Y_test)
    image_generator_val = image_datagen_val.flow(X_test, batch_size=batch_size)
    mask_generator_val = mask_datagen_val.flow(Y_test, batch_size=batch_size)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator


def _img_path(img_id, path):
    return os.path.join(path, 'images', img_id) + '.bmp'


def _mask_paths(img_id, path):
    return os.path.join(path, 'masks', img_id) + '.bmp'


def get_image_ids(path):
    imgs_glob = os.path.join(path, 'images', '*.bmp')
    for img_path in glob(imgs_glob):
        fname = os.path.basename(img_path)
        idx = fname[:-4]
        yield idx
