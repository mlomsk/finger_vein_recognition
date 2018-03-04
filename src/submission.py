# -*- coding: utf-8 -*-

from keras.engine import Model

from data import make_test_df, read_resize_images
from params import Params


def make_submission(model: Model, params: Params):
    tests_paths = make_test_df(params)
    test_imgs, sizes = read_resize_images(tests_paths)
    assert test_imgs.ndim == 4
    masks = model.predict(test_imgs)

    # TODO: save results