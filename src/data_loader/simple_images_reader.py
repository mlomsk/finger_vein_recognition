# -*- coding: utf-8 -*-
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from data_loader.readers import ImagesReader


class SimpleImagesReader(ImagesReader):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.channels = 1

    def read(self, paths: list) -> (np.ndarray, list):
        imgs = np.zeros((len(paths), self.height, self.width, self.channels), dtype=np.uint8)
        sizes = []
        for i, file in tqdm(enumerate(paths), total=len(paths)):
            img = self.read_image(file)
            sizes.append((img.shape[0], img.shape[1]))
            imgs[i] = resize(img, (self.height, self.width), mode='constant', preserve_range=True)
        return imgs, sizes

    def read_image(self, file_path):
        return imread(file_path)[:, :, np.newaxis]
