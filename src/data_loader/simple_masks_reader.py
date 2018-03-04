# -*- coding: utf-8 -*-
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from data_loader.readers import MasksReader


class SimpleMasksReader(MasksReader):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def read(self, paths: list) -> np.ndarray:
        imgs = np.zeros((len(paths), self.height, self.width, 1), dtype=np.uint8)
        for i, file in tqdm(enumerate(paths), total=len(paths)):
            img = self.read_image(file)
            imgs[i] = resize(img, (self.height, self.width), mode='constant', preserve_range=True)
        return imgs

    def read_image(self, file_path):
        return imread(file_path, as_grey=True)[:, :, np.newaxis]
