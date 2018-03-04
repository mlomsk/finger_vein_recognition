# -*- coding: utf-8 -*-
from abc import abstractmethod

import numpy as np


class ImagesReader:
    @abstractmethod
    def read(self, paths: list) -> (np.ndarray, list):
        pass


class MasksReader:
    @abstractmethod
    def read(self, paths: list) -> np.ndarray:
        pass
