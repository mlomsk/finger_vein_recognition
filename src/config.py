# -*- coding: utf-8 -*-
from params import Params

local = Params(train_path='/Users/ilya/Documents/machine_learning/veins/input/train/',
               test_path='/Users/ilya/Documents/machine_learning/veins/input/test/',
               tensorboard_dir='/tmp/tensorflow/',
               chekpoints_path='/Users/ilya/Documents/machine_learning/veins/output/',
               submission_dir='/Users/ilya/Documents/machine_learning/veins/submissions/',
               sample=5)

devbox = Params(train_path='/home/ilya/Data/veins/input/train/',
                test_path='/home/ilya/Data/veins/input/test/',
                tensorboard_dir='/home/ilya/Data/veins/tensorboard/',
                chekpoints_path='/home/ilya/Data/veins/output/',
                submission_dir='/home/ilya/Data/veins/submissions/',
                sample=None)
