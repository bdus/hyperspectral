#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 18:08:22 2018

@author: bdus

"""

import os
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', 'libs')
add_path(lib_path)

this_dir = osp.dirname(__file__)
data_path = osp.join(this_dir, '..', 'data')


import mxnet as mx
import numpy as np

from gluoncv import model_zoo as mzoo
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, loss as gloss

from indian_dataset import IndianDataset

train_data = gluon.data.DataLoader(dataset=IndianDataset(train=True,one_hot=True), batch_size=10,shuffle=True,last_batch='rollover')
val_data = gluon.data.DataLoader(dataset=IndianDataset(train=False,one_hot=True,ctx=gpu()), batch_size=10,shuffle=False)

