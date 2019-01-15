#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 20:14:03 2018

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
num_epochs = 5
batch_size = 100
out_put_num = 16
modelname = 'indian_try'
para_filepath = os.path.join(this_dir,'..','symbols','para','%s.params'%(modelname)) 
# dataset
train_data = gluon.data.DataLoader(dataset=IndianDataset(train=True), batch_size=batch_size ,shuffle=True,last_batch='rollover')
val_data = gluon.data.DataLoader(dataset=IndianDataset(train=False), batch_size=batch_size ,shuffle=False)

# model 
net = nn.Sequential()
net.add(
        nn.Dense(128,activation='sigmoid'),
        nn.Dense(64,activation='sigmoid'),
        nn.Dense(out_put_num,activation='sigmoid')
    )
#net.initialize(mx.init.Xavier(magnitude=2.24))
net.load_parameters(para_filepath)

# solve
loss = gloss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()

def test():
    metric = mx.metric.Accuracy()
    for data, label in val_data:               
        output = net(data)
        metric.update([label], [output])
    return metric.get()

def train(epochs,lr=0.1):
    trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
    for epoch in range(num_epochs):
        metric.reset()
        for i, (x, y) in enumerate(train_data):
            
            with autograd.record():
                output = net(x)
                L = loss(output,y)
                L.backward()
            trainer.step(batch_size)
            metric.update(y,output)
        
        if epoch % 10 == 0:
            name, acc = metric.get()
            print('[Epoch %d] Training: %s=%f'%(epoch, name, acc))
            name, val_acc = test()
            print('[Epoch %d] Validation: %s=%f'%(epoch, name, val_acc))
            net.save_parameters( para_filepath )            

if __name__ == '__main__':
    num_epochs = 1000
    train(num_epochs)