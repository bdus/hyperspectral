# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:02:21 2019

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

this_dir = osp.dirname(__file__)
model_path = osp.join(this_dir, '..', 'symbols')
add_path(model_path)

#from symbols import symbols
import symbols

import mxnet as mx
import numpy as np

from gluoncv import model_zoo as mzoo
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, loss as gloss

from indian_dataset import IndianDataset
num_epochs = 5
batch_size = 100
out_put_num = 16
dropout_rate=0.8
stochastic_ratio = 0.01
val_acc_bk = 0
ctx = mx.gpu()
#modelname = 'indian_try'
modelname = 'indian_simple2'
para_filepath = os.path.join(this_dir,'..','symbols','para','%s.params'%(modelname)) 
# dataset
train_data = gluon.data.DataLoader(dataset=IndianDataset(train=True), batch_size=batch_size ,shuffle=True,last_batch='rollover')
val_data = gluon.data.DataLoader(dataset=IndianDataset(train=False), batch_size=batch_size ,shuffle=False)

# g(x) : stochastic input augmentation function
#def g(x):
#    return x + nd.random.normal(0,stochastic_ratio,shape=x.shape)

# model 
basemodel_zoo = 'simple2'
net = symbols.get_model(basemodel_zoo)
#net_t = symbols.get_model(basemodel_zoo)

#net.initialize(mx.init.Xavier(magnitude=2.24))
net.initialize(mx.init.MSRAPrelu())
#net_t.initialize(mx.init.MSRAPrelu())
#net.initialize(mx.init.Normal(0.5) ,ctx=ctx)
#net.load_parameters(para_filepath)
#net_t.load_parameters(para_filepath)
net.collect_params().reset_ctx(ctx)
#net_t.collect_params().reset_ctx(ctx)

# solve
loss = gloss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()

def test():
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.reshape(-1,1,200)
        data = data.copyto(ctx)
        label = label.copyto(ctx)
        output = net(data)
        metric.update([label], [output])
    return metric.get()

def train(epochs,lr=0.1):
    val_acc_bk = 0
    trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
    for epoch in range(num_epochs):
        metric.reset()
        for i, (x, y) in enumerate(train_data): 
            x = x.reshape(-1,1,200)
            x = x.copyto(ctx)
            y = y.copyto(ctx)
            with autograd.record():
                output = net(x)
                L = loss(output,y)
                L.backward()
            trainer.step(batch_size)
            metric.update(y,output)
        
        if epoch % 50 == 0:
            name, acc = metric.get()
            print('[Epoch %d] Training: %s=%f'%(epoch, name, acc))
            name, val_acc = test()
            print('[Epoch %d] Validation: %s=%f'%(epoch, name, val_acc))
            if val_acc+acc > val_acc_bk:
                print('[Epoch %d] Validation: %s=%f'%(epoch, name, val_acc))
                net.save_parameters( os.path.join(this_dir,'..','symbols','para','%s%f.params'%(modelname, val_acc)) )                
                val_acc_bk = val_acc+acc
        if epoch % 1000 == 0:
            net.save_parameters( para_filepath ) 

if __name__ == '__main__':
    num_epochs = 10000
    #train(1000,0.2)
    train(1000,0.1)
    train(1600,0.01)
    train(2000,0.001)