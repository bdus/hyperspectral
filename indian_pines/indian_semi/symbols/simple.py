#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 22:14:08 2018

@author: bdus

https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=hybridblock#mxnet.gluon.HybridBlock
https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/vgg.py

"""

__all__ = ['Simple','simple0','simple1','simple2']
import os
import os
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', 'libs')
add_path(lib_path)

from mxnet.gluon import HybridBlock, nn
import mxnet as mx
import mxnet.ndarray as nd

#class MLP(nn.Block):
#    def __init__(dasein,**kwargs):
#        super(MLP,dasein).__init__(**kwargs)
#        

class Simple(HybridBlock):
    def __init__(self,index=0,**kwargs):
        super(Simple, self).__init__(**kwargs)        
        # use name_scope to give child Blocks appropriate names.
        self.index = index
        with self.name_scope(): 
            self.output = nn.HybridSequential()
            if index == 0:
                self.output.add(
                        nn.Dense(16))
            elif index == 1:
                self.output.add(
                        nn.Dense(500),
						nn.Dense(256),
						nn.Dropout(0.8),
                        nn.Dense(16)
                    )
            elif index == 2:
                self.output.add(
                        nn.Conv1D(8,kernel_size=5,activation='relu'),
                        nn.Conv1D(16,kernel_size=5,activation='relu'),
                        nn.BatchNorm(momentum=0.8),
                        nn.MaxPool1D(pool_size=2),
                        nn.Conv1D(16,kernel_size=1,activation='relu'),
                        nn.Conv1D(16,kernel_size=5,activation='relu'),
                        nn.Flatten(),
                        nn.Dense(256,activation='relu'),
                        nn.Dropout(0.25),
                        nn.Dense(16,activation='relu')
                        )                
            else:
                pass           

    def hybrid_forward(self, F, x):
        if self.index == 2:
            x = x.reshape((-1,1,200))
        x = self.output(x)
        return x
    
    def __getitem__(self, key):
        return self.output[key]
    
    def __len__(self):
        return len(self.output)



def simple0(**kwargs):
    return get_simple(index=0,**kwargs)

def simple1(**kwargs):
    return get_simple(index=1,**kwargs)

def simple2(**kwargs):
    return get_simple(index=2,**kwargs)

def get_simple(index,pretrained=False,
               root=osp.join(this_dir, 'para'), **kwargs):
    net = Simple(index,**kwargs)
    if pretrained:
        #https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/vgg.py#L116
        filepath = os.path.join(root,'simple%d.params'%(index))
        print(filepath)
        net.load_parameters(filepath)
    return net