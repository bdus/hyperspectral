#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:03:46 2018

@author: bdus

@Description:
    prototype: [MNIST](https://mxnet.incubator.apache.org/_modules/mxnet/gluon/data/vision/datasets.html#MNIST)
               & MNISTcsv via https://github.com/bdus/programpractice
    get indian dataset
"""
import os
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', 'data')
add_path(lib_path)
#print('---------------------------',lib_path)

import numpy as np
import mxnet as mx
from mxnet.gluon.data import dataset as gluondataset 

import scipy.io as sio

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def _preprocess(dataset, labels,del_background=True, normalization = True, pca = False):
    '''
    @Description:
        normalize the dataset
        reshape the dataset [ from (145,145,200) to (145*145,200) ]
        delete the background (where the (labels == 0) )
    
    @input:
        dataset : shape==(145,145,200)
        labels : shape==(145,145)
        normalization : to normalize or not
        pca : to perform pca to dataset or not 
    
    @output:
        dataset : shape==(10249/21025, 200)
        labels : shape==((10249/21025, 1)) 
    '''
    [m,n,b] = np.shape(dataset)
    dataset = np.asarray(dataset,dtype = 'float32').reshape([m*n,b,]) # (21025, 200)
    labels = np.asarray(labels).reshape([m*n,1,])   # (21025, 1)
    
    if pca:
        pca = PCA(n_components =50)
        dataset = pca.fit_transform(dataset)

    if normalization ==True:
        min_max_scaler = preprocessing.MinMaxScaler()
        dataset = min_max_scaler.fit_transform(dataset).astype('float32')
    
    if del_background == True:
        #删除背景部分：label为0的点默认为背景点，可忽略 
        index = np.argwhere(labels[:,-1] == 0).flatten()
        dataset = np.delete(dataset, index, axis = 0)
        labels = np.delete(labels, index, axis = 0)
    #将label放到光谱维最后一位，保证label与data同步
    #data_com = np.concatenate((dataset,labels),axis =1 )     
    return dataset,labels

    

class IndianDataset(gluondataset._DownloadedDataset):
    def __init__(dasein,data_dir_=os.path.join(lib_path),train=True,one_hot=False,transform=None):
        dasein.one_hot = one_hot
        dasein._TRAIN_DATA = osp.join(data_dir_, 'Indian_pines_corrected.mat')
        dasein._TEST_DATA =  osp.join(data_dir_, 'Indian_pines_gt.mat')
        
        dasein._one_hot = one_hot
        dasein.data_dir_ = data_dir_
        dasein._train = train
        dasein.transform = transform
        dasein.dtype = np.uint16
            
        dasein._epochs_completed = 0
        dasein._index_in_epoch = 0 
        super(IndianDataset,dasein).__init__(data_dir_,transform) 

    def _get_data(dasein):
        """
        这个函数是因为该类继承于_DownloadedDataset[link](https://mxnet.incubator.apache.org/_modules/mxnet/gluon/data/dataset.html#Dataset)
        执行super(MNIST_csv,dasein).__init__(data_dir_,transform) 的时候
        会将data和label初始化在
        self._data = None 和 self._label = None 中
        """
        raw_data = sio.loadmat(dasein._TRAIN_DATA)['indian_pines_corrected']
        raw_labels = sio.loadmat(dasein._TEST_DATA)['indian_pines_gt']
        mydata = raw_data.reshape([145,145,200])
        labels = raw_labels.reshape([145,145])        
        mydata,labels = _preprocess(dataset=mydata,labels=labels)
        
        train_images, test_images,train_labels, test_labels = train_test_split(mydata, labels, train_size=0.8, random_state=0)
        
        if dasein._train:
            data,labels = train_images, train_labels
        else:
            data,labels = test_images, test_labels
        
        # get data
        dasein._num_examples = data.shape[0]
        dasein._data = data
        
        # get label
        if True == dasein._one_hot:
            """
            def dense_to_onehot():
            https://www.cnblogs.com/wxshi/p/8645600.html
            https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html    
            """    
            dasein.enc = OneHotEncoder(categories='auto')
            dasein.enc.fit(labels)
            dasein._label = mx.nd.array(dasein.enc.transform(labels).toarray())
        else:
            dasein._label = mx.nd.array(labels)   
          



