#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:25:38 2018

@author: bdus

try to preprocess data

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt  
import scipy.io as sio
from sklearn import preprocessing
from sklearn import datasets, svm, metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# =========== get data ===========
pwd = os.path.dirname(__file__)
DATA_PATH = os.path.join(pwd,'..','data')

print('DATA_PATH',DATA_PATH)

raw_dataset = sio.loadmat(os.path.join(DATA_PATH,'Indian_pines_corrected.mat'))['indian_pines_corrected']
raw_labels = sio.loadmat(os.path.join(DATA_PATH,'Indian_pines_gt.mat'))['indian_pines_gt']
dataset = raw_dataset.reshape([145,145,200])
labels = raw_labels.reshape([145,145])


# =========== show data ===========
def draw_bar(labels):
    """
    @Description:
        Draw labels
    
    @input :
        labels.shape == (145,145)
    @output :
        None
    """
    labels = np.array(labels)
    x = np.arange(np.min(labels),np.max(labels)+1,1)
    y = np.array([np.sum(labels==i) for i in x])
    print('x: ',x)
    plt.stem(x,y,'-')
    #plt.bar(x,y,0.4,color="green")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    if labels.shape == (145,145):
        fig = plt.figure()
        plt.imshow(labels)
# =========== preprocessing ===========
def preprocess(dataset, labels,del_background=True, normalization = True, pca = False):
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

dataset,labels =preprocess(dataset=dataset,labels=labels)
train_images, test_images,train_labels, test_labels = train_test_split(dataset, labels, train_size=0.8, random_state=0)
draw_bar(labels)
enc = OneHotEncoder(categories='auto')
enc.fit(labels)
wlabel = enc.transform(labels).toarray()