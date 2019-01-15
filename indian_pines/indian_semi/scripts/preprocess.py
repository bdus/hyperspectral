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
import random
import scipy.io as sio
from sklearn import preprocessing
from sklearn import datasets, svm, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
# =========== get data ===========
pwd = os.path.dirname(__file__)
DATA_PATH = os.path.join(pwd,'..','data')

print('DATA_PATH',DATA_PATH)

raw_dataset = sio.loadmat(os.path.join(DATA_PATH,'Indian_pines_corrected.mat'))['indian_pines_corrected']
raw_labels = sio.loadmat(os.path.join(DATA_PATH,'Indian_pines_gt.mat'))['indian_pines_gt']
num_classes = 16
percentage = 0.1
num_b = 200

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
def scalar(data):
    '''
    0-1归一化
    '''
    maxnum = np.max(data)
    minnum = np.min(data)
    result = np.float32((data - minnum) / (maxnum - minnum))
    return result


def z_score(data):
    '''
    标准化
    '''
    mean = np.mean(data)
    stdnum = np.std(data)
    result = np.float32((data - mean) / stdnum)
    return result

def scalar_row(data):
    '''
    按行标准化
    '''
    sum_row = np.sqrt(np.sum(data**2,1)).reshape([-1,1])
    data = data / sum_row
    return data
    

def del_background(dataset, labels, normalization = 4, pca = False):
    '''
    对数据进行归一化处理;
    normalization = 1 : 0-1归一化
    normalization = 2 : 标准化
    normalization = 4 : 按行归一化
    
    
    #attation 数据归一化要做在划分训练样本之前；
    '''
    [m,n,b] = np.shape(dataset)
    dataset = np.asarray(dataset,dtype = 'float32').reshape([m*n,b,])
    labels = np.asarray(labels).reshape([m*n,1,])
    
    if pca:
        pca = PCA(n_components =50)
        dataset = pca.fit_transform(dataset)

    if normalization ==1:
        min_max_scaler = preprocessing.MinMaxScaler()  
        dataset = min_max_scaler.fit_transform(dataset).astype('float32')
    elif normalization ==2:
        stand_scaler = preprocessing.StandardScaler()
        dataset = stand_scaler.fit_transform(dataset).astype('float32')
    elif normalization ==3:
        stand_scaler = preprocessing.StandardScaler()
        dataset = stand_scaler.fit_transform(dataset).astype('float32')
        min_max_scaler = preprocessing.MinMaxScaler()  
        dataset = min_max_scaler.fit_transform(dataset).astype('float32')
    elif normalization ==4:
        dataset = scalar_row(dataset)

    else:
        pass

    #删除背景部分：label为0的点默认为背景点，可忽略   
    index = np.argwhere(labels[:,-1] == 0).flatten()
    dataset = np.delete(dataset, index, axis = 0)
    labels = np.delete(labels, index, axis = 0)
    #将label放到光谱维最后一位，保证label与data同步
    data_com = np.concatenate((dataset,labels),axis =1 )
     
    return(data_com)


def devided_train(data_com, num_classes  = 16, percentage = 0.1):
    '''
    data_com:二维矩阵，每行对应一个像素点，label为最后一位
    num_class: 地物类别数
    percentage: 训练样本百分比
    '''
    #划分训练样本与测试样本：
    b = data_com.shape[1]
    #创建两个空数组，用于后续拼接每一类样本
    train_com = np.empty([1, b])
    test_com = np.empty([1, b])
    
    for i in range(1, num_classes + 1):
        index_class = np.argwhere(data_com[:,-1] == i).flatten()
        data_class = data_com[index_class]
        num_class = len(data_class)
        #随机取一定数量的训练样本
        if percentage <= 1:
            num_train = np.ceil(num_class * percentage).astype('uint8')
        else:
            num_train = percentage
        index_train = random.sample(range(num_class), num_train)
        train_class = data_class[index_train]
        test_class = np.delete(data_class,index_train, axis = 0)
        #将各类训练样本拼接成完整的训练集与测试集
        train_com = np.concatenate((train_com, train_class), axis = 0)
        test_com = np.concatenate((test_com,test_class), axis = 0)
    #删除最初的空数组    
    train_com = np.delete(train_com, 0, axis = 0)
    test_com = np.delete(test_com, 0, axis = 0)
    return(train_com, test_com)

def preprocess(data_com, shuffle = True ):
#数据预处理
    #1. 打乱数据（训练集）
    if shuffle:
        num_train = data_com.shape[0]
        seed = [i for i in range(num_train)]
        random.shuffle(seed)
        data_com = data_com[seed]
    #2. 将数据与label分开
    label = data_com[:,-1].astype('uint8')
    data =np.delete(data_com, -1, axis = 1).astype('float32')

    return(data, label) 
    
# ============= get data ============
    
dataset = raw_dataset.reshape([145,145,200])
labels = raw_labels.reshape([145,145])
data_com = del_background(dataset, labels)
[train_com, test_com] = devided_train(data_com,num_classes = num_classes, percentage = percentage)
[train_data, train_label] = preprocess(train_com, shuffle = False)
[test_data, test_label] = preprocess(test_com, shuffle = False)

def one_hot_encode(num,cla):
    tmp = np.zeros(cla)
    tmp[num-1]=1
    return tmp

