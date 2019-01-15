# -*- coding: utf-8 -*-
"""
使用svm对indian_pines进行分类
"""
import os
import sys
import numpy as np
import random
import scipy.io as sio
from sklearn import preprocessing
from sklearn import datasets, svm, metrics
from sklearn.decomposition import PCA

#DATA_PATH = 'D:/python_codes/rpca_svm/data/indian_new/'
#SAVE_PATH = 'D:/python_codes/rpca_svm/result/indian_new/'
# DATA_PATH = 'D:/python_codes/rpca_svm/data/indian/'
# SAVE_PATH = 'D:/python_codes/rpca_svm/result/indian_old/'
#DATA_PATH = 'D:/python_codes/latlrr_svm/data/indian/'
#SAVE_PATH = 'D:/python_codes/latlrr_svm/result/indian/'

pwd = os.path.dirname(__file__)
DATA_PATH = os.path.join(pwd,'')
SAVE_PATH = os.path.join(pwd,'./old')


print('DATA_PATH',DATA_PATH)

dataset = sio.loadmat(DATA_PATH+'Indian_pines_corrected.mat')['indian_pines_corrected']
#dataset = sio.loadmat(DATA_PATH+'indian_latlrr_spa_52.mat')['clear_image']
labels = sio.loadmat(DATA_PATH+'Indian_pines_gt.mat')['indian_pines_gt']
#dataset = sio.loadmat(DATA_PATH +'indian_pines.mat')['fea']
#dataset = sio.loadmat(DATA_PATH +'InSSLR2_10.mat')['L']
#dataset = sio.loadmat(DATA_PATH +'Io_SSLR1_IALM2.mat')['L']
#labels = sio.loadmat(DATA_PATH + 'indian_pines.mat')['labels']

dataset = dataset.reshape([145,145,200])
labels = labels.reshape([145,145])

num_classes = 16
percentage = 0.1
num_b = 200

filename = 'formal.txt'
#filename2 = 'formal_avg.txt'

c = [0.1,1,10,100,1000]
gamma = [0.1,1,10,100,1000]
#c = [200,300,400]
#gamma = [0.002,0.003,0.004]
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

#----------------------------------------------------------------------------------
def kappa(confusion_matrix, k):
    #根据混淆矩阵计算kappa系数
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe  = float(ysum*xsum)/np.sum(dataMat)**2
    OA = float(P0/np.sum(dataMat)*1.0)
    cohens_coefficient = float((OA-Pe)/(1-Pe))
    return cohens_coefficient 

def AA(confusion_matrix):
    
    num_pre_class = np.sum(confusion_matrix, axis = 1);
    num_class = len(num_pre_class);
    result = np.zeros(num_class);
    for i in range(num_class):
        result[i] = confusion_matrix[i,i] / num_pre_class[i]
        average = np.mean(result)
    return average
    
#使用svm对数据集进行分类-----------------------------------------------------
data_com = del_background(dataset, labels)
[train_com, test_com] = devided_train(data_com,num_classes = num_classes, percentage = percentage)
[train_data, train_label] = preprocess(train_com, shuffle = False)
[test_data, test_label] = preprocess(test_com, shuffle = False)

#训练SVM分类器

check_point = 0
check_c = 0
check_gamma = 0

for i in c:
    for j in gamma:       
        clf = svm.SVC(C=i, kernel='rbf', gamma=j, decision_function_shape='ovr')
        clf.fit(train_data, train_label)
        train_score = clf.score(train_data, train_label)
        test_score = clf.score(test_data, test_label)
        print( 'gamma: %f--c: %f--train_acc: %f--test_acc: %f' % (j,i,train_score, test_score))
        if test_score > check_point:
            check_point = test_score
            check_c = i
            check_gamma = j
print('the best c is : %d-- the best gamma is : %f' % (check_c, check_gamma))
        
        
        

best = svm.SVC(kernel='rbf',C=check_c, gamma = check_gamma, decision_function_shape='ovr')
best.fit(train_data, train_label)
train_score = best.score(train_data, train_label)
test_score = best.score(test_data, test_label)

pred_label = best.predict(test_data)
true_label = test_com[:,-1].flatten().astype('uint8')

classify_report = metrics.classification_report(true_label, pred_label)
confusion_matrix = metrics.confusion_matrix(true_label, pred_label)
overall_accuracy = metrics.accuracy_score(true_label, pred_label)
acc_for_each_class = metrics.precision_score(true_label, pred_label, average=None)
average_accuracy_row = AA(confusion_matrix)
average_accuracy_col = np.mean(acc_for_each_class)
kappa_coefficient = kappa(confusion_matrix, num_classes)

print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy_row: {0:f}'.format(average_accuracy_row))
print('average_accuracy_col: {0:f}'.format(average_accuracy_col))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('kappa coefficient: {0:f}'.format(kappa_coefficient))

save_file = open(SAVE_PATH + filename,'a')

save_file.write('\n')
save_file.write('result with %.3f training samples each class: \n' % percentage)
save_file.write('\n')
save_file.write('param_c: %f--param_gamma: %f\n' % (check_c, check_gamma))
save_file.write('classify_report : \n'+ classify_report +'\n')
save_file.write('confusion_matrix : \n' +  str(confusion_matrix) + '\n')
save_file.write('acc_for_each_class : \n' +str( acc_for_each_class) +'\n')
save_file.write('average_accuracy_row: %.6f\n' % average_accuracy_row)
save_file.write('average_accuracy_col: %.6f\n' % average_accuracy_col)
save_file.write('overall_accuracy: %.6f\n' % overall_accuracy)
save_file.write('kappa coefficient: %.6f\n' % kappa_coefficient)
save_file.write('[train acc : %f--test acc : %f]\n' %(train_score, test_score))
save_file.close()
'''
save_file2 = open(SAVE_PATH + filename2,'a')
save_file2.write('%f %f %f %f\n' %(overall_accuracy,average_accuracy_row,average_accuracy_col,kappa_coefficient))
save_file2.close()
'''
