# 该module My_CrossValidation 用来切分华为的数据集
# 切出训练集，验证集，和测试集

import pandas as pd
import numpy as np
import time
import logging
import os, sys
import psutil
import lightgbm as lgb
from datetime import datetime

from itertools import cycle
from sklearn import svm
from sklearn.metrics import *
# from sklearn.cross_validation import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


# from sklearn.externals import joblib
# from scipy import interp


def find_data_index(id_lst, dataset, subject_id):
    # id_list: 病人的id
    # dataset：完整的数据集，用来找病人id对应的数据的索引index
    # subject_id： DataFrame的subject_id这一列
    data_idx = []
    for i in range(0, len(id_lst)):
        temp_idx = dataset[dataset[subject_id] == id_lst[i]].index.to_list()
        data_idx += temp_idx
    return data_idx


def my_cross_validation(file_name, folder_num, test_ratio, is_onehot):
    # file_name = 'Dataset/activity13.csv'
    dataset = pd.read_csv(file_name, header=0)  # 不保留行索引
    dataset = dataset.dropna()  # 注意到，去掉了有NULL的行，但行标没变，等下返回dataframe的坐标时会出问题，所以要reset index
    dataset = dataset.reset_index(drop=True)

    # dataset = dataset.loc[1:200]  #  小样本测试一下程序
    label = dataset.columns[-1]  # 病人处在病情的第几期
    subject_id = dataset.columns[-2]  # 病人id


    activity_id = dataset.columns[-3]  # 活动id：第几个活动
    feature_num = int(dataset.columns[-4]) + 1  # feature number，因为是从0开始计数的

    # 找出所有的病人id，然后按id划分
    subject_num = len(dataset[subject_id].value_counts())  # 就是那个subject_id这一列有多少不同的元素值，也就是总的病人个数
    sub_lst = dataset[subject_id].values  # 所有的id
    sub_lst = np.unique(sub_lst)  # 删掉相同的元素
    total_id = sub_lst.tolist()  # 所有病人对应的id,保存为list，方便处理

    # sub_label_lst = dataset[label].values # 所有的label
    # sub_label_lst = np.unique(sub_label_lst) # 删掉相同的元素
    # total_label = sub_label_lst.tolist() # 所有病人对应的id，保存为list，方便处理

    # 找到所有病人的id对应的病情阶段
    label_lst = []
    for i in sub_lst:
        temp_label = dataset[dataset[subject_id] == i][label]
        temp_label = temp_label.values
        temp_label = int(np.unique(temp_label))
        label_lst.append(temp_label)

    # 找到每个病人对应的病情阶段
    from collections import Counter
    print(Counter(label_lst))

    # 处理数据集，丢掉不需要的列
    dataset_drop = dataset.drop(columns=[subject_id, activity_id])  # 删掉病人的id和活动id，暂时不需要
    dataset_drop = dataset_drop.values  # 把pandas的dataframe转成matrix，方便操作

    # 分割出特征集和label集
    feature_set = dataset_drop[:, :-1]
    temp_minmax = MinMaxScaler()
    feature_set = temp_minmax.fit_transform(feature_set)  # 标准化，方便收敛
    temp_min = min(dataset_drop[:, -1])
    label_set = dataset_drop[:, -1] - temp_min  # 从0开始
    X = feature_set
    y = label_set

    # 是否需要改成one hot编码
    if is_onehot:
        label_temp_set = np.unique(label_set)  # 统计多少个类
        y = label_binarize(y, classes=label_temp_set)

    # 分出训练集病人的id和测试集病人的id,以及对应的label
    X_train_sub_id, X_test_sub_id, \
    y_train_sub_id, y_test_sub_id = \
        train_test_split(total_id, label_lst, test_size=test_ratio, random_state=None,
                         stratify=label_lst)  # 把total_id随机分成两个子集
    # y_train_sub_id = X_train_sub_id  # 训练集的病人id
    # y_test_sub_id = X_test_sub_id  # 测试集的病人id

    # folder_num = 5
    kf = StratifiedKFold(n_splits=folder_num)  # 5-cross validation

    # 根据病人id，在完整数据集中找到对应的index
    # print(type(kf.split(train_sub_id)))
    whole_tra_idx, whole_val_idx = [], []  # k个cross validation的数据集下标
    for tra_idx, \
        val_idx in kf.split(X_train_sub_id, y_train_sub_id):  # generator 生成5组不同的训练集和测试集的人的id对应的list
        # 注意到train_sub_id属于list，不能批量做下标操作，所以将list转成numpy的array形式
        sub_train_id = np.array(X_train_sub_id)[tra_idx.tolist()]
        whole_tra_idx.append(find_data_index(sub_train_id, dataset, subject_id))

        sub_val_id = np.array(X_train_sub_id)[val_idx.tolist()]
        whole_val_idx.append(find_data_index(sub_val_id, dataset, subject_id))
        # sub_val_id = train_sub_id[val_idx]  # 被分作验证集的病人id
        # whole_val_idx.append(find_data_index(sub_val_id))

    # 测试集，验证集的特征数据和label值
    X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
    for i in range(folder_num):
        X_temp1 = X[whole_tra_idx[i]]
        X_train_list.append(X_temp1)
        y_temp1 = y[whole_tra_idx[i]]
        y_train_list.append(y_temp1)

        X_temp2 = X[whole_val_idx[i]]
        X_val_list.append(X_temp2)
        y_temp2 = y[whole_val_idx[i]]
        y_val_list.append(y_temp2)

    # 总的测试集的特征数据和label值
    train_idx = find_data_index(X_train_sub_id, dataset, subject_id)
    X_train = X[train_idx]
    y_train = y[train_idx]
    # 测试集的特征数据和label值
    test_idx = find_data_index(X_test_sub_id, dataset, subject_id)
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train_list, y_train_list, \
           X_val_list, y_val_list, \
           X_train, y_train, \
           X_test, y_test, \
           dataset_drop


# test
if __name__ == '__main__':
    file_name_1 = 'Dataset2/activity0.csv'
    folder_num_1 = 3
    test_ratio_1 = 0.1
    is_onehot_1 = True

    X_train_list, y_train_list, \
    X_val_list, y_val_list, \
    X_train, y_train, \
    X_test, y_test, _ = my_cross_validation(file_name_1, folder_num_1, test_ratio_1, is_onehot_1)


    print('\033[0;35m done \033[m')
