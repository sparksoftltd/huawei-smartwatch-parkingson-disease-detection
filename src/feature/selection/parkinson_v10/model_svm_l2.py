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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler

# from sklearn.externals import joblib
# from scipy import interp
import My_CrossValidation
from numpy import *  # 平均值和方差


# 选择最佳的正则化参数
def choose_best_parameter(X_train_list, y_train_list, X_val_list, y_val_list, parameter_list):
    # 每一个参数都要跑folder number次实验，选出最佳参数
    folder_num = len(X_train_list)
    best_parameter = 0
    best_precision = 0
    for parameter in parameter_list:
        validation_precision = 0
        for i in range(folder_num):
            # 定义全局模型
            my_model = OneVsRestClassifier(
                svm.LinearSVC(penalty='l2', dual=True, C=parameter, random_state=None, verbose=0, max_iter=5000))
            my_model.fit(X_train_list[i], y_train_list[i])

            y_pred = my_model.predict(X_val_list[i])
            validation_precision += precision_score(y_val_list[i], y_pred, average='micro')

        validation_precision /= folder_num
        if best_precision < validation_precision:
            best_precision = validation_precision
            best_parameter = parameter

    return best_parameter


result_dic_column = ['Activity', 'Mean precision', 'Std precision']
result_dic_record = []
repetition_num = 20
total_activity = 12
activity_idx_list =[9]
# activity_idx_list = range(9, 9 + 1)

dataset_idx = 2  # 用第几个数据集
dataset_name = 'Dataset' + str(dataset_idx)  # 具体的文件夹

final_path = ''

for activity_idx in activity_idx_list:
    activity_name = 'activity' + str(activity_idx)  # 在运行第几个活动的数据集
    print(activity_name + ' is running')
    file_name = dataset_name + '/' + activity_name + '.csv'  # 输入数据具体的文件

    folder_num = 5
    test_ratio = 0.1
    is_onehot = 1  # 基于SVM的多分类需要

    # 生成结果的文件夹
    first_layer_dictionary = dataset_name + '_result'  # Dataset1_result or Dataset2_result
    second_layer_dictionary = 'Repetition_' + str(repetition_num)
    third_layer_dictionary = 'SVM_L2'
    final_path = first_layer_dictionary + '/' + second_layer_dictionary + '/' + third_layer_dictionary + '/'
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    result_list = []
    feature_num = 0
    important_features = []
    i = 0
    while i < repetition_num:
        print('\t' + activity_name + ' is running on ' + str(i + 1) + 'th repetition')
        X_train_list, y_train_list, \
        X_val_list, y_val_list, \
        X_train_total, y_train_total, \
        X_test, y_test, dataset_drop = My_CrossValidation.my_cross_validation(file_name, folder_num,
                                                                              test_ratio,
                                                                              is_onehot)  # prepare the dataset

        # 初始化import_features的维度
        feature_num = np.array(X_train_total).shape[1]
        important_features = np.zeros([feature_num, ])  #

        begin_time = datetime.now()  # 存储当前时间作为启示时间
        # 选择最佳参数
        parameter_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        best_parameter = choose_best_parameter(X_train_list, y_train_list, X_val_list, y_val_list, parameter_list)
        print('\tThe best paramter is', best_parameter)  # 输出最佳参数
        my_model = OneVsRestClassifier(
            svm.LinearSVC(penalty='l2', dual=True, C=best_parameter, random_state=None, verbose=0, max_iter=5000))
        my_model.fit(X_train_total, y_train_total)

        temp_coefficient = abs(my_model.coef_)
        print(temp_coefficient)
        end_time = datetime.now()  # 存储当前时间作为终止时间
        print('\t\t总耗时:\t', (end_time - begin_time).total_seconds())

        y_pred = my_model.predict(X_test)
        temp_result = precision_score(y_test, y_pred, average='micro')

        # if temp_result < 0.5:
        #     continue

        i += 1

        print('\t', temp_result)
        result_list.append(temp_result)

        # 每一个分类器都只要前num_k个特征
        num_k = 10
        for item in temp_coefficient:
            temp_item = - np.sort(-item)  # 非原地操作，创建了副本，逆序排列
            threshold = temp_item[num_k]
            item[item < threshold] = 0  # 权重不够大的，置为0

        temp_coefficient[temp_coefficient != 0] = 1  # 把不等于0的全部置换为1，表示这个特征出现的频数
        coefficient_sum = temp_coefficient.sum(axis=0)  # 按第一维度(行)求和
        important_features += coefficient_sum

    # 记录每一个activity运行repetation_num次的平均值和方差
    result_record = [activity_idx, mean(result_list), std(result_list)]
    result_dic_record.append(result_record)
    print('\n')

    save_path_scatter = final_path + activity_name + '_result.csv'
    pd_result_list = pd.DataFrame(data=result_list).round(3)
    pd_result_list.to_csv(save_path_scatter, index=False)

    important_features /= repetition_num
    save_path_features = final_path + activity_name + '_features.csv'
    important_features = pd.DataFrame(index=range(1, feature_num + 1), columns=['Frequency'], data=important_features)
    important_features.sort_values('Frequency', ascending=False, inplace=True)
    important_features.to_csv(save_path_features, index=True)

final_result = pd.DataFrame(columns=result_dic_column, data=result_dic_record)
final_result = final_result.round(3)  # 保留三位有效数字

save_path = final_path + 'SVM_L2.csv'  # 存储路径
final_result.to_csv(save_path, index=False)  # 保存文件
