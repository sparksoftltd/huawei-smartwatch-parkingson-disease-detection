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

import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler





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





result_dic_column = ['Activity', 'Mean precision', 'Std precision']
result_dic_record = []
repetition_num = 20

total_activity = 9
activity_idx_list =[1,2,3,4,5,6,7,8,9]


dataset_idx = 2  # 用第几个数据集
dataset_name = 'Dataset' + str(dataset_idx)  # 具体的文件夹

final_path =''

for activity_idx in activity_idx_list:
    activity_name = 'activity' + str(activity_idx)
    print(activity_name + ' is running')
    file_name = 'Dataset2/' + activity_name + '.csv'


    folder_num = 10
    test_ratio = 0.1
    is_onehot = 0

    # 生成结果的文件夹
    first_layer_dictionary = dataset_name + '_result'  # Dataset1_result or Dataset2_result
    second_layer_dictionary = 'Repetition_' + str(repetition_num)
    third_layer_dictionary = 'LightGBM'
    final_path = first_layer_dictionary + '/' + second_layer_dictionary + '/' + third_layer_dictionary + '/'
    if not os.path.exists(final_path):
        os.makedirs(final_path)


    result_list = []
    for i in range(repetition_num):
        print('\t' + activity_name + ' is running on ' + str(i + 1) + 'th repetition')
        X_train_list, y_train_list, \
        X_val_list, y_val_list, \
        X_train_total, y_train_total, \
        X_test, y_test, dataset_drop = My_CrossValidation.my_cross_validation(file_name, folder_num,
                                                                              test_ratio, is_onehot)  # prepare the dataset

        order_validation = +2  # 第几组validation
        X_train = X_train_list[order_validation - 1]
        y_train = y_train_list[order_validation - 1]
        X_val = X_val_list[order_validation - 1]
        y_val = y_val_list[order_validation - 1]

        feature_num = np.array(X_train_total).shape[1]
        important_features = np.zeros([feature_num, ])  #

        # 训练
        begin_time = datetime.now()  # 存储当前时间作为启示时间
        train_data = lgb.Dataset(X_train, label=y_train)
        validation_data = lgb.Dataset(X_val, label=y_val)
        params = {
            'learning_rate': 0.05,
            'lambda_l1': 0.3,
            'lambda_l2': 0.3,
            'max_depth': 2,
            'objective': 'multiclass',
            'num_class': 8,
        }

        clf = lgb.train(params, train_data, valid_sets=[validation_data])
        importance=clf.feature_importance()
        temp_coefficient = importance
        print(importance)
        print(temp_coefficient)
        num_k = 10
        threshold=np.sort(temp_coefficient)[-num_k]
        result=[]
        for item in temp_coefficient:
            if item<threshold:
                result.append(0)
            else:
                result.append(1)

        important_features=np.sum([important_features,result],axis=0)

        end_time = datetime.now()  # 存储当前时间作为终止时间
        print('总耗时:\t', (end_time - begin_time).total_seconds())

        # 1、AUC
        y_pred_pa = clf.predict(X_test)  # !!!注意lgm预测的是分数，类似 sklearn的predict_proba
        y_pred = y_pred_pa.argmax(axis=1)
        temp_result = precision_score(y_test, y_pred, average='micro')
        print('\t', temp_result)
        result_list.append(temp_result)

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

save_path = final_path + 'LightGBM.csv'  # 存储路径
final_result.to_csv(save_path, index=False)  # 保存文件






#
#
# # 训练
# begin_time = datetime.now()  # 存储当前时间作为启示时间
# train_data = lgb.Dataset(X_train_total, label=y_train_total)
# validation_data = lgb.Dataset(X_test, label=y_test)
# params = {
#     'learning_rate': 0.1,
#     'lambda_l1': 0.1,
#     'lambda_l2': 0.1,
#     'max_depth': 10,
#     'objective': 'multiclass',
#     'num_class': 5,
# }
# clf = lgb.train(params, train_data, valid_sets=[validation_data])
# end_time = datetime.now()  # 存储当前时间作为终止时间
# print('总耗时:\t', (end_time - begin_time).total_seconds())
#
# # 1、AUC
# y_pred_pa = clf.predict(X_test)  # !!!注意lgm预测的是分数，类似 sklearn的predict_proba
# # y_test_oh = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
# # AUC = roc_auc_score(y_test_oh, y_pred_pa, average='micro')
# # print('AUC:\t', AUC)
#
# #  2、混淆矩阵
# y_pred = y_pred_pa.argmax(axis=1)
# print('Precision：   \t', precision_score(y_test, y_pred, average='micro'))
#
#
#
#
