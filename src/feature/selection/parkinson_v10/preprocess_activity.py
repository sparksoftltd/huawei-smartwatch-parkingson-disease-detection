import io
import os
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

file_name = 'Dataset1/feturedrop6.csv'
dataset = pd.read_csv(file_name, header=0).dropna()

label = dataset.columns[-1]  # 病人处在病情的第几期
subject_id = dataset.columns[-2]  # 病人id
activity_id = dataset.columns[-3]  # 活动id：第几个活动

#  删掉病理5
idx_list = []
for i in [1, 2, 3, 4]:
    temp_list = dataset[dataset[label] == i].index.tolist()  # 找到各个活动的下标
    idx_list.extend(temp_list)
dataset = dataset.loc[idx_list, :]

if not os.path.exists('Dataset2'):
    os.makedirs('Dataset2')
dataset.to_csv('Dataset2/activity0.csv', index=None)

# 按活动分出数据集，总共activity_num个活动
activity_num = len(dataset[activity_id].value_counts())  # 计算activity_id这一列的不同元素值，也就是总的活动个数
for i in range(1, activity_num + 1):
    temp_list = dataset[dataset[activity_id] == i].index.tolist()  # 找到各个活动的下标
    temp_dataset = dataset.loc[temp_list, :]
    file_path = 'Dataset2/activity' + str(i) + '.csv'
    temp_dataset.to_csv(file_path, index=None)  # 不保留行索引
