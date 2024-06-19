import io
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





# d1 = pd.read_csv('Dataset/SVM_l2.csv', header=0)
# main_key = d1.columns[0]
# d2 = pd.read_csv('Dataset/Logistic_Regression_l2.csv', header=0)
# d3 = pd.read_csv('Dataset/LightGBM.csv', header=0)
# d4 = pd.read_csv('Dataset/DL_2layers.csv', header=0)
# d5 = pd.read_csv('Dataset/DL_3layers.csv', header=0)
# d6 = pd.read_csv('Dataset/DL_4layers.csv', header=0)
# d7 = pd.read_csv('Dataset/DL_5layers.csv', header=0)
#
# df = pd.merge(d1, d2, how='left', on=main_key)
# df = pd.merge(df, d3, how='left', on=main_key)
# df = pd.merge(df, d4, how='left', on=main_key)
# df = pd.merge(df, d5, how='left', on=main_key)
# df = pd.merge(df, d6, how='left', on=main_key)
# df = pd.merge(df, d7, how='left', on=main_key)
#

df = pd.read_csv('Dataset/Logistic_Regression_l2.csv', header=0)
df = df.round(3)
df.to_csv('Dataset/Logistic_Regression_l2.csv', index=False)