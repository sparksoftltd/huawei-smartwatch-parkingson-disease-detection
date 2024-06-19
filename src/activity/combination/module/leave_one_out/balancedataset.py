import plistlib
import random
import pandas as pd
import heapq
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from collections import Counter
import random
from sklearn.metrics import classification_report


'''
       Describe
       选出2，3，4级病人
       -----------------
       Parameters
       -----------------
       data:原始特征文件
       subject_id1:病人编号所在列的数字

       Return
       -----------------
       choospeopl2：所选择的病人
       label45：病人对应标签
       removel：要移除的病人（所在等级人数小于一）
       lenobj：每一类病人数目
       dict1：每一类对应病人编号）字典0      '''


def choosepeople(data, subject_id1, levelchoose,label):
    dict1 = {}
    encoder = {}
    inverencode = {}
    recorddata = data.drop_duplicates(subset=subject_id1, keep='first', inplace=False)

    for i in recorddata[label]:
        dict1[int(i)] = []
    for code, i in zip(levelchoose, range(0, len(levelchoose))):
        encoder[int(code)] = i
        inverencode[i] = code
    for i, j in zip(recorddata[subject_id1], recorddata[label]):
        dict1[int(j)].append(i)
    lenobj = []
    for zidian in levelchoose:
        lenobj.append(len(dict1[int(zidian)]))
    # 计算每个类人数
    choosepeople = []
    removel = []
    label23 = []
    for level in levelchoose:
        # print(len(dict1[format(level, ".1f")]))
        if len(dict1[int(level)]) <= 1:  #
            removel.append(level)
        else:
            choosepeople.append(dict1[int(level)])
            label23.append(level)
    choosepeople2 = []
    label45 = []
    for i, k in zip(choosepeople, label23):
        for j in i:
            choosepeople2.append(j)
            label45.append(k)

    return choosepeople2, label45, removel, lenobj, dict1, encoder, inverencode


'''
       Describe
       切分数据集
       -----------------
       Parameters
       -----------------
       data:原始特征文件
       subject_id1:病人编号所在列的数字
       trainperosonlist：训练集人
       testpersonlist：测试集人

       Return
       -----------------
       olddata：训练集数据
       testdata：测试集数据
       '''


def splitperson(data, subject_id1, trainpersonlist, testpersonlist):
    '''
    dict1 = {}
    recorddata = data.drop_duplicates(subset=subject_id1, keep='first', inplace=False)
    print(recorddata)
    for i in recorddata[label]:
        dict1[str(i)] = []
    for i, j in zip(recorddata[subject_id1], recorddata[label]):
        dict1[str(j)].append(i)

    #计算每个类人数
    removel=[]
    for level in range(0,5):
        print(len(dict1[format(level, ".1f")]))
        if len(dict1[format(level, ".1f")])<=1:
            removel.append(level)
    '''

    # trainpersonlist = [11,13,40,3,4,5,6,10,15,17,20,24,30,34,2,7,8,22,26,35,37,39,1,14,19,28,29,33,38,18]
    # testpersonlist = [16,23,27,32,36,31,9,21,25,12]
    old_data = pd.DataFrame()
    for subject_id in trainpersonlist:
        temp = data[data[subject_id1] == subject_id]
        old_data = pd.concat([old_data, temp])
    test_data = pd.DataFrame()
    for subject_id in testpersonlist:
        temp = data[data[subject_id1] == subject_id]
        test_data = pd.concat([test_data, temp])
    return old_data, test_data


'''
       Describe
       LGB模型训练
       -----------------
       Parameters
       -----------------
       data:原始特征文件
       subject_id1:病人编号所在列的数字
       trainperosonlist：训练集人
       testpersonlist：测试集人

       Return
       -----------------
       olddata：训练集数据
       testdata：测试集数据
       '''
def create_balance_dataset(dict1, dictlevel0, dictlevel1, dictlevel2, dictlevel3, dictlevel4, dictlevel5, turn,
                           chooselevel,lenobj):
    dictlevel0 = sorted(dictlevel0.items(), key=lambda x: -x[1])
    dictlevel1 = sorted(dictlevel1.items(), key=lambda x: -x[1])
    dictlevel2 = sorted(dictlevel2.items(), key=lambda x: -x[1])
    dictlevel3 = sorted(dictlevel3.items(), key=lambda x: -x[1])
    dictlevel4 = sorted(dictlevel4.items(), key=lambda x: -x[1])
    dictlevel5 = sorted(dictlevel5.items(), key=lambda x: -x[1])
    len123 = []
    recodata = []
    lebal2 = []
    xunhuan = turn + 1
    for i in chooselevel:
        len123.append(len(eval('dictlevel' + str(i))))
    for i in chooselevel:
        if len(eval('dictlevel' + str(i))) // min(len123) == 1:
            if xunhuan % 2 == 1:
                for changdu in range(0, min(lenobj), ):
                    recodata.append(int(eval('dictlevel' + str(i))[changdu][0]))
                    lebal2.append(i)
            if xunhuan % 2 == 0:
                for changdu in range(min(lenobj), len(eval('dictlevel' + str(i)))):
                    recodata.append(int(eval('dictlevel' + str(i))[changdu][0]))
                    lebal2.append(i)
                key = random.Random(100).sample(range(0, min(lenobj)),
                                                2 * min(lenobj) - len(eval('dictlevel' + str(i))))  # [0,26)
                for k in key:
                    recodata.append(int(eval('dictlevel' + str(i))[k][0]))
                    lebal2.append(i)
        else:
            for changdu in range(0 + turn % (len(eval('dictlevel' + str(i))) // min(len123)),
                                 len(eval('dictlevel' + str(i))) // min(len123) * min(lenobj),
                                 len(eval('dictlevel' + str(i))) // min(len123)):
                recodata.append(int(eval('dictlevel' + str(i))[changdu][0]))
                lebal2.append(i)

    print('正在第{}次采样训练'.format(turn+1))
    return recodata, lebal2