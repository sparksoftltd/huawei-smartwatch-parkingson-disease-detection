import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE


def get_Datasets():

    data = pd.read_csv(r"../../datasets/ocsvm_abnormal_detection/H_PD_AB_wr_40_fea_1.csv", sep=',')
    data.dropna(axis=0, how='any')
    data.dropna(inplace=True)  # 删除包含缺失值的行
    # 总体的
    xxx = data.iloc[:, :-3]
    yyy = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(xxx, yyy, test_size=0.3, random_state=7)
    return data, x_train, x_test, y_train, y_test

def train(x_train):
    clf = svm.OneClassSVM(nu=0.15, kernel="rbf")
    clf.fit(x_train)
    return clf

def remove_abnormal_data(data, clf):
    y_tsne = data.iloc[:, -1]
    input_data = data.iloc[:, :-3]
    input_data.reset_index(drop=True, inplace=True)
    y_tsne.reset_index(drop=True, inplace=True)

    index = []
    y_pred = clf.predict(input_data)
    for i in range(0, input_data.shape[0]):
        if y_pred[i] == -1:
            index.append(i)
    input_data_new = input_data.drop(index=index)
    y_tsne_new = y_tsne.drop(index=index)
    input_data_new.reset_index(drop=True, inplace=True)
    y_tsne_new.reset_index(drop=True, inplace=True)

    y_tsne_tocsv = data.iloc[:, -3:]
    y_tsne_tocsv.reset_index(drop=True, inplace=True)
    print(y_tsne_tocsv.index)

    y_tsne_new_tocsv = y_tsne_tocsv.drop(index=index)
    y_tsne_new_tocsv.reset_index(drop=True, inplace=True)

    # 去异常后数据和标签拼接保存
    result = pd.concat([input_data_new, y_tsne_new_tocsv], axis=1)
    result.to_csv(r"../../result/ocsvm_abnormal_detection/H_PD_AB_wr_40_fea_1.csv")
    print("已完成去异常后数据和标签拼接保存")
    return input_data, y_tsne, input_data_new, y_tsne_new

def visual(input_data, y_tsne):
    # df = data.drop(["label", "label2", "subject_id", "Activity_id"],axis=1)
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(input_data)
    print("Org data dimension is {}.Embedded data dimension is {}".format(input_data.shape[-1], X_tsne.shape[-1]))
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y_tsne[i]), color=plt.cm.Set1(y_tsne[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()