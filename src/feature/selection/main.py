import pandas as pd
import numpy as np
#将单个模型准确率结果合并
repetition=20
model=['LightGBM']
# for mod in model:
#     data_csvs = []
#     for i in range(0,13,1):
#         data=pd.read_csv('Parkinson_v10/Dataset2_result/Repetition_{}/{}/activity{}_result.csv'.format(repetition,mod,i))
#         data_csvs.append(data)
#     data_total = pd.concat(data_csvs,axis=1)  # 合并成一个dataframe
#     data_total.columns=['all']+[str(i) for i in range(1,13,1)]
#     data_total.to_csv('{}_all_result.csv'.format(mod),index=False)

##统计的是不同模型下每个活动中出现频次最高的特征
for mod in model:
    data_csvs = []
    for i in range(0,13,1):
        data=pd.read_csv('parkinson_v10/Dataset2_result/Repetition_{}/{}/activity{}_features.csv'.format(repetition,mod,i))
        data=data.iloc[0:15,0]
        data_csvs.append(data)
    data_total = pd.concat(data_csvs)  # 合并成一个dataframe
    #data_total.columns=['all','all']+['1','1','2','2','3','3','4','4','5','5','6','6','7','7','8','8','9','9','10','10','11','11','12','12']
    data_total.to_csv('{}_all_features.csv'.format(mod),index=False)

for mod in model:
    data=pd.read_csv('{}_all_features.csv'.format(mod))
    counts=data.iloc[:,0].value_counts()
    print(counts)
    counts.to_csv('{}_features_counts.csv'.format(mod))

# #统计的是每个模型对应活动特征
# repetition=20
# model=['SVM_L2','LightGBM']
# for i in range(0, 13, 1):
#     data_csvs = []
#     for mod in model:
#         data=pd.read_csv('Parkinson_v10/Dataset2_result/Repetition_{}/{}/activity{}_features.csv'.format(repetition,mod,i))
#         data=data.iloc[0:10,0]
#         data_csvs.append(data)
#     data_total = pd.concat(data_csvs)  # 合并成一个dataframe
#     #data_total.columns=['all','all']+['1','1','2','2','3','3','4','4','5','5','6','6','7','7','8','8','9','9','10','10','11','11','12','12']
#     data_total.to_csv('activity{}_3mod_features.csv'.format(i),index=False)
#
# for i in range(0, 13, 1):
#     data=pd.read_csv('activity{}_3mod_features.csv'.format(i))
#     counts=data.iloc[:,0].value_counts()
#     print(counts)
#     counts.to_csv('activity{}_3mod_counts_features.csv'.format(i))
#
