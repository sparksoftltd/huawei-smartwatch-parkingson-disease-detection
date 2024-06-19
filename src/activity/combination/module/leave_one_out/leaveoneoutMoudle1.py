import sys
import pandas as pd
import os
from leaveoneoutMoudle import *
from balancedataset import *
from collections import Counter
def leaveoneout(name1,yingshe1,chooselevel1,choose_featuretop1,dongzuo1):
    name=name1
    yingshe=yingshe1
    chooselevel=chooselevel1
    choose_featuretop=choose_featuretop1
    dongzuo=dongzuo1
    data = pd.read_csv(name, header=0).dropna()  # 读取总做数据文件
    if data.columns[-3] == 'Activity_id':
        featurenum = int(data.columns[-4])
        label = data.columns[-1]
        subject_id = data.columns[-2]
        activity_id = data.columns[-3]
        dongzuonum = 15
    else:
        featurenum = int(data.columns[-3])
        label = data.columns[-2]
        subject_id = data.columns[-1]
        dongzuonum = 2
    # #活动

    features = [str(i) for i in range(0, featurenum + 1)]  # feature=[0,1,2,3,4,5,6.....260]
    weightdict = {}
    for feanum in range(0, featurenum + 1):
        weightdict[feanum] = 0
    yingshe = {0: 0,
               1: 1,
               2: 1,
               3: 2,
               4: 3,
               5: 3}  # 等级映射
    chooselevel = ['1', '2', '3']  # 选择等级
    choose_featuretop = 10

    featureimportant = []
    if os.path.exists(r'../../result/leave_one_out/matrix/{}'.format(dongzuo)) == False:
        os.mkdir(r'../../result/leave_one_out/matrix/{}'.format(dongzuo), )
    trueall = []
    preall = []
    people_result_all = pd.DataFrame()
    data = pd.read_csv(name, header=0).dropna()  # 读取总做数据文件
    data['severity_level'] = data['severity_level'].map(yingshe)  # 等级映射，
    if data.columns[-3] == 'Activity_id':
        data = data.loc[data[activity_id] == dongzuo]  # 单个活动分类
        print("正在进行第{}个动作采样".format(dongzuo))
    else:
        print("正在平衡数据集采样")
    datahealth = data.loc[data['severity_level'] == 0]
    datahealth['subject_id'] = datahealth['subject_id'] + 15
    datapd = data.loc[data['severity_level'].isin([1, 2, 3, 4, 5])]
    data = pd.concat([datapd, datahealth], axis=0)
    # 选人
    choose_people, choose_label, reeo, lenobj, dict1, encoder, inverencode = choosepeople(data, subject_id,
                                                                                          levelchoose=chooselevel,
                                                                                          label=label)
    # 将训练标签-2，lgb从0开始
    data[label] = data[label].map(encoder)
    # 输出标签+1
    for long in range(0, len(choose_label)):
        choose_label[long] = choose_label[long]
    if len(reeo) != 0:
        if reeo[0] == 0:
            data[label] = data[label]
    # 记录人准确率，和等级准确率
    dictlevel0, dictlevel1, dictlevel2, dictlevel3, dictlevel4, dictlevel5 = leave_subject_to_one(data, choose_people,
                                                                                                  choose_label,
                                                                                                  chooselevel,
                                                                                                  choose_featuretop,
                                                                                                  weightdict,
                                                                                                  subject_id, features,
                                                                                                  label)
    len45 = []
    for choosele in chooselevel:
        len45.append(len(eval('dictlevel' + str(choosele))))
    print('********************************')
    # 计算伦次
    trun = max(len45) // min(len45)
    ACCALL = []
    F1ALL = []
    PRECISIONALL = []
    RECALLALL = []
    print("采样成功")
    print(trun)
    for i in range(0, trun):
        peoplechoosed, labelchoosed = create_balance_dataset(dict1, dictlevel0, dictlevel1, dictlevel2, dictlevel3,
                                                             dictlevel4, dictlevel5, i, chooselevel, lenobj)
        # 留一交叉验证，结果保存
        peopleacc, peoplelabel, precision_pre_people, F1SCORE_pre_people, recall_pre_people, levelacc, level_F1, level_recall, level_precison, people_result_all, featureimp, trueall, preall = leave_subject_out(
            data, peoplechoosed, labelchoosed, i, inverencode, encoder, chooselevel, choose_featuretop, weightdict,
            subject_id, features, label, dongzuo, trueall, preall)
        featureimportant.append(featureimp)
        # 保存结果
        save_result(peopleacc, peoplelabel, precision_pre_people, F1SCORE_pre_people, recall_pre_people, dongzuo,
                    levelacc, level_F1, level_recall, level_precison, people_result_all, i, chooselevel, trueall,
                    preall)
    averageacc(dongzuo, trun)
    saveresult2(dongzuo, trun)
    avgerecall_precion(dongzuo, trun)
    countfeature = []
    for first in featureimportant:
        for second in first:
            for third in second:
                for fourth in third:
                    countfeature.append(fourth)
    re2 = dict(Counter(countfeature))
    dictlevelacc = sorted(re2.items(), key=lambda x: -x[1])
    allfea = []
    for changdu in range(0, choose_featuretop + 1):
        allfea.append(int(dictlevelacc[changdu][0]))
    WEIGHT = [weightdict[wei] for wei in allfea]
    impfea = pd.DataFrame(allfea)
    impfea.columns = ['feature']
    impfea['weight'] = WEIGHT
    impfea.to_csv(r'../../result/leave_one_out/feature_important{}.csv'.format(dongzuo))