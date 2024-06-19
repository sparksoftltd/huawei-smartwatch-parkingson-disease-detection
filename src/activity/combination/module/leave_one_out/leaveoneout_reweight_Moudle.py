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
from sklearn.utils.class_weight import compute_sample_weight
import sys
sys.path.append(r'../../module/leave_one_out')
from balancedataset import*
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings('ignore')



class LGBModel:
    def __init__(self):
        # 模型超参数
        self.params = {
            'learning_rate': 0.05,
            'min_child_samples': 2,  # 子节点最小样本个数
            'max_depth': 1,  # 树的最大深度
            'lambda_l1': 0.3,  # 控制过拟合的超参数
            'boosting': 'gbdt',
            'objective': 'multiclass',
            'n_estimators': 300,  # 决策树的最大数量
            'metric': 'multi_error',
            'num_class': 8,
            'feature_fraction': .75,  # 每次选取百分之75的特征进行训练，控制过拟合
            'bagging_fraction': .75,  # 每次选取百分之85的数据进行训练，控制过拟合
            'seed': 0,
            'num_threads': 20,
            'verbose': -1,
            'early_stopping_rounds': 50,  # 当验证集在训练一百次过后准确率还没提升的时候停止
            'num_leaves': 128, }

    # LGB输入训练集和测试集，返回model训练好的模型和soc/5平均准确率
    # 输入分割好的训练集特征和标签，测试集特征和标签，返回模型参数和准确率
    # 训练模型
    def trainlgb_model(self, X_train, X_test, y_train, y_test, topn,weightdict,class_w):
        params = self.params  # 超参数初始化
        num_class = max([len(set(y_train.values)), len(set(y_test.values))])  # 初始化类别数
        models = []
        cate_feat = []
        res_5_fold = []
        featureimportant = []
        pred = []
        params["num_class"] = num_class
        soc = 0
        sample_w = compute_sample_weight(class_weight=class_w,y=y_train)
        train_set = lgb.Dataset(X_train, y_train,weight=sample_w)
        val_set = lgb.Dataset(X_train, y_train)
        model = lgb.train(params, train_set, valid_sets=[val_set], verbose_eval=0,categorical_feature=cate_feat,)
        impfea = list(model.feature_importance())
        threshold = np.sort(impfea)[-topn]
        result = []
        for position, im in zip(range(0, len(impfea)), impfea):
            if im >= threshold:
                result.append(position)
                weightdict[position] += impfea[position]
        featureimportant.append(result)
        test_pred = model.predict(X_test)
        test_pred = np.argmax(test_pred, axis=1)
        pred.append(test_pred)
        res_5_fold.append(list(test_pred))
        soc += metrics.accuracy_score(list(y_test), test_pred)
        pred = np.array(pred)
        labelall = []
        for i in range(0, len(y_test)):
            temp = list(pred[:, i])
            a = max(temp, key=temp.count)
            labelall.append(a)
        #print(str('准确率') + "：accuracy_score=" + str(soc))
        return models, soc, labelall, featureimportant

        # from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        # pre_rec_fscore = precision_recall_fscore_support(y_test, test_pred, average=None)
        # cm = confusion_matrix(y_test, test_pred)  # 计算单个混淆矩阵值

    # 输出测试集特征和标签，返回每个样本属于每一类的概率矩阵final_wristr_fea2_withhea_acc_gyro_40pd_200hz(1)(1)(1).csv
    def new_person_predict(self, models, new_X, new_y, num_class):  # 预测
        U_new_X_pred = np.zeros([new_X.shape[0], num_class])  # 用于接受在U上的预测结果
        soc = 0
        labelall = []
        # 返回在十个模型上的平均准确率
        for model in models:
            new_X_pred = model.predict(new_X)
            new_X_pred_label = np.argmax(new_X_pred, axis=1)
            U_new_X_pred = U_new_X_pred + new_X_pred
            soc += metrics.accuracy_score(list(new_y), new_X_pred_label)
            labelall.append(new_X_pred_label)
        return U_new_X_pred / 10, soc / 10, labelall, featureimportant


'''
       Describe
       留一法进行验证，对准确率进行排序
       -----------------
       Parameters
       -----------------
       data:原始特征文件dataframe

       Return
       -----------------
       dictlevel1：等级1病人准确率
       dictlevel2:等级2病人准确率
       dictlevel3：等级3病人准确率
       '''


def leave_subject_to_one(data, choose_people, choose_label, chooseleve, choosepeople_top,weightdict,subject_id,features,label,class_w):
    # 记录人准确率，和等级准确率
    dictlevel1 = {}
    dictlevel2 = {}
    dictlevel3 = {}
    dictlevel4 = {}
    dictlevel5 = {}
    dictlevel0 = {}
    for people, labelll in zip(choose_people, choose_label):
        trainperson = []
        for peopletemp in choose_people:
            if peopletemp == people:
                None
            else:
                trainperson.append(peopletemp)
        testperson = []
        testperson.append(people)
        train_data, test_data = splitperson(data, subject_id, trainperson, testperson)
        train_x = train_data[features].copy()
        train_y = train_data[label]
        test_x = test_data[features]
        test_y = test_data[label]
        test_y = test_y
        lgbtrain = LGBModel()
        models, soc, sample_label, fft = lgbtrain.trainlgb_model(train_x, test_x, train_y, test_y, choosepeople_top,weightdict,class_w)
        if labelll == 0 or labelll == '0':
            dictlevel0[str(people)] = soc
        if labelll == 1 or labelll == '1':
            dictlevel1[str(people)] = soc
        if labelll == 2 or labelll == '2':
            dictlevel2[str(people)] = soc
        if labelll == 3 or labelll == '3':
            dictlevel3[str(people)] = soc
        if labelll == 4 or labelll == '4':
            dictlevel4[str(people)] = soc
        if labelll == 5 or labelll == '5':
            dictlevel5[str(people)] = soc
    return dictlevel0, dictlevel1, dictlevel2, dictlevel3, dictlevel4, dictlevel5


'''
       Describe
       构建平衡数据集
       -----------------
       Parameters
       -----------------
       dictlevel1：等级1病人准确率
       dictlevel2:等级2病人准确率
       dictlevel3：等级3病人准确率

       Return
       -----------------
       recodata:平衡数据集病人编号
       lebal2： 对应标签
       '''


def create_balance_dataset_rank(dict1, dictlevel1, dictlevel2, dictlevel3):
    dictlevel1 = sorted(dictlevel1.items(), key=lambda x: -x[1])
    dictlevel2 = sorted(dictlevel2.items(), key=lambda x: -x[1])
    dictlevel3 = sorted(dictlevel3.items(), key=lambda x: -x[1])
    recodata = []
    lebal2 = []
    for changdu in range(0, min(lenobj)):
        recodata.append(int(dictlevel1[changdu][0]))
        lebal2.append(2)
    for changdu in range(0, min(lenobj)):
        recodata.append(int(dictlevel2[changdu][0]))
        lebal2.append(3)
    for changdu in range(0, min(lenobj)):
        recodata.append(int(dictlevel3[changdu][0]))
        lebal2.append(4)
    return recodata, lebal2


'''
           Describe
           留下一交叉验证，保留错误index，每个人准确率，每一类准确率，每个人混淆矩阵，每一类混淆矩阵
           -----------------
           Parameters
           -----------------
           data：原始数据集特征文件
           recodata：挑选的病人名单
           lebal2：对应标签

           Return
           -----------------
           peopleacc:每一类病人准确率
           peoplelabel：病人对应标签
           precision_pre_people：病人precision
           F1SCORE_pre_people：病人F1-score
           recall_pre_people：病人召回率
           '''


def leave_subject_out(data, recodata, lebal2, trun, encoder, iencoder, chooselevel, choosepeople_top,weightdict,subject_id,features,label,dongzuo,trueall,preall,class_w):
    # 记录人准确率，和等级准确率
    people_result_all = pd.DataFrame()
    peopleacc = {}
    levelacc = {}
    peoplelabel = {}
    precision_pre_people = {}
    F1SCORE_pre_people = {}
    recall_pre_people = {}
    level_F1 = {}
    level_recall = {}
    level_precison = {}
    predictlabel = []
    truelabel = []
    FAEIMP = []
    for i in range(0, 6):
        levelacc[str(i)] = []
        level_F1[str(i)] = []
        level_recall[str(i)] = []
        level_precison[str(i)] = []
    people_result = pd.DataFrame()
    for people, labelll in zip(recodata, lebal2):
        trainperson = []
        for peopletemp in recodata:
            if peopletemp == people:
                None
            else:
                trainperson.append(peopletemp)

        testperson = []
        testperson.append(people)
        train_data, test_data = splitperson(data, subject_id, trainperson, testperson)
        train_x = train_data[features].copy()
        train_y = train_data[label]
        test_x = test_data[features]
        # print(test_x)
        test_y = test_data[label]
        test_y = test_y
        lgbtrain = LGBModel()
        models, soc, test_lable, featureimportant = lgbtrain.trainlgb_model(train_x, test_x, train_y, test_y,
                                                                            choosepeople_top,weightdict,class_w)

        FAEIMP.append(featureimportant)

        soc2 = accuracy_score(test_y, test_lable)
        peopleacc[str(people)] = soc2
        peoplelabel[str(people)] = labelll
        levelacc[str(labelll)].append(soc2)
        people_label = []
        for i in range(0, len(test_lable)):
            people_label.append(people)

        test_lable = [encoder[value] for value in test_lable]
        test_y = test_y.map(encoder)

        predictlabel.append(Counter(test_lable).most_common()[0][0])
        truelabel.append(Counter(test_y).most_common()[0][0])

        for all_test, all_true in zip(test_lable, test_y.tolist()):
            trueall.append(all_true)
            preall.append(all_test)

        temppeople = pd.DataFrame(people_label)
        temppeople.index = test_y.index
        temp = pd.DataFrame(test_lable)
        temp.index = test_y.index
        people_result = pd.concat([test_y, temp, temppeople], axis=1)
        people_result_all = pd.concat([people_result_all, people_result], axis=0)
        # f1，recall, 召回率
        pre, rec, f1, sup = precision_recall_fscore_support(test_y, test_lable, labels=chooselevel)
        precision_pre_people[str(people)] = pre[iencoder[int(labelll)]]
        recall_pre_people[str(people)] = rec[iencoder[int(labelll)]]
        F1SCORE_pre_people[str(people)] = f1[iencoder[int(labelll)]]
        level_precison[str(labelll)].append(pre[iencoder[int(labelll)]])
        level_recall[str(labelll)].append(rec[iencoder[int(labelll)]])
        level_F1[str(labelll)].append(f1[iencoder[int(labelll)]])
        cm = confusion_matrix(test_y, test_lable, labels=chooselevel)
        cm = pd.DataFrame(cm)
        #plt.clf()
        cm.columns = chooselevel
        cm.index = chooselevel
        #sns.heatmap(cm, cmap="YlGnBu_r", fmt="d", annot=True)
        #plt.ylabel('True label')
        #plt.xlabel('Predicted label')
        # plt.show()
        #plt.savefig(r'../../result/leave-one-out/matrix/{}/{}.{}.{}.jpg'.format(dongzuo, people, dongzuo, trun))
        # 留一完成后，开始预测剩下的人
    acc_report_df = classification_report(truelabel, predictlabel, output_dict=True)
    reportexport = pd.DataFrame(acc_report_df).transpose()
    reportexport.to_csv(r'../../result/leave_one_out/ACC_result/{}.2.{}_report.csv'.format(dongzuo, trun))

    return peopleacc, peoplelabel, precision_pre_people, F1SCORE_pre_people, recall_pre_people, levelacc, level_F1, level_recall, level_precison, people_result_all, FAEIMP,trueall,preall


'''
               Describe
               结果保存
               ACC X.1保存动作X每个人准确率
               ACC X.2保存动作X总体每一类平均准确率
               INDEX：保存动作分类索引，预测标签，真实标签
               MATRIX：保存混淆矩阵
               -----------------
               Parameters
               -----------------
               peopleacc:每一类病人准确率
               peoplelabel：病人对应标签
               precision_pre_people：病人precision
               F1SCORE_pre_people：病人F1-score
               recall_pre_people：病人召回率
               Return
               -----------------
               None
           '''


def save_result(peopleacc, peoplelabel, precision_pre_people, F1SCORE_pre_people, recall_pre_people, dongzuo, levelacc,
                level_F1, level_recall, level_precison, people_result_all, trun, chooselevel,trueall,preall):
    people_result_all.columns = ['true_label', 'pre_label', 'subject_id']
    people_result_all = people_result_all.reset_index()
    people_result_all.to_csv(r'../../result/leave_one_out/index/{}.{}.csv'.format(dongzuo, trun))
    peopleacc = pd.DataFrame([peopleacc])
    peoplelabel = pd.DataFrame([peoplelabel])
    precision_pre_people = pd.DataFrame([precision_pre_people])
    F1SCORE_pre_people = pd.DataFrame([F1SCORE_pre_people])
    recall_pre_people = pd.DataFrame([recall_pre_people])
    precision_pre_people = pd.DataFrame(precision_pre_people.values.T, columns=precision_pre_people.index,
                                        index=precision_pre_people.columns)
    F1SCORE_pre_people = pd.DataFrame(F1SCORE_pre_people.values.T, columns=F1SCORE_pre_people.index,
                                      index=F1SCORE_pre_people.columns)
    recall_pre_people = pd.DataFrame(recall_pre_people.values.T, columns=recall_pre_people.index,
                                     index=recall_pre_people.columns)
    peopleacc = pd.DataFrame(peopleacc.values.T, columns=peopleacc.index, index=peopleacc.columns)
    peoplelabel = pd.DataFrame(peoplelabel.values.T, columns=peoplelabel.index, index=peoplelabel.columns)
    peopleacc[0] = peopleacc[0].apply(lambda x: x * 100)
    peopleacc[0] = peopleacc[0].apply(lambda x: format(x, '.2f'))
    result = pd.concat([peopleacc, peoplelabel, precision_pre_people, F1SCORE_pre_people, recall_pre_people], axis=1)
    result.columns = ['ACC', 'LABLE', 'precision', 'F1-SCORE', 'RECALL']
    print('*********************************')
    print('每个人的准确率')
    print(result)
    result.to_csv(r'../../result/leave_one_out/ACC_result/{}.1.{}.csv'.format(dongzuo, trun))
    print('*********************************')
    for avge in range(0, 6):
        levelacc[str(avge)] = np.mean(levelacc[str(avge)])
        level_recall[str(avge)] = np.mean(level_recall[str(avge)])
        level_F1[str(avge)] = np.mean(level_F1[str(avge)])
        level_precison[str(avge)] = np.mean(level_precison[str(avge)])
    print('*********************************')
    print('总体准确率')
    levelacc = pd.DataFrame([levelacc])
    level_F1 = pd.DataFrame([level_F1])
    level_precison = pd.DataFrame([level_precison])
    level_recall = pd.DataFrame([level_recall])
    levelacc = pd.DataFrame(levelacc.values.T, columns=levelacc.index, index=levelacc.columns)
    level_precison = pd.DataFrame(level_precison.values.T, columns=level_precison.index, index=level_precison.columns)
    level_recall = pd.DataFrame(level_recall.values.T, columns=level_recall.index, index=level_recall.columns)
    level_F1 = pd.DataFrame(level_F1.values.T, columns=level_F1.index, index=level_F1.columns)
    levelacc[0] = levelacc[0].apply(lambda x: x * 100)
    levelacc[0] = levelacc[0].apply(lambda x: format(x, '.2f'))
    levelacc = pd.concat([levelacc, level_precison, level_F1, level_recall], axis=1)
    levelacc.columns = ['ACC', 'precison', 'F1', 'recall']
    levelacc.to_csv(r'../../result/leave_one_out/ACC_result/{}.2.{}.csv'.format(dongzuo, trun))
    # eopleacc[0] = peopleacc[0].apply(lambda x: str(x * 100) + '%')
    print(levelacc)
    cm = confusion_matrix(trueall, preall, labels=chooselevel)
    cm = pd.DataFrame(cm)
    #plt.clf()
    cm.columns = chooselevel
    cm.index = chooselevel
    #sns.heatmap(cm, cmap="YlGnBu_r", fmt="d", annot=True)
    # cm.columns =set(test_lable)lightgbm as lgb
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    #plt.savefig(r'../../result/leave-one-out/matrix/{}.all.jpg'.format(dongzuo))


'''
               Describe
               按分布创建平衡数据集
               -----------------
               Parameters
               -----------------
               dictlevel1：等级1病人准确率
               dictlevel2:等级2病人准确率
               dictlevel3：等级3病人准确率

               Return
               -----------------
                recodata:平衡数据集病人编号
                lebal2： 对应标签
               None
           '''





'''
               Describe
               保存平衡数据集的平均结果
               -----------------
               Parameters
               -----------------
               dongzuo：当前执行的动作
               turn:总共训练的模型数

               Return
               -----------------
               None
           '''


def averageacc(dongzuo, turn):
    levelacc = {}
    level_F1 = {}
    level_precison = {}
    level_recall = {}
    acc = []
    F1 = []
    recall = []
    precison = []
    for i in range(0, turn):
        filename = r'../../result/leave_one_out/ACC_result/{}.2.{}.csv'.format(dongzuo, i)
        data = pd.read_csv(filename, header=0)
        acc.append(data['ACC'].values.tolist())
        F1.append(data['F1'].values.tolist())
        recall.append(data['recall'].values.tolist())
        precison.append(data['precison'].values.tolist())
    for avge in range(0, 5):
        acc1 = []
        F11 = []
        recall1 = []
        precison1 = []
        for j in range(0, turn):
            acc1.append(acc[j][avge])
            F11.append(F1[j][avge])
            recall1.append(recall[j][avge])
            precison1.append(precison[j][avge])
        levelacc[str(avge)] = np.mean(acc1)
        level_F1[str(avge)] = np.mean(F11)
        level_precison[str(avge)] = np.mean(precison1)
        level_recall[str(avge)] = np.mean(recall1)
    levelacc = pd.DataFrame([levelacc])
    level_F1 = pd.DataFrame([level_F1])
    level_precison = pd.DataFrame([level_precison])
    level_recall = pd.DataFrame([level_recall])
    levelacc = pd.DataFrame(levelacc.values.T, columns=levelacc.index, index=levelacc.columns)
    level_precison = pd.DataFrame(level_precison.values.T, columns=level_precison.index, index=level_precison.columns)
    level_recall = pd.DataFrame(level_recall.values.T, columns=level_recall.index, index=level_recall.columns)
    level_F1 = pd.DataFrame(level_F1.values.T, columns=level_F1.index, index=level_F1.columns)
    levelacc = pd.concat([levelacc, level_precison, level_F1, level_recall], axis=1)
    levelacc.columns = ['ACC', 'precison', 'F1', 'recall']
    levelacc.to_csv(r'../../result/leave_one_out/ACC_result_avge/{}.csv'.format(dongzuo))


def avgerecall_precion(dongzuo, turn):
    for i in range(0, turn):
        filename = r'../../result/leave_one_out/ACC_result/{}.2.{}_report.csv'.format(dongzuo, i)
        datatemp = pd.read_csv(filename, header=0)


def avgerecall_precion(dongzuo, turn):
    datacurrent = []
    for i in range(0, turn):
        filename = r'../../result/leave_one_out/ACC_result/{}.2.{}_report.csv'.format(dongzuo, i)
        df = pd.read_csv(filename, header=0)
        df.set_index(["Unnamed: 0"], inplace=True)
        datacurrent.append(df)
    df3 = pd.DataFrame()
    for i in range(0, turn):
        df3 = df3.add(datacurrent[i], fill_value=0)
    df3 = df3.div(turn)
    del df3['support']
    df3 = df3.drop('macro avg')
    df3 = df3.drop('weighted avg')
    df3 = df3.mul(100)
    df3 = df3.round(2)
    print("最终验证准确率")
    print('/***********************************/')
    print(df3)
    df3.to_csv(r'../../result/leave_one_out/ACC_result_avge/{}_report.csv'.format(dongzuo))


def saveresult2(dongzuo, turn):
    lunciacc = []
    accfinal = {}
    levelacc2 = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    for i in range(0, turn):
        acc = []
        levelacc = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        filename = r'../../result/leave_one_out/ACC_result/{}.1.{}.csv'.format(dongzuo, i)
        datatemp = pd.read_csv(filename, header=0)
        ddt = datatemp['ACC'].values.tolist()
        labellll = datatemp['LABLE'].values.tolist()
        for dt, lab in zip(ddt, labellll):
            if dt > 50:
                acc.append(1)
                levelacc[lab].append(1)
            else:
                acc.append(0)
                levelacc[lab].append(0)
        lunciacc.append(levelacc)
    for i in range(0, turn):
        temp = lunciacc[i]
        for j in range(0, 6):
            levelacc2[j].append(np.mean(temp[j]))
    for k in range(0, 6):
        accfinal[k] = np.mean(levelacc2[k])
    exp = pd.DataFrame([accfinal])
    exp = pd.DataFrame(exp.values.T, columns=exp.index, index=exp.columns)
    exp.columns = ['ACC']
    exp.to_csv(r'../../result/leave_one_out/ACC_result_avge/{}_person_ACC.csv'.format(dongzuo))

