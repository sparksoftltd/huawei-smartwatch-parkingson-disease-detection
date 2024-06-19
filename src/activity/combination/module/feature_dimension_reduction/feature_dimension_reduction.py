import pandas as pd
def feature_seletion(sequence,choosefeaturenum):
    temp=pd.DataFrame()
    global data
    data = pd.read_csv(r"../../datasets/242D_HC_PD_window300_wristr_acc_gro_200hz.csv")
    #data = pd.read_csv(r"242D_HC_PD_window300_wristr_acc_gro_200hz.csv")
    data.loc[data['severity_level'] == 0, 'subject_id'] += 15
    for se in sequence:
        print('正在提取动作{}的重要特征'.format(se))
        tempdata=pd.read_csv(r"../../datasets/featureimportant/feature_important{}.csv".format(se))
        #tempdata = pd.read_csv(r"feature_important/feature_important{}.csv".format(se))
        importantfeaturelist=[str(i) for i in  tempdata['feature'].values.tolist()]
        importantfeaturelist=importantfeaturelist[:choosefeaturenum]
        print(importantfeaturelist)
        importantfeaturelist.append('Activity_id')
        importantfeaturelist.append('subject_id')
        importantfeaturelist.append('severity_level')
        cdata=(data.loc[data['Activity_id']==se])[importantfeaturelist]
        colums=[str(i) for i in range(0,len(importantfeaturelist)-3)]
        colums.append('Activity_id')
        colums.append('subject_id')
        colums.append('severity_level')
        cdata.columns=colums
        temp=pd.concat([temp,cdata])
    print('特征挑选完成')
    print('正在保存特征')
    data = temp
    data.to_csv(r"../../datasets/featureimportant/important_all{}.csv".format(choosefeaturenum), index=False)
    print('特征保存成功')
if __name__ == '__main__':
    sequence=[2,3,4,5]
    times = 24
    choosefeaturenum=20
    feature_seletion(sequence,choosefeaturenum)