import pandas as pd
'''
       Describe
       选择指定活动进行拼接
       -----------------
       Parameters
       -----------------
       data:原始特征文件
       sequence：活动号序列
       samples:样本数量
       severity_level：患病等级列数字
       subject_id:病人编号所在列的数字
       info1:保存疾病等级信息
       info2:保存病人序号信息
       
       Return
       -----------------
       df6：活动拼接后的csv文件      '''


def choose_activity(sequence,samples,numnum):
    global name1
    global name2
    if numnum==242:
       file="242D_HC_PD_window300_wristr_acc_gro_200hz.csv"
    else:
      file="featureimportant/important_all{}.csv".format(numnum)
      numnum=str(numnum)+" " 
    print('正在进行活动拼接...每个活动包含{}维'.format(numnum))
    global df6
    df6 = pd.DataFrame()
    yingshe = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'n', 'j', 'k', 'l', 'm', 'cc']
    for number in range(len(sequence),len(sequence)+1):
        sequence1=sequence[:number]
        action=[]
        for num in sequence1:
            action.append(yingshe[num-1])
        action.append('info1')
        action.append('info2')
        df3 = pd.DataFrame()
        df6 = pd.DataFrame()
        for i in range(1,88,1):
            people = [i]
            data=pd.read_csv(r"../../datasets/{}".format(file))
            data.loc[data['severity_level'] == 0, 'subject_id'] += 15
            # datahealth=data.loc[data['severity_level'] == 0]
            #datahealth['subject_id'] = datahealth['subject_id'] + 17
            datainfo=data.loc[data['subject_id'].isin(people)]
            info1=datainfo['severity_level']
            info2=datainfo['subject_id']
            info1=info1[:samples]
            info2=info2[:samples]
            info1=info1.reset_index(drop=True)
            info2=info2.reset_index(drop=True)

            #1拇食指捏合
            a=data.loc[(data['Activity_id']==1)& (data['subject_id'].isin(people))]
            a=a.iloc[:samples,:-3]
            a=a.reset_index(drop=True)

            # # 2握拳张开
            b = data.loc[(data['Activity_id'] == 2) & (data['subject_id'].isin(people))]
            b = b.iloc[:samples, :-3]
            b = b.reset_index(drop=True)
            # # # #
            #3手腕翻转
            c=data.loc[(data['Activity_id']==3)& (data['subject_id'].isin(people))]
            c=c.iloc[:samples,:-3]
            c=c.reset_index(drop=True)
            # # # #
            #4右手翻转
            d=data.loc[(data['Activity_id']==4)& (data['subject_id'].isin(people))]
            d=d.iloc[:samples,:-3]
            d=d.reset_index(drop=True)
            # # #
            # 5左手翻转
            e = data.loc[(data['Activity_id'] == 5) & (data['subject_id'].isin(people))]
            e = e.iloc[:samples, :-3]
            e = e.reset_index(drop=True)
            # # #
            #6左手指鼻尖
            f = data.loc[(data['Activity_id']==6) & (data['subject_id'].isin(people))]
            f = f.iloc[:samples, :-3]
            f = f.reset_index(drop=True)
            #
            #7右手指鼻尖
            g=data.loc[(data['Activity_id']==7)&(data['subject_id'].isin(people))]
            g=g.iloc[:samples,:-3]
            g=g.reset_index(drop=True)
            # #
            #8抬手直立
            h=data.loc[(data['Activity_id']==8)&(data['subject_id'].isin(people))]
            h=h.iloc[:samples,:-3]
            h=h.reset_index(drop=True)
            # # # #
            #9往返走
            n=data.loc[(data['Activity_id']==9)&(data['subject_id'].isin(people))]
            n=n.iloc[:samples,:-3]
            n=n.reset_index(drop=True)
            # #
            #10起立
            j=data.loc[(data['Activity_id']==10)&(data['subject_id'].isin(people))]
            j=j.iloc[:samples,:-3]
            j=j.reset_index(drop=True)
            # # #
            #11喝水
            k=data.loc[(data['Activity_id']==11)&(data['subject_id'].isin(people))]
            k=k.iloc[:samples,:-3]
            k=k.reset_index(drop=True)
            # #
            #12弯腰捡东西
            l=data.loc[(data['Activity_id']==12)&(data['subject_id'].isin(people))]
            l=l.iloc[:samples,:-3]
            l=l.reset_index(drop=True)

            # #
            #静坐
            m=data.loc[(data['Activity_id']==13)&(data['subject_id'].isin(people))]
            m=m.iloc[:samples,:-3]
            m=m.reset_index(drop=True)

            # #静站
            cc=data.loc[(data['Activity_id']==14)&(data['subject_id'].isin(people))]
            cc=cc.iloc[0:samples,:-3]
            cc=cc.reset_index(drop=True)


            datacontact=[]
            for dongzuo in action:
                datacontact.append(eval(dongzuo))
            df3 = pd.concat(datacontact, axis=1, ignore_index=True)
            df6 = pd.concat([df6, df3])
        
        name=r'../../result/activity_combination/{}_comb'.format(numnum)
        counttt=0
        while number>0:
            number=number-1
            name=name+str(sequence1[counttt])+'+'
            counttt=counttt+1   
        name=name.strip('+')
        name=name+'.csv'
        print('已成功拼接活动，生成活动{}拼接后特征文件'.format(name[42:-4]))
        #print(name)
        df6 = df6.rename(columns={df6.columns[-2]: 'severity_level', df6.columns[-1]: 'subject_id'})
        df6.to_csv(name,index=False)
        if numnum==242:
           name1=name
        else:
           name2=name

def output_Feacom_file(num):
    if num==1:
      print("拼接后特征文件(未降维)")
      data=pd.read_csv(name1)
      print(data)
    else:
      print("拼接后特征文件(降维)")
      data=pd.read_csv(name2)
      print(data)
    


if __name__ == '__main__':
    sequence=[2,3]
    times = 24
    choose_activity(sequence,times)
