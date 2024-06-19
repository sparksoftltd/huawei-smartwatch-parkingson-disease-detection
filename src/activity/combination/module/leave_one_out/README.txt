用于留一交叉验证的功能函数

choosepeople：
'''
       Describe
       用于选出对应等级的病人，并对病人标签进行编码
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
       dict1：每一类对应病人编号）字典0      
'''

splitdata：
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


class LGB()
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


leave_subject_to_one
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

create_balance_dataset
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



leave_subject_out
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


save_result
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


执行流程
1.执行choosepeople，选择制定患病等级的病人
2.执行creat_balance_dataset采样平衡数据集
3.执行下leave_one_out 进行留一验证并保存结果

















