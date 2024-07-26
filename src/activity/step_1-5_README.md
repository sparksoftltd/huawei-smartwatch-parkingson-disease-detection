## step1-7简要说明

## step1 特征提取

```python
# 输入文件
feature_label.to_csv(r'../../../output/activity/step_1_feature_extraction' + r"/" + side_r + "_acc_gyro_mag_feature_label.csv", index=False)

# 将特征保存为csv文件，对应输出文件
feature_label.to_csv(r'../../../output/activity/step_1_feature_extraction' + r"/" + side_r + "_acc_gyro_mag_feature_label.csv", index=False)
```

## step2 选择指定传感器特征

```
    # 提取step输出文件（特征文件）的指定列，这些类为acc相关的220维度特征
    select_data(['acc',
                 ])
    # 生成筛选后的特征文件
        output_name_path = os.path.join(r'../../../output/activity/step_2_select_sensors', f'{output_name}')

```

## step3 输出特征重要性

```python
  #输入
    #这是PDClassifier输入两个文件，
    # acc_data.csv为Step2的筛选后的acc相关特征
    # fold_groups_new_with_combinations.csv文件会根据选择的activity_id，确定五折交叉验证中训练集和验证集
            classifier = PDClassifier(r"../../../output/activity/step_2_select_sensors/acc_data.csv", activity_id,
                                  r'../../../input/activity/step_2_select_sensors/fold_groups_new_with_combinations.csv')  # 初始化PDClassifier分类器
   
   # 输出每个activity_id数据对应的特征重要性排序文件，会生成16个动作的特征重要性文件
shap_summary.to_csv(os.path.join(r'../../../output/activity/step_3_output_feature_importance' ,
            f'{os.path.basename(self.data_path)}_{self.activity_id}_shap_importance.csv'), index=False
        )
```

## step4 特征降维

```python
#输入

#1、acc相关特征的特征文件
data_path_param1 = r"../../../output/activity/step_2_select_sensors/acc_data.csv"
#2、 指定需要降低的维度是多少这里选择20
    choosefeaturenum_param3 = 20
#3、指定需要处理降维的activity_id是哪一些，默认是将全部1-16序号动作全部降低纬度    
sequence_param2 = list(range(1, 17))
#4、对应每个activity_id特征重要性文件
        tempdata = pd.read_csv(os.path.join(r'../../../output/activity/step_3_output_feature_importance',
            f'{os.path.basename(data_path)}_{se}_shap_importance.csv'))

#输出activity_id为1-17的前20重要维度特征的降维后的特征文件，该特征文件每个activity_id对应的特征为20维度。
    filename = rf"../../../output/activity/step_4_feature_selection/{os.path.splitext(os.path.basename(data_path))[0]}_important_all_{choosefeaturenum}.csv"
```

## step5 无折交叉验证

```python
# 输入
# 1、选择需要五折交叉验证的activity_id是哪一些
    activity_range = list(range(1, 17))  # 16个活动
# 2、选择针对每个activity_id数据进行的训练预测的模型是哪一些
    classifier_range = ['rf', 'xgb', 'lgbm', 'logistic_l1', 'logistic_l2', 'svm_l1', 'svm_l2', 'knn', 'bayes',
                        'mlp_2', 'mlp_4', 'mlp_8']
# 3、选择acc_data_important_all_20.csv降维后的特征文件、选择activity_id作训练和测试，选择fold_groups_new_with_combinations.csv（其中存储了每个activity_id对应的五折交叉验证的分组信息，按照里面的分组进行训练和预测）
    for activity_id in tqdm(activity_range):
        classifier = PDClassifier(r"../../../output/activity/step_4_feature_selection/acc_data_important_all_20.csv", activity_id,
                                  r'../../../input/activity/step_3_output_feature_importance/fold_groups_new_with_combinations.csv')  # 初始化PDClassifier分类器
# 输出记录训练测试过程的日志文件
# 设置日志的配置，美包含INFO标签
# 1、训练日志
logging.basicConfig(
    filename=rf"../../../output/activity/step_5_five_fold_cross_validation/{month_day}_model_training.log",
    level=logging.INFO,
    format='%(message)s',
    filemode='a'
)
# 2、测试结果文件汇总（表格）
        file_path = os.path.join(r'../../../output/activity/step_5_five_fold_cross_validation', file_path)

```

## step6 活动拼接

```python
# 输入
# 1、指定降维后的acc特征文件
data_param1 = r"../../../output/activity/step_4_feature_selection/acc_data_important_all_20.csv"  # 14
# 2、传入需要拼接activity_id序列（列表列性），拼接方案scheme
# # - scheme: 整数，组合方案类型，0 表示垂直拼接，1 表示水平拼接，2 表示融合
#   scheme=0 垂直拼接，数据按照垂直方向，将活动数据按sequence次序垂直排列，先排列sequence[i]的全部数据，再排列sequence[i+1]的全部数据。
#   scheme=1 水平拼接，将按照sequence次序水平拼接，因为不同活动的执行时间不一样，可能出现水平上的NaN值。
#   scheme=2 数据融合，将活动数据按sequence次序，按照sequence[i]对应的活动数据*wights[i]+sequence[i+1]*wights[i+1]融合全部数据，
#   注意当遇到受试者确实sequence中指定活动序号的数据时，放弃该受试者的数据,不参与数据融合。

# - samples: 整数，非负，指定每个活动截取的窗口数
# - weights: 列表，数值型，指定融合时的权重。scheme 为 2 时必须提供，表示按照累加sequence[i]序号活动数据*weights[i]融合。
# 在入口程序目录下生成 m命名为：活动拼接次序按+号连接_拼接方式_comb.csv 形如：1+2_horizontal_comb.csv，活动1+活动2水平拼接的csv文件

        activityCombinationPreprocessor = ActivityCombinationPreprocessor(data_param1,
                                                                          sequence=list(sequence), scheme=1,
                                                                          samples=None)
# 输出对应拼接后的特征数据文件
        file_path = fr"../../../output/activity/step_6_comb/{filename}"
        # data.to_csv(filename, index=False)
        data.to_csv(file_path, index=False)

```

## step7 无折交叉验证

```python
# 同step5；
# 输入选择activity_id需要注意，这里需要选择对应的活动拼接后特征文件；
# 如9+11_None_h.csv这是activity_id9和11横向拼接的特征文件（命名格式为9+11表示拼接的互动序号，None表示没有指定每个activity_id对应每个人数据的长度、h表示Horizontal横向拼接，v表示Vertical垂直拼接，f表示fuse加权融合）。
```



