import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from src.utils.PDDataLoader import PDDataLoader
from sklearn.metrics import f1_score
from autofeatselect import AutoFeatureSelect


class FeatureSelector:
    def __init__(self, data):
        self.data = data
        self.correlation_threshold = None
        self.one_hot_correlated = False
        self.base_features = data.columns.tolist()
        self.one_hot_features = []
        self.corr_matrix = None
        self.record_collinear = pd.DataFrame()
        self.ops = {'collinear': []}

    def identify_collinear(self, correlation_threshold, one_hot=False):
        """
        Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greater than `correlation_threshold`,
        only one of the pair is identified for removal.

        Parameters
        --------
        correlation_threshold : float between 0 and 1
            Value of the Pearson correlation coefficient for identifying correlation features

        one_hot : boolean, default = False
            Whether to one-hot encode the features before calculating the correlation coefficients
        """

        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot

        # Calculate the correlations between every column
        if one_hot:
            # One hot encoding
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]

            # Add one hot encoded data to original data
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

            corr_matrix = features.corr()

        else:
            corr_matrix = self.data.corr()

        self.corr_matrix = corr_matrix

        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Select the features with correlations above the threshold
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = []

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to list if temp_df is not empty and has no all-NA columns
            if not temp_df.empty and temp_df.notna().any().any():
                record_collinear.append(temp_df)

        # Concatenate all the dataframes in the list
        if record_collinear:
            self.record_collinear = pd.concat(record_collinear, ignore_index=True)
        else:
            self.record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        self.ops['collinear'] = to_drop

        print('%d features with a correlation magnitude greater than %0.2f.\n' % (
            len(self.ops['collinear']), self.correlation_threshold))

        return self.ops['collinear']


def check_features(activity_id):
    data_path = "../../../output/activity/step_2_select_sensors"
    data_name = "acc_data.csv"
    fold_groups_path = "../../../input/activity/step_3_output_feature_importance"
    fold_groups_name = "fold_groups_new_with_combinations.csv"
    severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

    classifier = PDDataLoader(activity_id, os.path.join(data_path, data_name),
                              os.path.join(fold_groups_path, fold_groups_name), severity_mapping=severity_mapping)
    target = classifier.PD_data['Severity_Level']  # Replace 'target' with your actual target column name

    # Create the feature selector
    fs = FeatureSelector(classifier.PD_data[classifier.feature_name], target)

    # Identify and remove unwanted features
    missing_features = fs.identify_missing()
    collinear_features = fs.identify_collinear()
    zero_importance_features = fs.identify_zero_importance()
    low_importance_features = fs.identify_low_importance()
    single_unique_features = fs.identify_single_unique()


def single_activity_feature_selection(activity_id):
    # 准备数据
    data_path = "../../../output/activity/step_2_select_sensors"
    data_name = "acc_data.csv"
    fold_groups_path = "../../../input/activity/step_3_output_feature_importance"
    fold_groups_name = "fold_groups_new_with_combinations.csv"
    severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
    data_loader = PDDataLoader(activity_id, os.path.join(data_path, data_name),
                               os.path.join(fold_groups_path, fold_groups_name), severity_mapping=severity_mapping)

    # 设置分类特征
    num_feats = [c for c in data_loader.feature_name]
    data_loader.PD_data[num_feats] = data_loader.PD_data[num_feats].astype('float')
    train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids = data_loader.create_train_test_split(0,
                                                                                                      data_loader.fold_groups[
                                                                                                          0])
    new_test_X = np.vstack(test_X_ls)
    new_test_Y_ls = []
    for bag_test_Y, bag_test_X in zip(test_Y_ls, test_X_ls):
        new_test_Y_ls.append(np.full(bag_test_X.shape[0], bag_test_Y))
    new_test_Y = np.hstack(new_test_Y_ls)

    # 创建Dataloader以支持AutoFeatureSelect类
    train_X = pd.DataFrame(train_X, columns=num_feats)
    new_test_X = pd.DataFrame(new_test_X, columns=num_feats)
    train_Y = pd.Series(train_Y)
    new_test_Y = pd.Series(new_test_Y)
    # 创建AutoFeatureSelect类
    feat_selector = AutoFeatureSelect(modeling_type='classification',
                                      X_train=train_X,
                                      y_train=train_Y,
                                      X_test=new_test_X,
                                      y_test=new_test_Y,
                                      numeric_columns=num_feats,
                                      categorical_columns=[],
                                      seed=0)
    # 检测相关特征
    corr_features = feat_selector.calculate_correlated_features(static_features=None,
                                                                num_threshold=0.9,
                                                                cat_threshold=0.9)
    # 删除相关特征
    feat_selector.drop_correlated_features()

    # 确定要应用的选择方法
    # 所有方法的超参数都可以更改
    selection_methods = ['lgbm', 'xgb', 'rf', 'perimp', 'rfecv', 'boruta']
    # selection_methods = ['boruta']
    final_importance_df = feat_selector.apply_feature_selection(selection_methods=selection_methods,
                                                                lgbm_hyperparams=None,
                                                                xgb_hyperparams=None,
                                                                rf_hyperparams=None,
                                                                lassocv_hyperparams=None,
                                                                perimp_hyperparams=None,
                                                                rfecv_hyperparams=None,
                                                                boruta_hyperparams=None)
    # 定义一个字典，将方法名映射到相应的重要性列名
    method_to_importance_column = {
        'lgbm': 'lgbm_importance',
        'xgb': 'xgb_importance',
        'rf': 'rf_importance',
        'perimp': 'permutation_importance'
    }

    # 遍历 selection_methods，添加排名列
    for method in selection_methods:
        if method in method_to_importance_column:
            importance_col = method_to_importance_column[method]
            rank_col_name = importance_col.replace('importance', 'ranking')
            final_importance_df[rank_col_name] = final_importance_df[importance_col].rank(ascending=False)

    file_path = f'../../../output/activity/step_4_feature_selection/feature_selection_results_activity_{activity_id}.csv'
    # 检查并创建目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 保存结果到 CSV 文件
    final_importance_df.to_csv(file_path, index=False)


def single_activity_best_num_features(activity_id):
    data_path = "../../../output/activity/step_2_select_sensors"
    data_name = "acc_data.csv"
    fold_groups_path = "../../../input/activity/step_3_output_feature_importance"
    fold_groups_name = "fold_groups_new_with_combinations.csv"
    severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
    data_loader = PDDataLoader(activity_id, os.path.join(data_path, data_name),
                               os.path.join(fold_groups_path, fold_groups_name), severity_mapping=severity_mapping)

    ranking_importance_path = f'../../../output/activity/step_4_feature_selection/feature_selection_results_activity_{activity_id}.csv'
    ranking_df = pd.read_csv(ranking_importance_path)
    # 指定包含排名的列
    ranking_columns = ['rfecv_rankings', 'boruta_ranking', 'lgbm_ranking', 'xgb_ranking', 'rf_ranking',
                       'permutation_ranking']

    # 计算每个特征的平均排名
    ranking_df['average_ranking'] = ranking_df[ranking_columns].mean(axis=1)
    # 根据平均排名对特征进行排序
    sorted_features = ranking_df.sort_values(by='average_ranking').index
    # 初始化一个 DataFrame 来保存结果
    results_df = pd.DataFrame(columns=['num_features', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'mean_score'])
    # 设置 LightGBM 的参数（示例）
    params = {
        'learning_rate': 0.02,
        'min_child_samples': 1,  # 子节点最小样本个数
        'max_depth': 7,  # 树的最大深度
        'lambda_l1': 0.25,  # 控制过拟合的超参数
        'lambda_l2': 0.25,  # 控制过拟合的超参数
        # 'min_split_gain': 0.015,  # 最小分裂增益
        'boosting': 'gbdt',
        'objective': 'multiclass',
        # 'n_estimators': 300,  # 决策树的最大数量
        'metric': 'multi_error',
        'num_class': len(set(data_loader.severity_mapping.values())),
        'feature_fraction': .75,  # 每次选取百分之75的特征进行训练，控制过拟合
        'bagging_fraction': .75,  # 每次选取百分之85的数据进行训练，控制过拟合
        'seed': 0,
        'num_threads': -1,
        'verbose': -1,
        # 'early_stopping_rounds': 50,  # 当验证集在训练一百次过后准确率还没提升的时候停止ss
        'num_leaves': 128,
    }

    # 使用交叉验证评估不同数量特征的模型性能
    for i in range(1, len(sorted_features) + 1):
    # for i in range(1, 100):
        print(f'Using {i} feature(s)')
        selected_features = sorted_features[:i]
        f1_scores = []

        for fold_num, test_ids in enumerate(data_loader.fold_groups):
            train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids = data_loader.create_train_test_split(fold_num,
                                                                                                              test_ids)

            # 选择前 i 个特征
            train_X = train_X[:, selected_features]
            test_X_ls = [test_X[:, selected_features] for test_X in test_X_ls]

            lgb_train = lgb.Dataset(train_X, train_Y)
            new_test_X = np.vstack(test_X_ls)
            new_test_Y_ls = []
            for bag_test_Y, bag_test_X in zip(test_Y_ls, test_X_ls):
                new_test_Y_ls.append(np.full(bag_test_X.shape[0], bag_test_Y))
            new_test_Y = np.hstack(new_test_Y_ls)
            lgb_test = lgb.Dataset(new_test_X, new_test_Y, reference=lgb_train)

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_test],
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(100)]
            )
            y_pred = model.predict(new_test_X, num_iteration=model.best_iteration)
            y_pred_labels = y_pred.argmax(axis=1)
            f1 = f1_score(new_test_Y, y_pred_labels, zero_division=0, average='macro')
            f1_scores.append(f1)

        mean_f1_score = np.mean(f1_scores)

        # 将结果保存到 DataFrame 中
        result_row = pd.DataFrame([{
            'num_features': i,
            'fold_1': f1_scores[0],
            'fold_2': f1_scores[1],
            'fold_3': f1_scores[2],
            'fold_4': f1_scores[3],
            'fold_5': f1_scores[4],
            'mean_score': mean_f1_score
        }])
        # 仅在 result_row 不为空时进行合并
        if not result_row.empty:
            results_df = pd.concat([results_df, result_row], ignore_index=True)

    if not results_df.empty:
        best_num_features = results_df.loc[results_df['mean_score'].idxmax(), 'num_features']
        print(f'最佳特征数量: {best_num_features}')
        print(results_df)
        # 保存结果到新的文件
        results_df.to_csv(f'../../../output/activity/step_4_feature_selection/best_num_features_activity_{activity_id}.csv', index=False)
    else:
        print("No valid results were obtained. Please check your data and parameters.")


if __name__ == '__main__':
    pass
    # 选择哪些特征需要保留
    # for i in range(1, 17):
    #     single_activity_feature_selection(i)

    # 选择最佳特征数量
    single_activity_best_num_features(1)
