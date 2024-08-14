import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from src.utils.PDDataLoader import PDDataLoader
from sklearn.metrics import f1_score
from autofeatselect import AutoFeatureSelect
from src.utils import set_seed

seed = set_seed(0)


# class FeatureSelector:
#     def __init__(self, activity_id, data, target):
#         self.activity_id = activity_id
#         self.data = data
#         self.target = target,
#         self.features = data.columns
#         self.missing_threshold = 0.6
#         self.low_importance_threshold = 0.01
#
#     def remove_features(self):
#         self.data = self.data.drop(
#             columns=self.single_unique_features + self.zero_importance_features + self.low_importance_features)
#         print(f"Removed features. Remaining features: {self.data.shape[1]}")
#         return self.data


def identify_single_unique(activity_id):
    data_path = f'../../../output/activity/step_4_feature_selection'
    data_name = f'acc_data_4_activity_{activity_id}.csv'
    data = pd.read_csv(os.path.join(data_path, data_name))
    unique_counts = data.nunique()
    single_unique_features = list(unique_counts[unique_counts == 1].index)
    print(f"Identified {len(single_unique_features)} features with a single unique value.")
    return single_unique_features


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
                                      seed=seed)
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
    # selection_methods = ['lgbm', 'xgb', 'rf','boruta']
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
    # ranking_columns = ['boruta_ranking', 'lgbm_ranking', 'xgb_ranking', 'rf_ranking']

    # 计算每个特征的平均排名
    ranking_df['average_ranking'] = ranking_df[ranking_columns].mean(axis=1)
    # 根据平均排名对特征进行排序
    sorted_features = ranking_df.sort_values(by='average_ranking').index
    # 初始化一个 DataFrame 来保存结果
    results_df = pd.DataFrame(columns=['num_features', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'mean_score'])
    # 设置 LightGBM 的参数（示例）
    params = {
        'boosting': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_error',
        'num_class': len(set(data_loader.severity_mapping.values())),
        'max_depth': 7,
        'seed': seed,
        'verbose': -1,
    }

    # 使用交叉验证评估不同数量特征的模型性能
    for n in range(1, len(sorted_features) + 1):
        print(f'Using {n} feature(s)')
        selected_features = sorted_features[:n]
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
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
            )
            y_pred = model.predict(new_test_X, num_iteration=model.best_iteration)
            y_pred_labels = y_pred.argmax(axis=1)
            f1 = f1_score(new_test_Y, y_pred_labels, zero_division=0, average='macro')
            f1_scores.append(f1)

        mean_f1_score = np.mean(f1_scores)
        print('mean_f1_score ', mean_f1_score)

        # 将结果保存到 DataFrame 中
        result_row = pd.DataFrame([{
            'num_features': n,
            'fold_1': f1_scores[0],
            'fold_2': f1_scores[1],
            'fold_3': f1_scores[2],
            'fold_4': f1_scores[3],
            'fold_5': f1_scores[4],
            'mean_score': mean_f1_score
        }])
        # 在拼接之前，排除 results_df 和 result_row 中全是 NA 的列
        results_df_filtered = results_df.dropna(axis=1, how='all')
        result_row_filtered = result_row.dropna(axis=1, how='all')
        # 仅拼接非空的 DataFrame
        if not result_row_filtered.empty:
            results_df = pd.concat([results_df_filtered, result_row_filtered], ignore_index=True)
        else:
            results_df = results_df_filtered.copy()

    if not results_df.empty:
        best_num_features = results_df.loc[results_df['mean_score'].idxmax(), 'num_features']
        print(f'最佳特征数量: {best_num_features}')
        print(results_df)
        # 保存结果到新的文件
        results_df.to_csv(
            f'../../../output/activity/step_4_feature_selection/best_num_features_activity_{activity_id}.csv',
            index=False)
    else:
        print("No valid results were obtained. Please check your data and parameters.")


def important_feature_columns(activity_id):
    ranking_importance_path = f'../../../output/activity/step_4_feature_selection/feature_selection_results_activity_{activity_id}.csv'
    ranking_df = pd.read_csv(ranking_importance_path)
    # 指定包含排名的列
    ranking_columns = ['rfecv_rankings', 'boruta_ranking', 'lgbm_ranking', 'xgb_ranking', 'rf_ranking',
                       'permutation_ranking']

    # 计算每个特征的平均排名
    ranking_df['average_ranking'] = ranking_df[ranking_columns].mean(axis=1)
    # 根据平均排名对特征进行排序
    sorted_features = ranking_df.sort_values(by='average_ranking')
    # patch 消除唯一值Feature
    unique_columns = ['acc_a_max_a', 'acc_a_max_x', 'acc_a_max_y', 'acc_a_max_z', 'acc_a_mean_a', 'acc_a_mean_x',
                      'acc_a_mean_y', 'acc_a_mean_z']
    # 返回剩余的值
    remaining_values = [value for value in sorted_features['feature'] if value not in unique_columns]
    return remaining_values


def save_important_feature(activity_id: int):
    # 加载特征选择文件
    feature_path = f'../../../output/activity/step_4_feature_selection'
    feature_name = f'feature_selection_results_activity_{activity_id}.csv'
    feature = pd.read_csv(os.path.join(feature_path, feature_name))
    # 加载活动数据
    data_path = "../../../output/activity/step_2_select_sensors"
    data_name = "acc_data.csv"
    data = pd.read_csv(os.path.join(data_path, data_name))
    feature_column = important_feature_columns(activity_id)
    label_info = ['PatientID', 'activity_label', 'Severity_Level']
    data = data[feature_column + label_info]
    activity_id_filtered_data = data[data['activity_label'] == activity_id]
    # 保存文件
    file = os.path.join(feature_path, f'acc_data_activity_{activity_id}.csv')
    activity_id_filtered_data.to_csv(file, index=False)


if __name__ == '__main__':
    for i in range(1, 17):
        # 选择哪些特征需要保留
        # single_activity_feature_selection(i)
        # # 选择最佳特征数量
        save_important_feature(i)
        print(f"saved activity {i}")
