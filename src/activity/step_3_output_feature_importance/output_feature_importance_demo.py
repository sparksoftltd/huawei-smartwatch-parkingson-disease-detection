import sys
import shap
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, make_scorer, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import json


class PDClassifier:
    def __init__(self, data_path, activity_id, fold_groups_path, severity_mapping=None):
        if severity_mapping is None:
            severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}  # 映射关系
        self.data_path = data_path  # 手工特征文件路径
        self.activity_id = activity_id  # 从手工特征文件中选取指定activity_id对应的数据
        self.severity_mapping = severity_mapping  # 映射关系
        self.PD_data = self.load_and_preprocess_data()  # 预处理数据文件
        self.feature_name = self.PD_data.columns[:-3]

        self.fold_groups = self.process_fold_groups(fold_groups_path)
        self.bag_data_dict, self.patient_ids, self.fold_groups = self.group_data(self.fold_groups)  # 组织数据

    def process_fold_groups(self, csv_file_path):
        """
        读取CSV文件，筛选出activity_label为3的行，并处理每组的HC、Hy1、Hy2、Hy3、Hy4列数据。
        参数:
        csv_file_path (str): CSV文件的路径
        返回:
        list: 处理后的fold_group_ls列表
        """
        # 读取CSV文件
        try:
            data = pd.read_csv(csv_file_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return []
        data['activity_label'] = data['activity_label'].astype(str)
        # 筛选活动 activity_id 的患者数据
        filtered_data = data[data['activity_label'] == str(self.activity_id)]
        # 准备一个空列表来收集所有组的数据
        fold_groups = []
        # 对每个组进行操作
        for i in range(1, 6):  # 从Group1到Group5
            group_data = []
            for col_type in ['HC', 'Hy1', 'Hy2', 'Hy3', 'Hy4']:  # 对每种类型HC, Hy1, Hy2, Hy3, Hy4进行迭代
                # 获取列名，例如 'Group1_HC'
                column_name = f'Group{i}_{col_type}'
                # 将列数据转换为列表并扩展到group_data中
                group_data.extend(json.loads(filtered_data.iloc[0][column_name]))
            # 将处理完的单个组数据添加到fold_group_ls中
            fold_groups.append(group_data)
        return fold_groups

    # 返回指定对应acitivity_id的数据
    def load_and_preprocess_data(self):
        data = pd.read_csv(self.data_path)
        # 检查 self.activity_id 是否可以转换为数字，并且是否在 1 到 16 之间
        try:
            activity_id_num = int(self.activity_id)
            if 1 <= activity_id_num <= 16:
                data = data.loc[data['activity_label'] == activity_id_num]
            else:
                raise AssertionError(f"activity_id {self.activity_id} is not in the range 1 to 16.")
        except ValueError:
            pass
        print(f"Loading data of activity_id: {self.activity_id}")
        data['Severity_Level'] = data['Severity_Level'].map(self.severity_mapping)
        data = data.dropna()
        # ============================标准化=====================================
        # 获取 'Severity_Level' 列之前的所有列名
        numerical_columns = data.columns[:-3]
        # 对这些列进行 z-score 标准化
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        # ============================标准化=====================================
        return data

    @staticmethod
    # 检查 self.fold_groups 中是否有重复的 ID
    def check_no_duplicates(fold_groups):
        seen_ids = set()
        for fold in fold_groups:
            for id_ in fold:
                if id_ in seen_ids:
                    raise ValueError(f"Duplicate ID found: {id_}")
                seen_ids.add(id_)
        print("No duplicate IDs found in fold_groups.")

    @staticmethod
    # 将嵌套的 fold_groups 摊平
    def flatten_fold_groups(fold_groups):
        return [id_ for fold in fold_groups for id_ in fold]

    @staticmethod
    # 检查摊平后的 fold_groups 是否全部元素属于 patient_ids
    def check_all_ids_in_patient_ids(flattened_fold_groups, patient_ids):
        patient_id_set = set(patient_ids)
        if len(patient_ids) != len(set(patient_id_set)):
            raise ValueError("patient_id_set中有重复的id")
        invalid_ids = [id_ for id_ in flattened_fold_groups if id_ not in patient_id_set]
        if invalid_ids:
            raise ValueError(f"The following IDs are not in patient_ids: {invalid_ids}")
        print("All IDs in fold_groups are in patient_ids.")

    # 返回病人为单位的数据
    def group_data(self, fold_groups):
        grouped = self.PD_data.groupby(['PatientID', 'Severity_Level'])
        # 每个病人特征数据为list中一个元素，每个元素是一个二维ndarray，存储该病人手工特征数据，例如一个元素shape为(24, 220)
        # 表示该二维ndarry中，该病人有24个滑窗（活动片段），每个滑窗对应220维特征
        # 每个病人的标签为list中一个元素，每个元素是一维ndarray元素，存储该病人滑动窗口的标签，例如一个元素shape为(24,)
        # 表示该一维ndarray中，该病人24个滑动窗口（活动片段），每个滑窗对应1个标签，有24个标签
        bag_data_dict = dict()
        patient_ids = []
        for (patient_id, _), group in grouped:
            # 检查 self.activity_id 是否可以转换为数字，并且是否在 1 到 16 之间
            try:
                activity_id_num = int(self.activity_id)
                if 1 <= activity_id_num <= 16:
                    bag_data = np.array(group.iloc[:, :-3])  # 获取特征数据
                else:
                    raise AssertionError(f"activity_id {self.activity_id} is not in the range 1 to 16.")
            except ValueError:
                bag_data = np.array(group.iloc[:, :-2])  # 获取特征数据
            bag_data_instance_label = (np.array(group.loc[:, 'Severity_Level']))  # 获取标签数据
            patient_ids.append(patient_id)
            if patient_id not in bag_data_dict.keys():
                bag_data_dict[patient_id] = {"pid": patient_id, "bag_data": bag_data,
                                             "bag_data_label": bag_data_instance_label[0],
                                             "bag_data_instance_label": bag_data_instance_label
                                             }
        # 运行检查
        self.check_no_duplicates(fold_groups)
        flattened_fold_groups = self.flatten_fold_groups(fold_groups)
        self.check_all_ids_in_patient_ids(flattened_fold_groups, patient_ids)
        for fold_num, flod_group in enumerate(fold_groups):
            print(f"第{fold_num}组， {len(flod_group)}人")
        print(
            f"总参与人数：{len(flattened_fold_groups)}人, activity_id:{self.activity_id}, 特征维度：{bag_data.shape[1]}")
        # return bag_data_dict, patient_ids
        return bag_data_dict, patient_ids, fold_groups

    # 训练和测试数据集
    def train_and_evaluate(self, classifier):
        total_pred_group_ls = []  # 记录预测的病人病情标签
        total_test_Y_group_ls = []  # 记录真实的病人病情标签
        params = None  # 记录当前classifier使用的参数
        id_records_for_each_fold_dict = dict()
        # 遍历bag_data_list, 按病人为预测单位进行留一验证
        # for selected_group in tqdm(range(0, len(self.bag_data_list))):
        # =============shap=========================
        # 初始化数组以聚合SHAP值
        # shap_values_aggregated = np.zeros(220)
        shap_values_aggregated = np.zeros((self.bag_data_dict[self.patient_ids[0]]["bag_data"].shape[1],
                                           len(set(self.severity_mapping.values()))))
        test_samples_num = 0
        shap_values_fold = list()  # 保存每一折的shap value
        new_test_X_fold = list()  # 保存每一折使用的测试，用于可视化shap value
        # =============shap=========================
        for fold_num, test_ids in enumerate(self.fold_groups):  # 遍历每个fold
            # train_X, train_Y, test_X, test_Y = self.create_train_test_split(selected_group)  # 获取当前留一验证的数据集
            train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids = self.create_train_test_split(fold_num,
                                                                                                       test_ids)  # 获取当前交叉验证的数据集
            # ========================标准化=======================
            # 创建 StandardScaler 实例
            dataset_scaler = StandardScaler()
            # 对训练数据 train_X 进行标准化，并同时拟合 (fit) 和转换 (transform)
            train_X = dataset_scaler.fit_transform(train_X)
            # 对测试数据 test_X_ls 进行标准化，使用已拟合的 scaler 对象进行转换
            for idx in range(len(test_X_ls)):
                test_X_ls[idx] = dataset_scaler.transform(test_X_ls[idx])
            # ======================================================
            if fold_num not in id_records_for_each_fold_dict:
                id_records_for_each_fold_dict[fold_num] = {'train_ids': train_ids, 'test_ids': test_ids}
            # 选择指定的classifier进行训练与测试
            if classifier == 'lgbm':
                lgb_train = lgb.Dataset(train_X, train_Y)  # 二维train_X，一维train_Y
                # ================================================
                new_test_Y_ls = []
                for bag_test_Y, bag_test_X in zip(test_Y_ls, test_X_ls):
                    new_test_Y_ls.append(np.full(bag_test_X.shape[0], bag_test_Y))
                new_test_X = np.vstack(test_X_ls)
                new_test_Y = np.hstack(new_test_Y_ls)
                # ================================================
                lgb_test = lgb.Dataset(new_test_X, new_test_Y, reference=lgb_train)  # 二维test_X，一维test_Y
                num_round = 500  # 训练的总轮数，即总共要训练多少棵树
                # Train the model
                _, cur_classifier_params = self.create_model(classifier)
                if params is None:
                    params = cur_classifier_params
                model = lgb.train(cur_classifier_params, lgb_train, num_boost_round=num_round,
                                  valid_sets=[lgb_train, lgb_test],
                                  callbacks=[early_stopping(stopping_rounds=50)])
                # ===================shap====================================================
                # 进行预测
                y_pred_ls = self.predict_most_likely_class(model, test_X_ls, classifier)

                # 使用SHAP解释模型预测
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(new_test_X)
                test_samples_num += new_test_X.shape[0]
                # 累积绝对SHAP值以计算特征重要性
                # shap_values_aggregated += np.abs(shap_values).sum(axis=0)
                # 特征可视化
                shap_values_fold.append(shap_values)
                new_test_X_fold.append(new_test_X)
                # ===================shap====================================================
            elif classifier == 'xgb':
                # Create DMatrix for XGBoost
                xgb_train = xgb.DMatrix(train_X, label=train_Y)
                xgb_test_X_ls = []
                # if len(test_X_ls) != len(test_Y_ls):
                # raise ValueError("Length of test_X_ls and test_Y_ls must be the same.")
                for test_X, test_Y in zip(test_X_ls, test_Y_ls):
                    test_Y = np.full(test_X.shape[0], test_Y)
                    # Check if test_X_ls and test_Y_ls have the same length
                    xgb_test = xgb.DMatrix(test_X, label=test_Y)
                    xgb_test_X_ls.append(xgb_test)
                test_X_ls = xgb_test_X_ls
                # Train the model
                num_round = 500  # 训练的总轮数，即总共要训练多少棵树
                early_stopping_rounds = 50
                # evallist = [(xgb_train, 'train')]
                evallist = [(xgb_train, 'train'), (xgb_test, 'test')]
                _, cur_classifier_params = self.create_model(classifier)
                if params is None:
                    params = cur_classifier_params
                model = xgb.train(cur_classifier_params, xgb_train, num_boost_round=num_round, evals=evallist,
                                  early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
            else:
                model, cur_classifier_params = self.create_model(classifier)
                if params is None:
                    params = cur_classifier_params
                model.fit(train_X, train_Y)
            # 使用训练好的模型，进行测试
            y_pred_ls = self.predict_most_likely_class(model, test_X_ls, classifier)
            # 将当前病人级别的预测标签和真实标签记录
            total_pred_group_ls.append(y_pred_ls)
            total_test_Y_group_ls.append(test_Y_ls)
            print('F1 score:', f1_score(test_Y_ls, y_pred_ls, zero_division=0, average='macro'))
        return shap_values_fold, new_test_X_fold

    # 制造五折交叉验证数据集
    def create_train_test_split(self, fold_num, test_ids):
        train_ids = []
        for num, fold_ids in enumerate(self.fold_groups):
            if num != fold_num:
                train_ids.extend(fold_ids)
        train_X = [self.bag_data_dict[pid]["bag_data"] for pid in train_ids]
        train_X = np.vstack(train_X)
        train_Y = [self.bag_data_dict[pid]["bag_data_instance_label"] for pid in train_ids]
        train_Y = np.hstack(train_Y)

        test_X_ls = [self.bag_data_dict[pid]["bag_data"] for pid in test_ids]
        test_Y_ls = [self.bag_data_dict[pid]["bag_data_label"] for pid in test_ids]

        return train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids

    # 创建模型和返回参数(lgb和xgb只返回参数，模型返回为None，需要自己手动创建模型在训练测试中)
    def create_model(self, classifier):
        if classifier == 'logistic_l1':

            logistic_l1_params = {
                'penalty': 'l1',  # 正则方式为l1
                'solver': 'saga',  # 'saga'  solver supports L1 regularization and is suitable for large datasets
                'C': 0.1,  # 正则化力度，C越小正则越强，越大正则越弱(重要调整)
                'random_state': 0,  # 随机种子
                'multi_class': 'multinomial',
                'max_iter': 50,  # 迭代次数(重要调整)
                'n_jobs': -1,  # 使用所有可用的CPU核心
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = logistic_l1_params
            model = make_pipeline(StandardScaler(), LogisticRegression(**logistic_l1_params))
        elif classifier == 'logistic_l2':
            # Logistic Regression with L2 regularization
            logistic_l2_params = {
                'penalty': 'l2',  # 正则方式为l2
                'solver': 'saga',  # 'saga' solver supports L1 regularization and is suitable for large datasets
                'C': 0.01,  # c是正则化参数的倒数，越小正则越强(重要调整)
                'random_state': 0,  # 随机种子
                'multi_class': 'multinomial',
                'max_iter': 50,  # 迭代次数(重要调整)
                'n_jobs': -1,  # 使用所有可用的CPU核心
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = logistic_l2_params
            model = make_pipeline(StandardScaler(), LogisticRegression(**logistic_l2_params))
        elif classifier == 'svm_l1':
            # Linear SVM with L1 regularization
            linear_svm_l1_params = {
                'penalty': 'l1',  # 正则方式为l1
                'loss': 'squared_hinge',  # 损失计算，尽量使得远离边界
                'dual': False,
                'C': 0.01,  # c是正则化参数的倒数，越小正则越强(重要调整)
                'random_state': 0,
                'max_iter': 1000,  # 迭代次数(重要调整)
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = linear_svm_l1_params
            model = make_pipeline(StandardScaler(), LinearSVC(**linear_svm_l1_params))
        elif classifier == 'svm_l2':
            # Linear SVM with L2 regularization
            linear_svm_l2_params = {
                'penalty': 'l2',  # 正则方式为l1
                'loss': 'squared_hinge',  # 损失计算，尽量使得远离边界
                'dual': True,
                'C': 0.01,  # c是正则化参数的倒数，越小正则越强(重要调整)
                'random_state': 0,
                'max_iter': 1000,  # 迭代次数(重要调整)
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = linear_svm_l2_params
            model = make_pipeline(StandardScaler(), LinearSVC(**linear_svm_l2_params))
        elif classifier == 'knn':
            knn_params = {
                'n_neighbors': 10,  # 邻居数量(重要调整)
                'weights': 'distance',  # 距离衡量
                'algorithm': 'auto',  # 距离计算算法，是否使用KTTree
                'n_jobs': -1,  # 使用所有可用的CPU核心
            }
            params = knn_params
            model = make_pipeline(StandardScaler(), KNeighborsClassifier(**knn_params))
        elif classifier == 'bayes':
            bayes_params = {
                'priors': None,  # 使用训练数据自动计算先验概率
                'var_smoothing': 1e-9  # 避免除以零错误的平滑参数
            }
            params = bayes_params
            model = make_pipeline(StandardScaler(), GaussianNB(**bayes_params))
        elif classifier == 'rf':
            rf_params = {
                'n_estimators': 500,  # 树的数量
                'max_depth': 5,
                'min_samples_split': 2,  # 只有当该节点包含至少两个样本时，才会进行分裂
                'min_samples_leaf': 1,  # 树中的叶子节点必须拥有的最小样本数量
                'max_features': 0.75,  # 分裂节点时考虑的随机特征的数量是总特征数量的 75%
                'bootstrap': True,  # 为 True 时，每棵树训练数据是通过从原始训练数据中进行有放回的抽样得到的，同一个数据点可能会被多次选中。
                'random_state': 0,
                'n_jobs': -1,  # 使用所有可用的CPU核心
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = rf_params
            model = make_pipeline(StandardScaler(), RandomForestClassifier(**rf_params))
        elif classifier == 'lgbm':
            lgb_params = {
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
                'num_class': len(set(self.severity_mapping.values())),
                'feature_fraction': .75,  # 每次选取百分之75的特征进行训练，控制过拟合
                'bagging_fraction': .75,  # 每次选取百分之85的数据进行训练，控制过拟合
                'seed': 0,
                'num_threads': -1,
                'verbose': -1,
                # 'early_stopping_rounds': 50,  # 当验证集在训练一百次过后准确率还没提升的时候停止ss
                'num_leaves': 128,
            }
            params = lgb_params
            model = None
        elif classifier == 'xgb':
            xgb_params = {
                'max_depth': 10,  # 树的最大深度
                'learning_rate': 0.05,  # 学习率
                # 'n_estimators': 200,  # 树的数量
                # 'gamma': 0.1,  # 最小分裂增益
                # 'min_child_weight': 1,  # 最小子权重
                'subsample': 0.75,  # 子采样比例
                'colsample_bytree': 0.75,  # 每棵树的特征采样比例
                'reg_alpha': 0.25,  # L1正则化
                'reg_lambda': 0.25,  # L2正则化
                'objective': 'multi:softprob',  # 多分类目标
                'eval_metric': 'mlogloss',  # 评估指标
                'num_class': len(set(self.severity_mapping.values())),  # 类别数量
                'seed': 0,  # 随机种子
                'nthread': 20,  # 线程数
            }

            params = xgb_params
            model = None
        elif classifier == 'mlp_2':
            mlp_2_params = {
                'hidden_layer_sizes': (128, 64),  # 隐藏层神经元数量
                'activation': 'relu',  # 激活函数
                'solver': 'adam',  # 优化算法
                'alpha': 0.0001,  # l2正则(重要调整)
                'batch_size': 'auto',  # 批量大小
                'learning_rate': 'constant',  # 学习率调度，表示是否学习率不变
                'learning_rate_init': 5e-4,  # 学习率大小(重要调整)
                'max_iter': 200,  # epoch(重要调整)
                'random_state': 2024,  # 随机种子
                'early_stopping': True,  # 早停
                'validation_fraction': 0.1,  # 使用10%的数据作为验证集
                'n_iter_no_change': 10,  # 若10个迭代不再提升，则提前停止
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = mlp_2_params
            model = make_pipeline(StandardScaler(), MLPClassifier(**mlp_2_params))
        elif classifier == 'mlp_4':
            mlp_4_params = {
                'hidden_layer_sizes': (64, 32, 16, 8),
                'activation': 'relu',  # 激活函数
                'solver': 'adam',  # 优化算法
                'alpha': 0.001,  # l2正则
                'batch_size': 'auto',  # 批量大小
                'learning_rate': 'constant',  # 学习率调度，表示是否学习率不变
                'learning_rate_init': 5e-4,  # 学习率大小
                'max_iter': 200,  # epoch
                'random_state': 2024,  # 随机种子
                'early_stopping': True,  # 早停
                'validation_fraction': 0.1,  # 使用10%的数据作为验证集
                'n_iter_no_change': 10,  # 若10个迭代不再提升，则提前停止
                # 'n_jobs': -1,  # 使用所有可用的CPU核心
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = mlp_4_params
            model = make_pipeline(StandardScaler(), MLPClassifier(**mlp_4_params))
        elif classifier == 'mlp_8':
            mlp_8_params = {
                'hidden_layer_sizes': (256, 128, 64, 64, 32, 32, 16, 8),
                'activation': 'relu',  # 激活函数
                'solver': 'adam',  # 优化算法
                'alpha': 0.001,  # l2正则
                'batch_size': 'auto',  # 批量大小
                'learning_rate': 'constant',  # 学习率调度，表示是否学习率不变
                'learning_rate_init': 5e-4,  # 学习率大小
                'max_iter': 200,  # epoch(重要调整)
                'random_state': 2024,  # 随机种子
                'early_stopping': True,  # 早停
                'validation_fraction': 0.1,  # 使用10%的数据作为验证集
                'n_iter_no_change': 10,  # 若10个迭代不再提升，则提前停止
                # 'n_jobs': -1,  # 使用所有可用的CPU核心
                'verbose': 1  # 打印训练过程，注释掉将无打印
            }
            params = mlp_8_params
            model = make_pipeline(StandardScaler(), MLPClassifier(**mlp_8_params))
        else:
            raise ValueError("Unsupported classifier type. Supported types are 'knn', 'mlp', and 'svm'.")

        return model, params

    def predict_most_likely_class(self, model, test_X_ls, classifier):
        y_pred_ls = []
        for test_X in test_X_ls:
            if classifier in ['logistic_l1', 'logistic_l2', 'knn', 'bayes', 'rf', 'mlp_2', 'mlp_4', 'mlp_8']:
                y_pred_prob = model.predict_proba(test_X)
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif classifier in ['svm_l1', 'svm_l2']:
                y_pred = model.predict(test_X)
            elif classifier == 'lgbm':
                y_pred_prob = model.predict(test_X, num_iteration=model.best_iteration)
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif classifier == 'xgb':
                y_pred_prob = model.predict(test_X, iteration_range=(0, model.best_iteration))
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:
                raise ValueError("Unsupported classifier type for prediction.")
            counts = np.bincount(y_pred)
            y_pred_ls.append(np.argmax(counts))
        return y_pred_ls


def single_activity_shap_importance(activity_id, model='lgbm'):
    classifier = PDClassifier(r"../../../output/activity/step_2_select_sensors/acc_data.csv", activity_id,
                              r'../../../input/activity/step_3_output_feature_importance'
                              r'/fold_groups_new_with_combinations.csv')  # 初始化PDClassifier分类器
    shap_values_fold, new_test_X_fold = classifier.train_and_evaluate(classifier=model)  # 选择对应的model
    shap_values_all_fold = np.vstack(shap_values_fold)  # 堆叠
    # 可视化
    shap_values_mean = np.mean(shap_values_all_fold, axis=2)  # 计算所有类的shap均值
    shap.summary_plot(shap_values_mean, np.vstack(new_test_X_fold), feature_names=classifier.feature_name)

    # 获取类数
    num_columns = shap_values_all_fold.shape[2]
    # 生成列名
    column_names = [f"class_{i}" for i in range(num_columns)]
    # 取abs,消除正负影响
    shap_values_class_mean = np.abs(shap_values_all_fold).mean(axis=0)
    shap_summary = pd.DataFrame(shap_values_class_mean, columns=column_names)
    shap_summary['shap_values_mean'] = shap_values_class_mean.mean(axis=1)
    # 将特征名称作为第一列
    shap_summary.insert(0, 'feature name', classifier.feature_name)
    shap_summary = shap_summary.sort_values(by='shap_values_mean', ascending=False)
    # 将SHAP值保存到CSV文件
    shap_summary.to_csv(os.path.join(r'../../../output/activity/step_3_output_feature_importance',
                                     f'{os.path.basename(classifier.data_path)}_{classifier.activity_id}_shap_importance.csv'),
                        index=False)

    return


if __name__ == '__main__':
    # 测试单个活动
    single_activity_shap_importance(1, 'lgbm')

    # # 测试多个活动
    # for activity_id in range(1, 16 + 1):
    #     single_activity_shap_importance(activity_id, 'lgbm')
