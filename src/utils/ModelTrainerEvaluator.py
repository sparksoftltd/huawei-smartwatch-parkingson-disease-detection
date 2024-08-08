import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import xgboost as xgb
from src.utils.PDDataLoader import PDDataLoader
from src.utils import set_seed
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, \
    f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import warnings
from imblearn.over_sampling import RandomOverSampler
warnings.filterwarnings(
    "ignore", category=UserWarning, module="lightgbm.engine", lineno=172
)
seed = set_seed(0)


class ModelTrainer:
    def __init__(self, classifier, params):
        self.classifier = classifier
        self.params = params
        self.model = None
        self.create_model()

    def create_model(self):
        if self.classifier == 'logistic_l1':
            self.params = {
                'penalty': 'l1',
                'solver': 'sag',
                'C': 100,
                'random_state': seed,
                'multi_class': 'multinomial',
                'max_iter': 50,
                'n_jobs': -1,
                'verbose': 0
            }
            self.model = make_pipeline(StandardScaler(), LogisticRegression(**self.params))
        elif self.classifier == 'logistic_l2':
            self.params = {
                'penalty': 'l2',
                'solver': 'saga',
                'C': 1,
                'random_state': seed,
                'multi_class': 'multinomial',
                'max_iter': 50,
                'n_jobs': -1,
                'verbose': 0
            }
            self.model = make_pipeline(StandardScaler(), LogisticRegression(**self.params))
        elif self.classifier == 'svm_l1':
            self.params = {
                'penalty': 'l1',
                'loss': 'squared_hinge',
                'dual': False,
                'C': 0.01,
                'random_state': 0,
                'max_iter': 1000,
                'verbose': 1
            }
            self.model = make_pipeline(StandardScaler(), LinearSVC(**self.params))
        elif self.classifier == 'svm_l2':
            self.params = {
                'penalty': 'l2',
                'loss': 'squared_hinge',
                'dual': True,
                'C': 0.01,
                'random_state': 0,
                'max_iter': 1000,
                'verbose': 1
            }
            self.model = make_pipeline(StandardScaler(), LinearSVC(**self.params))
        elif self.classifier == 'knn':
            self.params = {
                'n_neighbors': 10,
                'weights': 'distance',
                'algorithm': 'auto',
                'n_jobs': -1
            }
            self.model = make_pipeline(StandardScaler(), KNeighborsClassifier(**self.params))
        elif self.classifier == 'bayes':
            self.params = {
                'priors': None,
                'var_smoothing': 1e-9
            }
            self.model = make_pipeline(StandardScaler(), GaussianNB(**self.params))
        elif self.classifier == 'rf':
            self.model = make_pipeline(StandardScaler(), RandomForestClassifier(**self.params))
        elif self.classifier == 'lgbm':
            pass
        elif self.classifier == 'xgb':
            pass
        elif self.classifier == 'mlp_2':
            self.params = {
                'hidden_layer_sizes': (128, 64),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'batch_size': 'auto',
                'learning_rate': 'constant',
                'learning_rate_init': 5e-4,
                'max_iter': 200,
                'random_state': 2024,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,
                'verbose': 1
            }
            self.model = make_pipeline(StandardScaler(), MLPClassifier(**self.params))
        elif self.classifier == 'mlp_4':
            self.params = {
                'hidden_layer_sizes': (64, 32, 16, 8),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'batch_size': 'auto',
                'learning_rate': 'constant',
                'learning_rate_init': 5e-4,
                'max_iter': 200,
                'random_state': 2024,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,
                'verbose': 1
            }
            self.model = make_pipeline(StandardScaler(), MLPClassifier(**self.params))
        elif self.classifier == 'mlp_8':
            self.params = {
                'hidden_layer_sizes': (256, 128, 64, 64, 32, 32, 16, 8),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'batch_size': 'auto',
                'learning_rate': 'constant',
                'learning_rate_init': 5e-4,
                'max_iter': 200,
                'random_state': 2024,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,
                'verbose': 1
            }
            self.model = make_pipeline(StandardScaler(), MLPClassifier(**self.params))
        else:
            raise ValueError("Unsupported classifier type.")


class ModelEvaluator:
    def __init__(self, data_loader, model_trainer):
        self.data_loader = data_loader
        self.model_trainer = model_trainer
        self.classifier = model_trainer.classifier
        self.model = model_trainer.model
        print("Training...")

    def train_evaluate(self):
        total_pred_group_ls = []
        total_test_Y_group_ls = []
        id_records_for_each_fold_dict = dict()
        for fold_num, test_ids in enumerate(self.data_loader.fold_groups):
            # 创建一个 RandomOverSampler 实例以验证安装成功
            ros = RandomOverSampler(random_state=0)
            train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids = self.data_loader.create_train_test_split(
                fold_num, test_ids)
            train_X, train_Y = ros.fit_resample(train_X, train_Y)
            if fold_num not in id_records_for_each_fold_dict:
                id_records_for_each_fold_dict[fold_num] = {'train_ids': train_ids, 'test_ids': test_ids}

            if self.classifier == 'lgbm':
                dataset_scaler = StandardScaler()
                train_X = dataset_scaler.fit_transform(train_X)
                for idx in range(len(test_X_ls)):
                    test_X_ls[idx] = dataset_scaler.transform(test_X_ls[idx])
                lgb_train = lgb.Dataset(train_X, train_Y)
                new_test_Y_ls = [np.full(bag_test_X.shape[0], bag_test_Y) for bag_test_Y, bag_test_X in
                                 zip(test_Y_ls, test_X_ls)]
                new_test_X = np.vstack(test_X_ls)
                new_test_Y = np.hstack(new_test_Y_ls)
                lgb_test = lgb.Dataset(new_test_X, new_test_Y, reference=lgb_train)

                self.model = lgb.train(
                    self.model_trainer.params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
                )
                y_pred = self.model.predict(new_test_X, num_iteration=self.model.best_iteration)
                y_pred_labels = y_pred.argmax(axis=1)
                f1 = f1_score(new_test_Y, y_pred_labels, zero_division=0, average='macro')
                print(f1)
            elif self.classifier == 'xgb':
                dataset_scaler = StandardScaler()
                train_X = dataset_scaler.fit_transform(train_X)
                for idx in range(len(test_X_ls)):
                    test_X_ls[idx] = dataset_scaler.transform(test_X_ls[idx])
                xgb_train = xgb.DMatrix(train_X, label=train_Y)
                xgb_test_X_ls = [xgb.DMatrix(test_X, label=np.full(test_X.shape[0], test_Y)) for test_X, test_Y in
                                 zip(test_X_ls, test_Y_ls)]
                evallist = [(xgb_train, 'train'), (xgb_test_X_ls[0], 'test')]
                self.model = xgb.train(self.model_trainer.params, xgb_train,
                                       evals=evallist,
                                       early_stopping_rounds=50, verbose_eval=False)
                test_X_ls = xgb_test_X_ls
            else:
                self.model.fit(train_X, train_Y)
            y_pred_ls = self.predict_most_likely_class(test_X_ls)
            # 将当前留一验证的病人级别的预测标签和真实标签记录
            total_pred_group_ls.append(y_pred_ls)
            total_test_Y_group_ls.append(test_Y_ls)

        # 指标估计
        metrics = ModelEvaluator._metrics_calculation(total_test_Y_group_ls, total_pred_group_ls)

        # 打印返回的各个值
        print("Mean Accuracy:", metrics['mean_accuracy'])
        print("Mean Precision:", metrics['mean_precision'])
        print("Mean Recall:", metrics['mean_recall'])
        print("Mean F1 Score:", metrics['mean_f1'])
        print("Mean Specificity:", metrics['mean_specificity'])
        print("Mean Report:\n", metrics['mean_report'])
        return metrics

    def predict_most_likely_class(self, test_X_ls):
        y_pred_ls = []
        for test_X in test_X_ls:
            if self.classifier in ['logistic_l1', 'logistic_l2', 'knn', 'bayes', 'rf', 'mlp_2', 'mlp_4', 'mlp_8']:
                y_pred_prob = self.model.predict_proba(test_X)
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif self.classifier in ['svm_l1', 'svm_l2']:
                y_pred = self.model.predict(test_X)
            elif self.classifier == 'lgbm':
                y_pred_prob = self.model.predict(test_X, num_iteration=self.model.best_iteration)
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif self.classifier == 'xgb':
                y_pred_prob = self.model.predict(test_X, iteration_range=(0, self.model.best_iteration))
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:
                raise ValueError("Unsupported classifier type for prediction.")
            counts = np.bincount(y_pred)
            y_pred_ls.append(np.argmax(counts))
        return y_pred_ls

    @staticmethod
    def _calculate_specificity(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn = cm[0, 0]
        fp = cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return specificity

    def _metrics_calculation(total_test_Y_group_ls, total_pred_group_ls):
        print("Evaluating...")
        total_acc_ls = []
        total_precision_ls = []
        total_recall_ls = []
        total_f1_ls = []
        total_specificity_ls = []
        report_ls = []

        for fold_num, (total_test_Y, total_pred) in enumerate(zip(total_test_Y_group_ls, total_pred_group_ls)):
            total_acc = accuracy_score(total_test_Y, total_pred)
            total_precision = precision_score(total_test_Y, total_pred, zero_division=0, average='macro')
            total_recall = recall_score(total_test_Y, total_pred, zero_division=0, average='macro')
            total_f1 = f1_score(total_test_Y, total_pred, zero_division=0, average='macro')
            report = classification_report(total_test_Y, total_pred, zero_division=0)

            # 调用 calculate_specificity 静态方法计算特异性
            specificity = ModelEvaluator._calculate_specificity(total_test_Y, total_pred)

            total_acc_ls.append(total_acc)
            total_precision_ls.append(total_precision)
            total_recall_ls.append(total_recall)
            total_f1_ls.append(total_f1)
            total_specificity_ls.append(specificity)
            report_ls.append(report)

        mean_acc = round(np.mean(total_acc_ls), 4)
        mean_precision = round(np.mean(total_precision_ls), 4)
        mean_recall = round(np.mean(total_recall_ls), 4)
        mean_f1 = round(np.mean(total_f1_ls), 4)
        mean_specificity = round(np.mean(total_specificity_ls), 4)
        all_total_test_Y = [item for sublist in total_test_Y_group_ls for item in sublist]
        all_total_pred = [item for sublist in total_pred_group_ls for item in sublist]
        all_report = classification_report(all_total_test_Y, all_total_pred, zero_division=0)

        return {
            'mean_accuracy': mean_acc,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mean_specificity': mean_specificity,
            'mean_report': all_report
        }




if __name__ == '__main__':
    activity_id = 3
    data_path = "../../output/activity/step_2_select_sensors"
    data_name = "acc_data.csv"
    fold_groups_path = "../../input/activity/step_3_output_feature_importance"
    fold_groups_name = "fold_groups_new_with_combinations.csv"
    severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

    data_loader = PDDataLoader(activity_id, os.path.join(data_path, data_name),
                               os.path.join(fold_groups_path, fold_groups_name), severity_mapping=severity_mapping)
    model_trainer = ModelTrainer('logistic_l2', 4)
    study = ModelEvaluator(data_loader, model_trainer)
    study.train_evaluate()
