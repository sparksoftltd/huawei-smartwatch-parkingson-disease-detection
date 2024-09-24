import yaml
import os
from src.utils.PDDataLoader import PDDataLoader
from src.utils.ModelTrainerEvaluator import ModelTrainer, ModelEvaluator
import pandas as pd
from datetime import datetime
from typing import List
from src.utils.utils import set_seed
import copy

os.environ["OMP_NUM_THREADS"] = "1"  # 控制 OpenMP 的线程数
os.environ["MKL_NUM_THREADS"] = "1"  # 控制 MKL（Math Kernel Library）的线程数


class SeverityAssessment:
    def __init__(self, back_to_root: str, data_path: str, data_name: str, activity_id: List, classifier: str, **kwargs):
        self.back_to_root = back_to_root
        self.activity_id = activity_id
        self.classifier = classifier
        self.data_path = data_path
        self.data_name = data_name
        self.fold_groups_path = "input/feature_extraction"
        self.fold_groups_name = "fold_groups_new_with_combinations.csv"
        self.shap_importance = False
        self.data_loader = PDDataLoader(activity_id, os.path.join(self.back_to_root, self.data_path, self.data_name),
                                        os.path.join(self.back_to_root, self.fold_groups_path, self.fold_groups_name))
        self.single_activity = len(activity_id) == 1
        self.roc = kwargs.get('roc', False)
        self.model_evaluator = None
        set_seed(0)

    def load_config(self):
        if self.single_activity:
            config_path = f'src/utils/config/activity_{self.activity_id[0]}.yaml'
        else:
            config_path = f'src/utils/config/comb_{len(self.activity_id)}.yaml'
        with open(os.path.join(self.back_to_root, config_path), 'r') as file:
            return yaml.safe_load(file)

    def assessment(self):
        # loading config
        config = self.load_config()
        # print(type(config['mlp_2']['params']['alpha']))
        if self.single_activity:
            assert self.activity_id[0] == config['activity_id'], "error activity_id module"
        else:
            assert 'comb_' + str(len(self.activity_id)) == config['activity_id'], "error activity_id module"
        # loading hyperparameters
        params = config[self.classifier]['params']
        print(params)
        # severity assessment
        print(f"classifier [{self.classifier}] on activity_id {config['activity_id']}")
        model_trainer = ModelTrainer(self.classifier, params)
        study = ModelEvaluator(self.data_loader, model_trainer, roc=self.roc)
        metrics = study.train_evaluate()
        if self.shap_importance:
            study.shap_importance(self.back_to_root)

        self.model_evaluator = copy.deepcopy(study)
        return metrics


def get_roc_curve(severity_assessment: SeverityAssessment):
    severity_assessment.roc = True
    severity_assessment.assessment()
    return severity_assessment.model_evaluator.roc_result


def get_roc_curve_class(severity_assessment: SeverityAssessment):
    severity_assessment.roc = True
    severity_assessment.assessment()
    return severity_assessment.model_evaluator.roc_class_result


def get_confusion_matrices(severity_assessment: SeverityAssessment):
    metrics = severity_assessment.assessment()
    return metrics.get('mean_confusion_matrix', False)


def show_activity_shap_importance(severity_assessment: SeverityAssessment):
    severity_assessment.shap_importance = True
    severity_assessment.assessment()


def save_assessment_result(back_to_root: str, activity_list: list, classifier_list: list):
    data_path = "output/feature_selection"
    # 用于存储所有结果的列表
    results = []
    # 迭代所有活动 ID 和分类器组合
    for c in classifier_list:
        for a in activity_list:
            # 创建 SeverityAssessment 实例并进行评估
            data_name = f"activity_{a}.csv"
            sa = SeverityAssessment(back_to_root, data_path, data_name, [a], str(c))
            metrics = sa.assessment()

            # 将评估结果格式化为 DataFrame 行
            row = {
                'activity_id': a,
                'classifier': c,
                'acc_mean': metrics['mean_accuracy'],
                'precision_mean': metrics['mean_precision'],
                'recall_mean': metrics['mean_recall'],
                'f1_mean': metrics['mean_f1'],
                'specificity_mean': metrics['mean_specificity'],
            }

            # 添加每一折的结果
            for fold_metric in metrics['fold_metrics']:
                fold_num = fold_metric['fold_num']
                row[f'acc_fold_{fold_num}'] = fold_metric['accuracy']
                row[f'precision_fold_{fold_num}'] = fold_metric['precision']
                row[f'recall_fold_{fold_num}'] = fold_metric['recall']
                row[f'f1_fold_{fold_num}'] = fold_metric['f1']
                row[f'specificity_fold_{fold_num}'] = fold_metric['specificity']

            # 将当前行添加到结果列表中
            results.append(row)
    # 保存到 CSV 文件
    save_data_path = "output/severity_assessment"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(back_to_root, save_data_path,
                                   f'activity_classifier_metrics_{current_time}.csv'), index=False)
    print(f"All results have been saved to 'activity_classifier_metrics_{current_time}.csv'")


def save_comb_activity_assessment_result(back_to_root: str, activity_list: list, classifier_list: list,
                                         combination_mode: str):
    data_path = "output/activity_combination"
    # 用于存储所有结果的列表
    results = []
    # 迭代所有活动 ID 和分类器组合
    for c in classifier_list:
        for a in activity_list:
            # 创建 SeverityAssessment 实例并进行评估
            assert isinstance(c, str), "error classifier type"
            assert isinstance(a, List), "combination should be a list type"
            activity_ids_str = "_".join(map(str, a))
            file_name = f"merged_activities_{activity_ids_str}_{combination_mode}.csv"
            comb_sa = SeverityAssessment(back_to_root, data_path, file_name, a, str(c))
            comb_metrics = comb_sa.assessment()

            # 将评估结果格式化为 DataFrame 行
            row = {
                'activity_id': a,
                'classifier': c,
                'acc_mean': comb_metrics['mean_accuracy'],
                'precision_mean': comb_metrics['mean_precision'],
                'recall_mean': comb_metrics['mean_recall'],
                'f1_mean': comb_metrics['mean_f1'],
                'specificity_mean': comb_metrics['mean_specificity'],
            }

            # 添加每一折的结果
            for fold_metric in comb_metrics['fold_metrics']:
                fold_num = fold_metric['fold_num']
                row[f'acc_fold_{fold_num}'] = fold_metric['accuracy']
                row[f'precision_fold_{fold_num}'] = fold_metric['precision']
                row[f'recall_fold_{fold_num}'] = fold_metric['recall']
                row[f'f1_fold_{fold_num}'] = fold_metric['f1']
                row[f'specificity_fold_{fold_num}'] = fold_metric['specificity']

            # 将当前行添加到结果列表中
            results.append(row)
    # 保存到 CSV 文件
    save_data_path = "output/severity_assessment"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(back_to_root, save_data_path,
                                   f'activity_combination_{combination_mode}_classifier_metrics_{current_time}.csv'),
                      index=False)
    print(
        f"All results have been saved to 'activity_combination_{combination_mode}classifier_metrics_{current_time}.csv'")


if __name__ == '__main__':
    _back_to_root = "../../.."
    # # 单一活动测试
    activity_id = [1]
    _data_path = "output/feature_selection"
    _data_name = f"activity_{activity_id[0]}.csv"
    sa = SeverityAssessment(_back_to_root, _data_path, _data_name, activity_id, 'xgb')
    sa.assessment()

    # 全活动全算法测试
    # classifiers = ['rf', 'xgb', 'lgbm', 'logistic_l1', 'logistic_l2', 'svm_l1', 'svm_l2', 'knn', 'bayes', 'mlp_2',
    #                'mlp_4', 'mlp_8']
    # activity_ids = list(range(1, 17))
    # save_assessment_result(activity_ids, classifiers)

    # # 可视化展示
    # show_activity_shap_importance(sa)

    # 多活动测试
    # _activity_id = [14, 15, 16]
    # combination_mode = 'horizontal'
    # _data_path = "output/activity_combination"
    # activity_ids_str = "_".join(map(str, _activity_id))
    # _file_name = f"merged_activities_{activity_ids_str}_{combination_mode}.csv"
    # sa = SeverityAssessment(_back_to_root, _data_path, _file_name, _activity_id, 'mlp_2')
    # sa.assessment()

    # 多活动多算法测试
    # classifiers = ['xgb', 'lgbm', 'mlp_2']
    # activity_ids = [[1, 2, 3], [14, 15, 16]]
    # save_comb_activity_assessment_result(activity_ids, classifiers, 'horizontal')

__all__ = [
    'SeverityAssessment',
    'show_activity_shap_importance',
    'save_assessment_result',
    'save_comb_activity_assessment_result',
    'get_roc_curve',
    'get_roc_curve_class',
    'get_confusion_matrices'
]
