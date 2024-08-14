import yaml
import os
from src.utils.PDDataLoader import PDDataLoader
from src.utils.ModelTrainerEvaluator import ModelTrainer, ModelEvaluator
import pandas as pd
from datetime import datetime


class SeverityAssessment:
    def __init__(self, activity_id, classifier):
        self.activity_id = activity_id
        self.classifier = classifier
        self.data_path = "../../../output/activity/step_4_feature_selection"
        self.data_name = f"acc_data_activity_{activity_id}.csv"
        self.fold_groups_path = "../../../input/activity/step_3_output_feature_importance"
        self.fold_groups_name = "fold_groups_new_with_combinations.csv"
        self.severity_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
        self.data_loader = PDDataLoader(activity_id, os.path.join(self.data_path, self.data_name),
                                        os.path.join(self.fold_groups_path, self.fold_groups_name),
                                        self.severity_mapping)
        self.shap_importance = False

    def load_config(self):
        config_path = f'../../utils/config/activity_{self.activity_id}.yaml'
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def assessment(self):
        # loading config
        config = self.load_config()
        assert self.activity_id == config['activity_id'], "error activity parameters"
        # loading hyperparameters
        params = config[self.classifier]['params']
        print(params)
        # severity assessment
        print(f'classifier [{self.classifier}] on activity [{self.activity_id}]')
        model_trainer = ModelTrainer(self.classifier, params)
        study = ModelEvaluator(self.data_loader, model_trainer)
        metrics = study.train_evaluate()
        if self.shap_importance:
            study.single_activity_shap_importance()
        return metrics


def show_activity_shap_importance(activity_id):
    sa = SeverityAssessment(activity_id, 'lgbm')
    sa.shap_importance = True
    sa.assessment()


def save_assessment_result(activity_list: list, classifier_list: list):
    # 用于存储所有结果的列表
    results = []
    # 迭代所有活动 ID 和分类器组合
    for c in classifier_list:
        for a in activity_list:
            # 创建 SeverityAssessment 实例并进行评估
            sa = SeverityAssessment(a, c)
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
    save_data_path = "../../../output/activity/step_5_five_fold_cross_validation"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_data_path, f'activity_classifier_metrics_{current_time}.csv'), index=False)
    print("All results have been saved to 'activity_classifier_metrics.csv'")


if __name__ == '__main__':
    # 单一活动测试
    sa = SeverityAssessment([3], 'lgbm')
    sa.assessment()

    # 全活动全算法测试
    # classifiers = ['rf', 'xgb', 'lgbm', 'logistic_l1', 'logistic_l2', 'svm_l1', 'svm_l2', 'knn', 'bayes', 'mlp_2',
    #                'mlp_4', 'mlp_8']
    # activity_ids = list(range(1, 17))
    # save_assessment_result(activity_ids, classifiers)

    # # 可视化展示
    # show_activity_shap_importance(3)

__all__ = [
    'SeverityAssessment',
    'show_activity_shap_importance',
    'save_assessment_result'
]
