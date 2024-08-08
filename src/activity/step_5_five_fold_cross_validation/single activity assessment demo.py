import yaml
import os
from src.utils.PDDataLoader import PDDataLoader
from src.utils.ModelTrainerEvaluator import ModelTrainer, ModelEvaluator


def load_config(activity_id):
    config_path = f'config/activity_{activity_id}.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


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
                            os.path.join(self.fold_groups_path, self.fold_groups_name), self.severity_mapping)

    def assessment(self):
        # loading config
        config = load_config(self.activity_id)
        assert self.activity_id == config['activity_id'], "error activity parameters"
        # loading hyperparameters
        params = config[self.classifier]['params']
        # severity assessment
        print(f'classifier [{self.classifier}] on activity [{self.activity_id}]')
        model_trainer = ModelTrainer(self.classifier, params)
        study = ModelEvaluator(self.data_loader, model_trainer)
        study.train_evaluate()


if __name__ == '__main__':
    activity_id = 1
    classifier = 'lgbm'
    sa = SeverityAssessment(activity_id, classifier)
    sa.assessment()

