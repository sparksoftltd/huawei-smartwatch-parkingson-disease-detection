import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
import sys
import os


from src.utils.PDDataLoader import PDDataLoader
from src.utils.utils import suggest_int, suggest_float, set_seed


def outer_objective(cfg: DictConfig):
    def objective(trial):
        dataset_cfg = cfg.dataset.default
        model_params_cfg = cfg.model.params
        data_loader = PDDataLoader(
            dataset_cfg.activity_id,
            os.path.join(dataset_cfg.data_path, dataset_cfg.data_name),
            os.path.join(dataset_cfg.fold_groups_path, dataset_cfg.fold_groups_name),
            dataset_cfg.severity_mapping
        )

        max_depth = suggest_int(trial, model_params_cfg, "max_depth")
        num_leaves_max = 2 ** max_depth

        param = {
            "objective": model_params_cfg.objective,
            "num_class": model_params_cfg.num_class,
            "metric": model_params_cfg.metric,
            "verbosity": model_params_cfg.verbosity,
            "boosting_type": model_params_cfg.boosting_type,
            # "max_depth": max_depth,
            # "num_leaves_max": num_leaves_max,
            # "learning_rate": suggest_float(trial, model_params_cfg, "learning_rate"),
            # "max_bin": suggest_int(trial, model_params_cfg, "max_bin"),
            # "lambda_l1": suggest_float(trial, model_params_cfg, "lambda_l1"),
            # "lambda_l2": suggest_float(trial, model_params_cfg, "lambda_l2"),
            "min_gain_to_split": suggest_float(trial, model_params_cfg, "min_gain_to_split"),
            "feature_fraction": suggest_float(trial, model_params_cfg, "feature_fraction"),
            "bagging_fraction": suggest_float(trial, model_params_cfg, "bagging_fraction"),
        }

        f1_scores = []

        for fold_num, test_ids in enumerate(data_loader.fold_groups):
            train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids = data_loader.create_train_test_split(fold_num, test_ids)

            lgb_train = lgb.Dataset(train_X, train_Y)
            new_test_X = np.vstack(test_X_ls)
            new_test_Y_ls = []
            for bag_test_Y, bag_test_X in zip(test_Y_ls, test_X_ls):
                new_test_Y_ls.append(np.full(bag_test_X.shape[0], bag_test_Y))
            new_test_Y = np.hstack(new_test_Y_ls)
            lgb_test = lgb.Dataset(new_test_X, new_test_Y, reference=lgb_train)

            model = lgb.train(
                param,
                lgb_train,
                valid_sets=[lgb_train, lgb_test],
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(100)]
            )
            y_pred = model.predict(new_test_X, num_iteration=model.best_iteration)
            y_pred_labels = y_pred.argmax(axis=1)
            f1 = f1_score(new_test_Y, y_pred_labels, zero_division=0, average='macro')
            f1_scores.append(f1)

        mean_f1_score = np.mean(f1_scores)
        return mean_f1_score

    return objective

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f'\n{OmegaConf.to_yaml(cfg)}')

    set_seed(cfg.seed)
    expname = cfg.expname
    n_trials = cfg.optuna.n_trials

    study = optuna.create_study(direction=cfg.optuna.direction,
                                storage=cfg.optuna.storage,
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                                load_if_exists=True,
                                study_name=expname)

    study.set_user_attr('Completed', False)

    study.optimize(outer_objective(cfg), n_trials=n_trials,
                   gc_after_trial=True, show_progress_bar=True)
    study.set_user_attr('Completed', True)

    print(f'best params: {study.best_params}')
    print(f'best value: {study.best_value}')
    print(study.trials_dataframe())
    print(f'{expname}')

if __name__ == "__main__":
    main()
