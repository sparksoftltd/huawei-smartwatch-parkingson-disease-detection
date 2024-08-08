import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
import os
from src.utils.PDDataLoader import PDDataLoader
from src.utils.utils import set_seed, suggest_int, suggest_float
from src.utils.ModelTrainerEvaluator import ModelTrainer, ModelEvaluator


def model_params(trial, cfg: DictConfig, classifier):
    model_params_cfg = cfg.model.params
    params = None
    if classifier == 'lgbm':
        max_depth = suggest_int(trial, model_params_cfg, "max_depth")
        num_leaves_max = max(2 ** max_depth - 1, 2)
        params = {
            "objective": model_params_cfg["objective"],
            "num_class": model_params_cfg["num_class"],
            "metric": model_params_cfg["metric"],
            "verbosity": model_params_cfg["verbosity"],
            "boosting_type": model_params_cfg["boosting_type"],
            "num_threads": model_params_cfg["num_threads"],
            "seed": model_params_cfg["seed"],
            "n_estimators": model_params_cfg["n_estimators"],
            "learning_rate": suggest_float(trial, model_params_cfg, "learning_rate"),
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int("num_leaves", 2, num_leaves_max),
            "min_child_samples": suggest_int(trial, model_params_cfg, "min_child_samples"),
            "lambda_l1": suggest_float(trial, model_params_cfg, "lambda_l1"),
            "lambda_l2": suggest_float(trial, model_params_cfg, "lambda_l2"),
        }
    elif classifier == 'xgb':
        pass
    else:
        raise ValueError("Unsupported classifier parameters.")
    return params


def outer_objective(cfg: DictConfig, activity_id: int):
    def objective(trial):
        dataset_cfg = cfg.dataset.default
        data_loader = PDDataLoader(
            activity_id,
            os.path.join(dataset_cfg.data_path, f"acc_data_activity_{activity_id}.csv"),
            os.path.join(dataset_cfg.fold_groups_path, dataset_cfg.fold_groups_name),
            dataset_cfg.severity_mapping
        )
        classifier = 'lgbm'
        params = model_params(trial, cfg, classifier)
        model_trainer = ModelTrainer(classifier, params)
        study = ModelEvaluator(data_loader, model_trainer)
        metrics = study.train_evaluate()
        return metrics['mean_f1']

    return objective


def study(cfg: DictConfig, activity_id: int):
    print(f'\n{OmegaConf.to_yaml(cfg)}')
    set_seed(cfg.seed)
    expname = cfg.expname
    n_trials = cfg.optuna.n_trials

    study = optuna.create_study(direction=cfg.optuna.direction,
                                storage=cfg.optuna.storage,
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                                load_if_exists=True,
                                study_name='activity_id ' + str(activity_id) + ' ' + expname)

    study.set_user_attr('Completed', False)

    study.optimize(outer_objective(cfg, activity_id), n_trials=n_trials,
                   gc_after_trial=True, show_progress_bar=True)
    study.set_user_attr('Completed', True)

    print(f'best params: {study.best_params}')
    print(f'best value: {study.best_value}')
    print(study.trials_dataframe())
    print(f'{expname}')


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # study(cfg, 3)
    for a in range(1, 16 + 1):
        study(cfg, a)


if __name__ == "__main__":
    main()
