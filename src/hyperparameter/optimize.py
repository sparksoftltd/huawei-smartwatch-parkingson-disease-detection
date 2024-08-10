import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
import os
from src.utils.PDDataLoader import PDDataLoader
from src.utils.utils import set_seed, suggest_int, suggest_float
from src.utils.ModelTrainerEvaluator import ModelTrainer, ModelEvaluator
from optuna import Trial


def optimize_lgbm_params(trial, model_params_cfg):
    # Suggest parameters
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

    return params


def optimize_xgb_params(trial: Trial, model_params_cfg):
    # Suggest parameters based on YAML config
    params = {
        "objective": model_params_cfg["objective"],
        "num_class": model_params_cfg["num_class"],
        "eval_metric": model_params_cfg["eval_metric"],
        "verbosity": model_params_cfg["verbosity"],
        "seed": model_params_cfg["seed"],
        "eta": suggest_float(trial, model_params_cfg, "eta"),
        "max_depth": suggest_int(trial, model_params_cfg, "max_depth"),
        "min_child_weight": suggest_int(trial, model_params_cfg, "min_child_weights"),
        "subsample": suggest_float(trial, model_params_cfg, "subsample"),
        "gamma": suggest_float(trial, model_params_cfg, "gamma"),
        "lambda": suggest_float(trial, model_params_cfg, "lambda"),
        "alpha": suggest_float(trial, model_params_cfg, "alpha"),
    }

    return params


def optimize_rf_params(trial, model_params_cfg):
    params = {
        "bootstrap": model_params_cfg["bootstrap"],
        "random_state": model_params_cfg["random_state"],
        "n_jobs": model_params_cfg["n_jobs"],
        "verbose": model_params_cfg["verbose"],
        "n_estimators": suggest_int(trial, model_params_cfg, "n_estimators"),
        "max_depth": suggest_int(trial, model_params_cfg, "max_depth"),
        "min_samples_split": suggest_int(trial, model_params_cfg, "min_samples_split"),
        "max_features": suggest_float(trial, model_params_cfg, "max_features"),
    }

    return params


def optimize_mlp_8_params(trial: Trial, model_params_cfg):
    params = {
        "early_stopping": model_params_cfg["early_stopping"],
        "random_state": model_params_cfg["random_state"],
        "verbose": model_params_cfg["verbose"],
        "alpha": suggest_float(trial, model_params_cfg, "alpha"),
        "learning_rate_init": suggest_float(trial, model_params_cfg, "learning_rate_init"),
        "max_iter": suggest_int(trial, model_params_cfg, "max_iter"),
    }

    return params


def optimize_mlp_4_params(trial: Trial, model_params_cfg):
    params = {
        "early_stopping": model_params_cfg["early_stopping"],
        "random_state": model_params_cfg["random_state"],
        "verbose": model_params_cfg["verbose"],
        "alpha": suggest_float(trial, model_params_cfg, "alpha"),
        "learning_rate_init": suggest_float(trial, model_params_cfg, "learning_rate_init"),
        "max_iter": suggest_int(trial, model_params_cfg, "max_iter"),
    }

    return params


def optimize_mlp_2_params(trial: Trial, model_params_cfg):
    params = {
        "early_stopping": model_params_cfg["early_stopping"],
        "random_state": model_params_cfg["random_state"],
        "verbose": model_params_cfg["verbose"],
        "alpha": suggest_float(trial, model_params_cfg, "alpha"),
        "learning_rate_init": suggest_float(trial, model_params_cfg, "learning_rate_init"),
        "max_iter": suggest_int(trial, model_params_cfg, "max_iter"),
    }

    return params


def model_params(trial, cfg: DictConfig, classifier):
    assert cfg.classifier.name == classifier, "missing classifier!"
    model_params_cfg = cfg.classifier.params
    if classifier == 'lgbm':
        params = optimize_lgbm_params(trial, model_params_cfg)
    elif classifier == 'xgb':
        params = optimize_xgb_params(trial, model_params_cfg)
    elif classifier == 'rf':
        params = optimize_rf_params(trial, model_params_cfg)
    elif classifier == 'mlp_8':
        params = optimize_mlp_8_params(trial, model_params_cfg)
    elif classifier == 'mlp_4':
        params = optimize_mlp_4_params(trial, model_params_cfg)
    elif classifier == 'mlp_2':
        params = optimize_mlp_2_params(trial, model_params_cfg)
    else:
        raise ValueError("Unsupported classifier parameters.")
    return params


def outer_objective(cfg: DictConfig, activity_id: int, classifier: str):
    def objective(trial):
        dataset_cfg = cfg.dataset.default
        data_loader = PDDataLoader(
            activity_id,
            os.path.join(dataset_cfg.data_path, f"acc_data_activity_{activity_id}.csv"),
            os.path.join(dataset_cfg.fold_groups_path, dataset_cfg.fold_groups_name),
            dataset_cfg.severity_mapping
        )
        params = model_params(trial, cfg, classifier)
        model_trainer = ModelTrainer(classifier, params)
        study = ModelEvaluator(data_loader, model_trainer)
        metrics = study.train_evaluate()
        return metrics['mean_f1']

    return objective


def study(cfg: DictConfig, activity_id: int, classifier: str):
    print(f'\n{OmegaConf.to_yaml(cfg)}')
    set_seed(cfg.seed)
    expname = cfg.expname
    n_trials = cfg.optuna.n_trials
    assert cfg.classifier.name == classifier, "missing match classifier"

    study = optuna.create_study(direction=cfg.optuna.direction,
                                storage=cfg.optuna.storage,
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                                load_if_exists=True,
                                study_name='activity_id ' + str(
                                    activity_id) + ' classifier ' + classifier + " " + expname)

    study.set_user_attr('Completed', False)

    study.optimize(outer_objective(cfg, activity_id, classifier), n_trials=n_trials,
                   gc_after_trial=True, show_progress_bar=True)
    study.set_user_attr('Completed', True)

    print(f'best params: {study.best_params}')
    print(f'best value: {study.best_value}')
    print(study.trials_dataframe())
    print(f'{expname}')


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # study(cfg, 11, 'xgb')

    for a in range(1, 16 + 1):
        study(cfg, a, 'mlp_8')


if __name__ == "__main__":
    main()
