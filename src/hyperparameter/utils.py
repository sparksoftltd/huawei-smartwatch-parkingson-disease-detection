import os
import random
from typing import Callable, Union
from omegaconf import DictConfig
from optuna import Trial
import numpy as np

def suggest_int(trial: Trial, cfg: DictConfig, *route: str) -> int:
    return _suggest(trial.suggest_int, cfg, *route)

def suggest_float(trial: Trial, cfg: DictConfig, *route: str) -> float:
    return _suggest(trial.suggest_float, cfg, *route)

def _suggest(func: Callable, cfg: DictConfig, *route: str) -> Union[float, int]:
    d = cfg
    for p in route:
        d = d[p]
    name = route[-1]
    if isinstance(d, DictConfig):
        low = d['low']
        high = d['high']
        if 'log' in d and d['log']:
            v = func(name, low, high, log=True)
        elif 'step' in d:
            step = d['step']
            v = func(name, low, high, step=step)
        else:
            v = func(name, low, high)
    else:
        v = d
    print(f'Fetching name: {name}={v} from {"/".join(route)}', flush=True)
    return v

def set_seed(seed: int):
    print(f'Set seed={seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
