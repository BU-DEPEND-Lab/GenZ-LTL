import argparse
import os


def get_experiment_path(env: str, name: str, seed: int) -> str:
    if '.' in env:
        # '.' is used to indicate alternative versions of the environment, e.g. fixed letters in LetterEnv. This is
        # only used for evaluation, and thus the same models as for the original environment should be loaded.
        env = env.split('.')[0]
    path = f'experiments/rco/{env}/{name}/{seed}' # ppo or ppo-lag or p3o or rco
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    eval_path = f'{path}/eval'
    if not os.path.exists(eval_path):
        os.makedirs(eval_path, exist_ok=True)
    return path


# def get_eval_results_path(env: str, name: str, seed: int) -> str:
#     return f'eval_results/{env}/{name}/{seed}.csv'

def get_eval_results_path(env: str, name: str, seed: int) -> str:
    return f'experiments/{env}/{name}/{seed}/log.csv'


def get_experiment_path_from_config(config: argparse.Namespace) -> str:
    experiment = config.experiment
    return get_experiment_path(experiment.env, experiment.name, experiment.seed)


def get_pretraining_experiment_path(env: str, pretraining_experiment: str, seed: int) -> str:
    return f'experiments/ppo/pretraining_{env}/{pretraining_experiment}/{seed}'
