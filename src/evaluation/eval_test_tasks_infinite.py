import os

import pandas as pd
from tqdm import tqdm

from evaluation.simulate import simulate

import argparse
import multiprocessing as mp

env_to_tasks = {
    'PointLtlSafety2-v0': [
        'GF blue & GF green & G !(yellow | magenta)',
        'GF blue & GF yellow & GF green & G !magenta',
        'FG yellow & G !(green | blue | magenta)',
    ],
    'LetterSafetyEnv-v0': [
        'GF (e & (!a U f)) & G !(c | d)',
        'GF a & GF b & GF c & G !(e | f | i)',
        'GF c & GF a & GF (e & !f U g) & GF k & G !(i | j)',
    ],
}


def main(env, exp, seeds):
    num_episodes = 100
    tasks = env_to_tasks[env]
    gamma = {
        'PointLtl2-v0': 0.998,
        'PointLtlSafety2-v0': 0.998,
        'PointLtlSafety3-v0': 0.998,
        'PointLtlSafety4-v0': 0.998,
        'PointLtlSafety5-v0': 0.998,
        'LetterEnv-v0': 0.94,
        'LetterSafetyEnv-v0': 0.94,
    }[env]

    num_procs = len(seeds)
    with mp.Pool(num_procs) as pool:
        args = []
        for task in tasks:
            print(f'Running task: {task}')
            for seed in seeds:
                args.append([env, gamma, exp, seed, num_episodes, task, False, False, True])
        for result in tqdm(pool.imap_unordered(eval_task, args), total=len(seeds)):
            results = []
            (accepting_visits, vr), seed, task = result
            results.append([exp, task, seed, accepting_visits, vr])
            df = pd.DataFrame(results, columns=['method', 'task', 'seed', 'accepting_visits', 'violation_rate'])
            os.makedirs('results_infinite', exist_ok=True)

            file_path = f'results_infinite/{env}_{exp}.csv'
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(file_path, index=False)


def eval_task(simulate_args):
    return simulate(*simulate_args), simulate_args[3], simulate_args[5]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['PointLtlSafety2-v0', 
                                                    'PointLtlSafety3-v0', 
                                                    'PointLtlSafety4-v0', 
                                                    'PointLtlSafety5-v0',
                                                    'LetterSafetyEnv-v0'], default='PointLtlSafety2-v0')
    parser.add_argument('--exp', type=str, default='GenZ-LTL')
    parser.add_argument('--seed', type=int, nargs='+', default=[1], help="List of seeds")
    args = parser.parse_args()
    main(args.env, args.exp, args.seed)
