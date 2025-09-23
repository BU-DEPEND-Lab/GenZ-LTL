import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from envs import make_env, make_env_safety
from evaluation.eval_sync_env import EvalSyncEnv
from ltl import FixedSampler
from model.model import build_model, build_model_safety
from config import model_configs
from model.parallel_agent import ParallelAgent
from sequence.search import ExhaustiveSearch, ExhaustiveSearchSafety
from utils.model_store import ModelStore
import multiprocessing as mp

env = None

def set_env():
    global env
    sampler = FixedSampler.partial('this_will_be_overridden')
    envs = [make_env_safety(env_name, sampler, render_mode=None) for _ in range(8)]
    world_info_paths = []
    if env_name == "PointLtlSafety2-v0":
        new_env_name = "PointLtl2-v0" 
    if env_name == "LetterSafetyEnv-v0":
        new_env_name = "LetterEnv-v0"
    if os.path.exists(f'eval_datasets/{new_env_name}/worlds'):
        world_info_paths = [f'eval_datasets/{new_env_name}/worlds/world_info_{i}.pkl' for i in range(num_eval_episodes)]
    with open(f'eval_datasets/{new_env_name}/tasks.txt') as f:
        tasks = [line.strip() for line in f]
    env = EvalSyncEnv(envs, world_info_paths, tasks)


env_name = sys.argv[1]
# 'PointLtlSafety2-v0'
config = model_configs[env_name]
exp = 'GenZ-LTL'
seed = int(sys.argv[2])
print(f"env_name: {env_name}, seed: {seed}")
deterministic = True
gamma = 0.998
num_procs = 8
num_eval_episodes = 50
device = 'cpu'
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)


def aux(status):
    global env
    model = build_model_safety(env.envs[0], status, config)
    s, v, num_steps, adr = eval_model(model, env, num_eval_episodes, deterministic, gamma)
    return status['num_steps'], s, v, num_steps, adr


def main():
    start_time = time.time()
    model_store = ModelStore(env_name, exp, seed, None)
    statuses = model_store.load_eval_training_statuses(map_location=device)
    model_store.load_vocab()

    results = []
    with mp.Pool(num_procs, initializer=set_env) as pool:
        for r in tqdm(pool.imap_unordered(aux, statuses), total=len(statuses)):
            results.append(r)

    print(f'Total time: {time.time() - start_time:.2f}s')
    result = {r[0]: (r[1], r[2], r[3], r[4]) for r in results}

    df = pd.DataFrame.from_dict(result, orient='index', columns=['success_rate', 'violation_rate', 'average_steps', 'return'])
    df.sort_index(inplace=True)
    out_path = f'eval_results/{env_name}/{exp}'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    df.to_csv(f'{out_path}/{seed}.csv', index_label='num_steps')


def eval_model(model, env, num_eval_episodes, deterministic, gamma):
    props = env.envs[0].get_propositions()
    search = ExhaustiveSearchSafety(env.envs[0], model, props, num_loops=2)
    agent = ParallelAgent(env, model, search=search, propositions=props, num_envs=len(env.envs))
    num_successes = 0
    num_violations = 0
    steps = []
    obss = env.reset()
    infos = [{} for _ in range(len(obss))]
    finished_episodes = 0
    num_steps = [0] * len(env.envs)
    returns = []
    while finished_episodes < num_eval_episodes:
        action = agent.get_action(obss, infos, deterministic=deterministic)
        obss, rewards, dones, infos = env.step(action)

        for i, done in enumerate(dones):
            if done:
                finished_episodes += 1
                if 'success' in infos[i]:
                    num_successes += 1
                    steps.append(num_steps[i] + 1)
                    returns.append(pow(gamma, num_steps[i] + 1))
                elif 'violation' in infos[i]:
                    num_violations += 1
                    returns.append(0)
                else:
                    returns.append(0)
                num_steps[i] = 0
            else:
                num_steps[i] += 1

        obss = [obs for obs in obss if obs is not None]

    assert len(env.active_envs) == 0

    return num_successes / finished_episodes, num_violations / finished_episodes, np.mean(steps) if steps else -1, np.mean(returns) if returns else -1


if __name__ == '__main__':
    if device == 'cuda':
        mp.set_start_method('spawn')
    # elif device == 'cpu':
    #     torch.set_num_threads(1)
    main()
