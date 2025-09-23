import random

import numpy as np
import torch
from tqdm import tqdm

from envs import make_env, make_env_safety
from ltl import FixedSampler
from model.model import build_model, build_model_safety
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch, ExhaustiveSearchSafety, NoPathsException
from utils.model_store import ModelStore
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['PointLtl2-v0', 'PointLtlSafety2-v0', 'LetterEnv-v0', 'FlatWorld-v0'], default='PointLtl2-v0')
    parser.add_argument('--exp', type=str, default='deepset')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--formula', type=str, default='(F blue) & (!blue U (green & F yellow))')  # !(red | green) U magenta  !blue U (magenta & red)
    parser.add_argument('--finite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    gamma = 0.94 if args.env == 'LetterEnv-v0' else 0.998 if args.env == 'PointLtl2-v0' or args.env == 'PointLtlSafety2-v0' else 0.98
    return simulate(args.env, gamma, args.exp, args.seed, args.num_episodes, args.formula, args.finite, args.render, args.deterministic)


def simulate(env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic=True):
    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    sampler = FixedSampler.partial(formula)
    max_steps = None
    env = make_env_safety(env_name, sampler, max_steps, render_mode='human' if render else None) if "Safety" in env_name \
        else make_env(env_name, sampler, render_mode='human' if render else None)
    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp, seed, None)
    training_status = model_store.load_training_status(map_location='cpu')
    model = build_model_safety(env, training_status, config) if "Safety" in env_name \
        else build_model(env, training_status, config)
    props = env.get_propositions()
    search = ExhaustiveSearchSafety(env, model, props, num_loops=2) if "Safety" in env_name \
        else ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(env, model, search=search, propositions=props, verbose=render)

    num_successes = 0
    num_violations = 0
    num_unreachable = 0
    num_accepting_visits = []
    steps = []
    rets = []

    env.reset(seed=seed)

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)
    for i in pbar:
        obs, info = env.reset(), {}
        if render:
            print(obs['goal'])
        agent.reset()
        done = False
        num_steps = 0
        while not done:
            try:
                action = agent.get_action(obs, info, deterministic=deterministic)
            except NoPathsException:
                num_unreachable += 1
                rets.append(0)
                break
            action = action.flatten()
            # if type(action) != int else np.array(action)
            if action.shape == (1,):
                action = action[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1
            if done:
                if finite:
                    final_reward = int('success' in info)
                    if 'success' in info:
                        num_successes += 1
                        steps.append(num_steps)
                    elif 'violation' in info:
                        num_violations += 1
                    rets.append(final_reward * gamma ** (num_steps - 1))
                    if not render:
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),
                            'V': num_violations / (i + 1),
                            'ADR': np.mean(rets),
                            'AS': np.mean(steps),
                        })
                else:
                    if 'violation' in info:
                        num_violations += 1
                    else:
                        num_accepting_visits.append(info['num_accepting_visits'])
                    if not render:
                        pbar.set_postfix({
                            'A': np.mean(num_accepting_visits),
                            'V': num_violations / (i + 1),
                        })

    env.close()
    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        unreachable_rate = num_unreachable / num_episodes
        average_steps = np.mean(steps)
        adr = np.mean(rets)
        print(f'{seed}: {success_rate:.3f},{violation_rate:.3f},{unreachable_rate:.3f},{adr:.3f},{average_steps:.3f}')
        return num_successes, num_violations, average_steps
    else:
        average_visits = np.mean(num_accepting_visits)
        violation_rate = num_violations / num_episodes
        unreachable_rate = num_unreachable / num_episodes
        print(f'{seed}: {average_visits:.3f},{violation_rate:.3f},{unreachable_rate:.3f}')
        return average_visits, violation_rate


if __name__ == '__main__':
    main()
