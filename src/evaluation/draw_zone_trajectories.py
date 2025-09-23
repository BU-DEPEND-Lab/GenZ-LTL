import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

from envs import make_env, make_env_safety
from ltl import FixedSampler
from model.model import build_model, build_model_safety
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearchSafety, NoPathsException
from utils.model_store import ModelStore
from visualize.zones import draw_trajectories


env_name = 'PointLtlSafety2-v0'
exp = 'GenZ-LTL'
seed = 1
formula = '!green U ((blue | magenta) & (!green U yellow))'

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

sampler = FixedSampler.partial(formula)
deterministic = True
max_steps = None # will use default episode length


if "Safety" in env_name:
    env = make_env_safety(env_name, sampler, render_mode=None, max_steps=max_steps)
else:
    env = make_env(env_name, sampler, render_mode=None, max_steps=max_steps)
config = model_configs[env_name]
model_store = ModelStore(env_name, exp, seed, None)
model_store.load_vocab()
training_status = model_store.load_training_status(map_location='cpu')
if "Safety" in env_name:
    model = build_model_safety(env, training_status, config)
else:
    model = build_model(env, training_status, config)

props = env.get_propositions()
search = ExhaustiveSearchSafety(env, model, props, num_loops=2)
agent = Agent(env, model, search=search, propositions=props, verbose=False)

num_episodes = 16
trajectories = []
zone_poss = []
titles = []
success, violation = 0, 0
num_unreachable = 0

env.reset(seed=seed)

pbar = trange(num_episodes)

for i in pbar:
    obs, info = env.reset(), {}
    agent.reset()
    done = False

    zone_radius = env.zone_radius
    zone_poss.append(env.zone_positions)
    agent_traj = []
    
    while not done:
        try:
            action = agent.get_action(obs, info, deterministic=deterministic)
            action = action.flatten()
            obs, reward, done, info = env.step(action)
            agent_traj.append(env.agent_pos[:2])
        except NoPathsException:
            num_unreachable += 1
            done = True

        if done:

            trajectories.append(agent_traj)
            if "success" in info:
                titles.append(f"{formula}_success")
                success += 1
            elif "violation" in info:
                titles.append(f"{formula}_violation")
                violation += 1
            else:
                titles.append(f"{formula}_not_finish")
            
            # print(f"accpeting num = {accept_num}")
            pbar.set_postfix({
                            'S': success / (i+1),
                            'V': violation / (i+1),
                        })

print(f"Formula: {formula}, Success: {success}, Violation: {violation}")
env.close()
cols = 4 if len(zone_poss) > 4 else len(zone_poss)
rows = 1 if len(zone_poss) <= 4 else 4
fig = draw_trajectories(zone_poss, zone_radius, trajectories, titles, cols, rows)

plt.savefig(f"{env_name}_{exp}_s{seed}_{formula}_trajectories.png", dpi=300)
