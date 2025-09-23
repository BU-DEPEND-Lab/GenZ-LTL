from envs import make_env_safety
# from ltl.samplers import AvoidSampler
from sequence.samplers import CurriculumSampler, curricula

env_name = 'PointLtlSafety2-v0'
seed = 1
max_steps = 1000
curriculum = curricula[env_name]
sampler = CurriculumSampler.partial(curriculum)
env = make_env_safety(env_name, sampler, render_mode=None, max_steps=max_steps, sequence=True)
path = f"eval_datasets/{env_name}"
obs = env.reset(seed=seed)
# print(obs['features'])

for i in range(5):
    obs = env.reset()
    print(f"agent_pos = {env.agent_pos[:2]}, agent_rot = {env.agent_rot}")
    print(obs['features'])

env.close()

# Goal: ((frozenset({yellow, green}), frozenset({magenta, blue})),) x 0
# Success: False
# Goal: ((frozenset({yellow}), frozenset({green})),) x 0
# Success: False
# Goal: ((frozenset({blue}), frozenset({yellow})),) x 0
# Success: False
# Goal: ((frozenset({green}), frozenset({magenta, blue})),) x 0
# Success: False
# Goal: ((frozenset({magenta, green}), frozenset({yellow})),) x 0
# Success: False