from typing import Callable, Optional

import gymnasium
from gymnasium.wrappers import FlattenObservation, TimeLimit

from envs.remove_trunc_wrapper import RemoveTruncWrapper


def get_env_attr(env, attr: str):
    if hasattr(env, attr):
        return getattr(env, attr)
    if hasattr(env, 'env'):
        return getattr(env.unwrapped, attr)
    else:
        raise AttributeError(f'Attribute {attr} not found in env.')


def make_env(
        name: str,
        sampler: Callable[[list[str]], Callable],
        max_steps: Optional[int] = None,
        render_mode: str | None = None,
        sequence=False
):
    from envs.pretraining.pretraining_env import PretrainingEnv
    from envs.seq_wrapper import SequenceWrapper
    from envs.ldba_wrapper import LDBAWrapper
    from envs.ltl_wrapper import LTLWrapper

    if name.startswith('pretraining_'):
        underlying = name[len('pretraining_'):]
        underlying_env = make_env(underlying, sampler, max_steps, render_mode)
        propositions = get_env_attr(underlying_env, 'get_propositions')()
        impossible_assignments = get_env_attr(underlying_env, 'get_impossible_assignments')()
        env = PretrainingEnv(propositions, impossible_assignments)
        max_steps = max_steps or 100
    elif is_safety_gym_env(name):
        env = make_safety_gym_env(name, render_mode)
        max_steps = max_steps or 1000
    elif name.startswith('Letter'):
        env = make_letter_env(name, render_mode)
        max_steps = max_steps or 75
    elif name.startswith('FlatWorld'):
        env = make_flatworld_env()
        max_steps = max_steps or 500
    else:
        raise ValueError(f'Unknown environment: {name}')

    propositions = get_env_attr(env, 'get_propositions')()
    sample_task = sampler(propositions)
    if not sequence:
        # env = PartiallyOrderedWrapper(env, sample_task)
        env = LTLWrapper(env, sample_task)
        env = LDBAWrapper(env)
    else:
        env = SequenceWrapper(env, sample_task)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = RemoveTruncWrapper(env)
    return env


def make_env_safety(
        name: str,
        sampler: Callable[[list[str]], Callable],
        max_steps: Optional[int] = None,
        render_mode: str | None = None,
        sequence=False,
):
    from envs.pretraining.pretraining_env import PretrainingEnv
    from envs.seq_wrapper import SequenceSafetyWrapper
    from envs.ldba_wrapper import LDBAWrapper
    from envs.ltl_wrapper import LTLWrapper

    if name.startswith('pretraining_'):
        underlying = name[len('pretraining_'):]
        underlying_env = make_env(underlying, sampler, max_steps, render_mode)
        propositions = get_env_attr(underlying_env, 'get_propositions')()
        impossible_assignments = get_env_attr(underlying_env, 'get_impossible_assignments')()
        env = PretrainingEnv(propositions, impossible_assignments)
        max_steps = max_steps or 100
    elif is_safety_gym_env(name):
        env = make_safety_gym_env(name, render_mode)
        max_steps = max_steps or 1000
    elif name.startswith('Letter'):
        env = make_letter_env(name, render_mode)
        max_steps = max_steps or 75
    elif name.startswith('FlatWorld'):
        env = make_flatworld_env(name)
        max_steps = max_steps or 500
    else:
        raise ValueError(f'Unknown environment: {name}')

    propositions = get_env_attr(env, 'get_propositions')()
    sample_task = sampler(propositions)
    if not sequence:
        # env = PartiallyOrderedWrapper(env, sample_task)
        env = LTLWrapper(env, sample_task)
        env = LDBAWrapper(env)
    else:
        env = SequenceSafetyWrapper(env, sample_task)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = RemoveTruncWrapper(env)
    return env


def is_safety_gym_env(name: str) -> bool:
    return any([name.startswith(agent_name) for agent_name in ['Point', 'Car', 'Racecar', 'Doggo', 'Ant']])


def make_safety_gym_env(name: str, render_mode: str | None = None):
    # noinspection PyUnresolvedReferences
    import safety_gymnasium
    from envs.zones.safety_gym_wrapper import SafetyGymWrapper

    env = safety_gymnasium.make(name, render_mode=render_mode)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)
    return env


def make_letter_env(name: str, render_mode: str | None = None):
    import envs.letter_world

    env = gymnasium.make(name, render_mode=render_mode)
    return env


def make_flatworld_env(name: str):
    import envs.flatworld

    env = gymnasium.make(name)
    return env
