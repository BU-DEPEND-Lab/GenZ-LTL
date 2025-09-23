from dm_control import suite

from envs.dmc.ltl_cartpole import ltl_cartpole
from envs.dmc.ltl_point_mass import ltl_point_mass


def register_env(env_name, module):
    suite._DOMAINS[env_name] = module


def refresh_suite():
    # A sequence containing all (domain name, task name) pairs.
    suite.ALL_TASKS = suite._get_tasks(tag=None)

    # Subsets of ALL_TASKS, generated via the tag mechanism.
    suite.BENCHMARKING = suite._get_tasks('benchmarking')
    suite.EASY = suite._get_tasks('easy')
    suite.HARD = suite._get_tasks('hard')
    suite.EXTRA = tuple(sorted(set(suite.ALL_TASKS) - set(suite.BENCHMARKING)))
    suite.NO_REWARD_VIZ = suite._get_tasks('no_reward_visualization')
    suite.REWARD_VIZ = tuple(sorted(set(suite.ALL_TASKS) - set(suite.NO_REWARD_VIZ)))

    # A mapping from each domain name to a sequence of its task names.
    suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)


def register_with_suite():
    envs = [ltl_cartpole, ltl_point_mass]
    for env in envs:
        env_name = env.__name__.split('.')[-1]
        register_env(env_name, env)
    refresh_suite()
