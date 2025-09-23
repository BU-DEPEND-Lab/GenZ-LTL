from multiprocessing import Process, Pipe
import gymnasium as gym


class SyncEnv(gym.Env):
    """A sequential execution of multiple environments."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        results = [env.reset() for env in self.envs]
        return results

    def step(self, actions):
        results = []
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(actions[i])
            if done:
                obs = env.reset()
            results.append((obs, reward, done, info))
        results = zip(*results)
        return results

    def render(self):
        raise NotImplementedError
