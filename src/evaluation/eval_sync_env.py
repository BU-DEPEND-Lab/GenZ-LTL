import gymnasium as gym


class EvalSyncEnv(gym.Env):
    """A sequential execution of multiple environments, with predefined zones and tasks."""

    def __init__(self, envs, world_info_paths, tasks):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.world_info_paths = world_info_paths
        self.tasks = tasks
        if len(self.world_info_paths) > 0 and len(self.world_info_paths) != len(self.tasks):
            raise ValueError("Number of world info paths and tasks must be the same.")
        self.current = 0
        self.active_envs = [env for env in self.envs]

    def reset(self):
        self.current = 0
        self.active_envs = [env for env in self.envs]
        results = []
        for env in self.envs:
            if self.world_info_paths:
                env.load_world_info(self.world_info_paths[self.current])
            env.set_goal(self.tasks[self.current])
            results.append(env.reset())
            self.current += 1
        return results

    def step(self, actions):
        results = []
        to_remove = set()
        for i, env in enumerate(self.active_envs):
            obs, reward, done, info = env.step(actions[i])
            if done:
                if self.current < len(self.tasks):
                    if self.world_info_paths:
                        env.load_world_info(self.world_info_paths[self.current])
                    env.set_goal(self.tasks[self.current])
                    self.current += 1
                    obs = env.reset()
                else:
                    to_remove.add(i)
                    obs = None
            results.append((obs, reward, done, info))
        self.active_envs = [env for i, env in enumerate(self.active_envs) if i not in to_remove]
        results = zip(*results)
        return results

    def render(self):
        raise NotImplementedError
