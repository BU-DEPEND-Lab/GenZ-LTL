import copy

import numpy as np

import preprocessing
from model.model import Model


class SequenceAgent:
    def __init__(self, model: Model, propositions: set[str], verbose=False):
        self.model = model
        self.propositions = propositions

    def get_action(self, obs, info, deterministic=False) -> np.ndarray:
        assert 'goal' in obs
        return self.forward(obs, deterministic)

    def reset(self):
        pass

    def forward(self, obs, deterministic=False) -> np.ndarray:
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs, self.propositions)
        dist, value = self.model(preprocessed)
        action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy()

    def get_value(self, obs, goal):
        obs = copy.deepcopy(obs)
        obs['goal'] = goal
        preprocessed = preprocessing.preprocess_obss([obs])
        _, value = self.model(preprocessed)
        return value
