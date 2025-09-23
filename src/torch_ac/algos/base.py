from abc import ABC, abstractmethod
from collections import defaultdict

import torch

from torch_ac.utils import DictList, ParallelEnv

import numpy as np
from tqdm import trange

from torch_ac.utils.sync_env import SyncEnv


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, model, device, num_steps_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, preprocess_obss, recurrence=1, parallel=False):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_steps_per_proc : int
            the number of steps collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.propositions = envs[0].get_propositions()
        self.env = ParallelEnv(envs) if parallel else SyncEnv(envs)
        self.model = model
        self.device = device
        self.num_steps_per_proc = num_steps_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss
        self.action_space_shape = envs[0].action_space.shape

        # Control parameters

        assert self.model.recurrent or self.recurrence == 1
        assert self.num_steps_per_proc % self.recurrence == 0

        # Configure model

        self.model.to(self.device)
        self.model.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_steps = self.num_steps_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_steps_per_proc, self.num_procs)
        act_shape = shape + self.action_space_shape

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        if self.model.recurrent:
            self.memory = torch.zeros(shape[1], self.model.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.model.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*act_shape, device=self.device)  # , dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.qs = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_steps = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_num_steps = [0] * self.num_procs
        self.log_success = [0] * self.num_procs
        self.log_violation = [0] * self.num_procs

        self.goal_success = defaultdict(int)
        self.goal_counts = defaultdict(int)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_steps_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_steps_per_proc` steps contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_steps_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, self.propositions, device=self.device)
            with torch.no_grad():
                if self.model.recurrent:
                    dist, value, memory = self.model(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.model(preprocessed_obs)
            action = dist.sample()

            obs, reward, done, info = self.env.step(action.cpu().numpy())

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.model.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value

            self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action).detach()

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_num_steps += torch.ones(self.num_procs, device=self.device)

            for j, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[j].item())
                    self.log_num_steps.append(self.log_episode_num_steps[j].item())
                    self.log_success.append(int('success' in info[j]))
                    self.log_violation.append(int('violation' in info[j]))
                    assert not (('success' in info[j]) and ('violation' in info[j]))

                    goal = self.obss[i][j]['initial_goal']
                    self.goal_success[goal] += int('success' in info[j])
                    self.goal_counts[goal] += 1

            self.log_episode_return *= self.mask
            self.log_episode_num_steps *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, self.propositions, device=self.device)
        with torch.no_grad():
            if self.model.recurrent:
                _, next_value, _ = self.model(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.model(preprocessed_obs)

        for i in reversed(range(self.num_steps_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_steps_per_proc - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_steps_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_steps_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
            self.qs[i] = self.rewards[i] + self.discount * next_value * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_steps_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_steps_per_proc)]
        if self.model.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape((-1,) + self.action_space_shape)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.qs = self.qs.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        # exps.log_prob = self.log_probs.transpose(0, 1).reshape((-1,) + self.action_space_shape)  # TODO: fix

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, self.propositions, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "num_steps_per_episode": self.log_num_steps[-keep:],
            "success_per_episode": self.log_success[-keep:],
            "violation_per_episode": self.log_violation[-keep:],
            "num_steps": self.num_steps,
            "avg_goal_success": {k: float(v) / self.goal_counts[k] for k, v in self.goal_success.items()},
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_num_steps = self.log_num_steps[-self.num_procs:]
        self.goal_success = defaultdict(int)
        self.goal_counts = defaultdict(int)

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass


class BaseAlgoLag(ABC):
    """The base class for RCO algorithms."""

    def __init__(self, envs, model, device, num_steps_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, preprocess_obss, recurrence=1, parallel=False):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_steps_per_proc : int
            the number of steps collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.propositions = envs[0].get_propositions()
        self.env = ParallelEnv(envs) if parallel else SyncEnv(envs)
        self.model = model
        self.device = device
        self.num_steps_per_proc = num_steps_per_proc
        self.discount = discount
        self.cost_discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.cost_gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss
        self.action_space_shape = envs[0].action_space.shape

        # Control parameters

        assert self.model.recurrent or self.recurrence == 1
        assert self.num_steps_per_proc % self.recurrence == 0

        # Configure model

        self.model.to(self.device)
        self.model.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_steps = self.num_steps_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_steps_per_proc, self.num_procs)
        act_shape = shape + self.action_space_shape

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*act_shape, device=self.device)  # , dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.cost_values = torch.zeros(*shape, device=self.device) # for cost
        self.qs = torch.zeros(*shape, device=self.device)
        self.cost_qs = torch.zeros(*shape, device=self.device) # for cost
        self.rewards = torch.zeros(*shape, device=self.device)
        self.returnn = torch.zeros(*shape, device=self.device)
        self.costs = torch.zeros(*shape, device=self.device) # for cost
        self.cost_returnn = torch.zeros(*shape, device=self.device) # for cost
        self.advantages = torch.zeros(*shape, device=self.device)
        self.cost_advantages = torch.zeros(*shape, device=self.device) # for cost
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_cost_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_steps = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_cost_return = [0] * self.num_procs
        self.log_num_steps = [0] * self.num_procs
        self.log_success = [0] * self.num_procs
        self.log_violation = [0] * self.num_procs

        self.goal_success = defaultdict(int)
        self.goal_counts = defaultdict(int)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_steps_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_steps_per_proc` steps contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in trange(self.num_steps_per_proc, desc="collecting data ...", leave=False):
            # Do one agent-environment interaction

            # src/preprocessing/preprocessing.py
            preprocessed_obs = self.preprocess_obss(self.obs, self.propositions, device=self.device)
            # print(f"preprocessed_obs = {preprocessed_obs}")
            with torch.no_grad():
                # src/model/model.py
                dist, value, cost_value = self.model(preprocessed_obs)
            action = dist.sample()

            obs, reward, done, info = self.env.step(action.cpu().numpy())
            reward_cost = np.array(reward) # reward is a tuple that contains reward and cost
            reward = reward_cost[:, 0]
            cost = reward_cost[:, 1]
            
            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            self.masks[i] = self.mask
            
            # reward == 1 means success and a new reach will be sampled
            # the episode continues but store them into separate rollouts
            # done == 1 means violation and the episode will be terminated
            new_done = np.logical_or(reward, done)
            self.mask = 1 - torch.tensor(new_done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            self.cost_values[i] = cost_value

            self.rewards[i] = torch.tensor(reward, device=self.device)
            self.costs[i] = torch.tensor(cost, device=self.device)
            self.log_probs[i] = dist.log_prob(action).detach()

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            log_cost = np.where(cost > 0, 1.0, 0.0)
            self.log_episode_cost_return += torch.tensor(log_cost, device=self.device, dtype=torch.float)
            self.log_episode_num_steps += torch.ones(self.num_procs, device=self.device)

            for j, done_ in enumerate(new_done):
                if done_:

                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[j].item())
                    self.log_cost_return.append(self.log_episode_cost_return[j].item())
                    self.log_num_steps.append(self.log_episode_num_steps[j].item())
                    self.log_success.append(int('success' in info[j]))
                    self.log_violation.append(int('violation' in info[j]))
                    assert not (('success' in info[j]) and ('violation' in info[j]))

                    goal = self.obss[i][j]['initial_goal']
                    self.goal_success[goal] += int('success' in info[j])
                    self.goal_counts[goal] += 1

            self.log_episode_return *= self.mask
            self.log_episode_cost_return *= self.mask
            self.log_episode_num_steps *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, self.propositions, device=self.device)
        with torch.no_grad():
            _, next_value, next_cost_value = self.model(preprocessed_obs)

        for i in reversed(range(self.num_steps_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_steps_per_proc - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_steps_per_proc - 1 else next_value
            next_cost_value = self.cost_values[i + 1] if i < self.num_steps_per_proc - 1 else next_cost_value
            next_advantage = self.advantages[i + 1] if i < self.num_steps_per_proc - 1 else 0
            next_cost_advantage = self.cost_advantages[i + 1] if i < self.num_steps_per_proc - 1 else 0
            
            # compute return-to-go and cost-to-go
            next_return = self.returnn[i + 1] if i < self.num_steps_per_proc - 1 else 0
            next_cost_return = self.cost_returnn[i + 1] if i < self.num_steps_per_proc - 1 else 0
            
            self.returnn[i] = self.rewards[i] + self.discount * next_return * next_mask
            self.cost_returnn[i] = torch.maximum(self.costs[i], next_cost_return * next_mask)
            
            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            cost_delta = (1 - self.cost_discount) * self.costs[i] + \
                self.cost_discount * (torch.maximum(self.costs[i], next_cost_value)) * next_mask - self.cost_values[i]
            
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
            self.cost_advantages[i] = cost_delta + self.cost_discount * self.cost_gae_lambda * next_cost_advantage * next_mask
            self.qs[i] = self.rewards[i] + self.discount * next_value * next_mask
            self.cost_qs[i] = self.costs[i] + self.cost_discount * next_cost_value * next_mask
            
        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_steps_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_steps_per_proc)]

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape((-1,) + self.action_space_shape)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.cost_value = self.cost_values.transpose(0, 1).reshape(-1)
        exps.qs = self.qs.transpose(0, 1).reshape(-1)
        exps.cost_qs = self.cost_qs.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.costs = self.costs.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.cost_advantage = self.cost_advantages.transpose(0, 1).reshape(-1) # TODO: try add an integer
        
        exps.returnn = exps.value + exps.advantage
        exps.cost_returnn = self.cost_returnn.transpose(0, 1).reshape(-1)
        
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        
        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, self.propositions, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "cost_return_per_episode": self.log_cost_return[-keep:],
            "num_steps_per_episode": self.log_num_steps[-keep:],
            "success_per_episode": self.log_success[-keep:],
            "violation_per_episode": self.log_violation[-keep:],
            "num_steps": self.num_steps,
            "avg_goal_success": {k: float(v) / self.goal_counts[k] for k, v in self.goal_success.items()},
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_cost_return = self.log_cost_return[-self.num_procs:]
        self.log_num_steps = self.log_num_steps[-self.num_procs:]
        self.goal_success = defaultdict(int)
        self.goal_counts = defaultdict(int)

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
