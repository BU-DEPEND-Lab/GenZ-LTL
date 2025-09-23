from typing import Callable

import numpy
import torch
import torch.nn.functional as F

from config import RCOConfig
from torch_ac.algos.base import BaseAlgoLag
import math


class RCO(BaseAlgoLag):
    """
    https://arxiv.org/pdf/2205.07536 and https://arxiv.org/abs/1705.10528
    """
    def __init__(self, envs, model, device, config: RCOConfig, preprocess_obss: Callable, parallel=False):

        num_steps_per_proc = config.steps_per_process

        super().__init__(envs, model, device, num_steps_per_proc, config.discount, config.lr, config.gae_lambda,
                         config.entropy_coef, config.value_loss_coef, config.max_grad_norm, preprocess_obss, parallel=parallel)

        self.clip_eps = config.clip_eps
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.target_kl = config.target_kl
        self.target_cost = config.target_cost
        self.min_lag = config.min_lag
        self.max_lag = config.max_lag
        self.act_shape = envs[0].action_space.shape

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), config.lr, eps=config.optim_eps)
        
        self.batch_num = 0
        
        print(f"target_cost = {self.target_cost}, min_lag = {self.min_lag}, max_lag = {self.max_lag}")

        
    def update_parameters(self, exps):
        # update networks

        for n in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_cost_values = []
            log_lags = []
            log_policy_losses = []
            log_policy_losses_reward = []
            log_policy_losses_cost = []
            log_value_losses = []
            log_cost_value_losses = []
            log_lag_losses = []
            log_grad_norms = []
            iter_counts, approx_kl = 0, 0.0
            
            for inds in self._get_batches_starting_indexes():

                # Create a sub-batch of experience
                sb = exps[inds]

                # Compute loss
                dist, value, cost_value, lag = self.model(sb.obs, collect=False)

                entropy = dist.entropy().mean()
                
                log_prob = dist.log_prob(sb.action)
                delta_log_prob = log_prob - sb.log_prob
                ratio = torch.exp(delta_log_prob)
                
                approx_kl += -delta_log_prob.mean().item()
                iter_counts += 1
                
                reward_surr1 = ratio * sb.advantage
                reward_surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                policy_loss_reward = torch.min(reward_surr1, reward_surr2)
                
                policy_loss_cost = ratio * sb.cost_advantage
                lag = torch.clamp(lag, self.min_lag, self.max_lag)
                
                policy_loss = (-policy_loss_reward + \
                    lag.detach() * (policy_loss_cost + (1 - self.cost_discount) * sb.cost_returnn - self.target_cost)).mean() # HJ_v4 / HJ_v5
                
                lag_loss = -(lag * (policy_loss_cost.detach() + (1 - self.cost_discount) * sb.cost_returnn - self.target_cost)).mean() # HJ_v4 / HJ_v5
                
                value_clipped = sb.value + torch.clamp(value - sb.value, 
                                                        -self.clip_eps, self.clip_eps)
                value_surr1 = (value - sb.returnn).pow(2)
                value_surr2 = (value_clipped - sb.returnn).pow(2)
                value_loss = torch.max(value_surr1, value_surr2).mean()
                
                cost_value_clipped = sb.cost_value + torch.clamp(cost_value - sb.cost_value, 
                                                                    -self.clip_eps, self.clip_eps)
                cost_value_surr1 = (cost_value - sb.cost_returnn).pow(2)
                cost_value_surr2 = (cost_value_clipped - sb.cost_returnn).pow(2)
                cost_value_loss = torch.max(cost_value_surr1, cost_value_surr2).mean()

                total_loss = policy_loss + \
                    self.value_loss_coef * value_loss + \
                    self.value_loss_coef * cost_value_loss + \
                    self.value_loss_coef * lag_loss - \
                    self.entropy_coef * entropy
                    
                if total_loss.isnan():
                    print("Loss is NaN")

                # Update actor-critic
                self.optimizer.zero_grad()
                total_loss.backward()
                # p.grad can be None if the GNN is not used (e.g. because all assignments only involve a single proposition)
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.requires_grad and p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad and p.grad is not None],
                                               self.max_grad_norm)
                self.optimizer.step()  # TODO: multiply by probability of choosing continuous action or discrete. update step() in seq_wrapper

                if any(torch.isnan(p).any() for p in self.model.parameters()):
                    print("Model parameters are NaN")

                # Update log values
                log_entropies.append(entropy.item())
                log_values.append(value.mean().item())
                log_cost_values.append(cost_value.mean().item())
                log_lags.append(lag.mean().item())
                log_policy_losses.append(policy_loss.item())
                log_policy_losses_reward.append(policy_loss_reward.mean().item())
                log_policy_losses_cost.append(policy_loss_cost.mean().item())
                log_value_losses.append(value_loss.item())
                log_cost_value_losses.append(cost_value_loss.item())
                log_lag_losses.append(lag_loss.item())
                log_grad_norms.append(grad_norm)

            approx_kl /= iter_counts + 1e-7
            if approx_kl > self.target_kl:
                print(f"Early stop at epoch {n} due to reaching max kl: {self.target_kl}, current kl: {approx_kl}")
                break

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "cost_value": numpy.mean(log_cost_values),
            "lagrangian": numpy.mean(log_lags),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "cost_value_loss": numpy.mean(log_cost_value_losses),
            "lag_loss": numpy.mean(log_lag_losses),
            "grad_norm": numpy.mean(log_grad_norms),
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_steps, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_steps_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
