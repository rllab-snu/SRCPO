from collections import deque
from copy import deepcopy
from typing import Tuple
import numpy as np
import ctypes
import torch
import os

EPS = 1e-8 

def ctypeArrayConvert(arr):
    arr = np.ravel(arr)
    return (ctypes.c_double * len(arr))(*arr)


class ReplayBuffer:
    def __init__(
            self, device:torch.device, 
            discount_factor:float, 
            gae_coeff:float, 
            n_target_quantiles:int,
            n_envs:int, 
            n_steps:int) -> None:
        self.device = device
        self.discount_factor = discount_factor
        self.gae_coeff = gae_coeff
        self.n_target_quantiles = n_target_quantiles
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_steps_per_env = int(self.n_steps/self.n_envs)
        self.storage = [deque(maxlen=self.n_steps_per_env) for _ in range(self.n_envs)]

        # projection operator
        self._lib = ctypes.cdll.LoadLibrary(f'{os.path.dirname(os.path.abspath(__file__))}/cpp_modules/main.so')
        self._lib.projection.restype = None

    ################
    # Public Methods
    ################

    def getLen(self) -> int:
        return np.sum([len(self.storage[i]) for i in range(self.n_envs)])

    def addTransition(
            self, states:np.ndarray, actions:np.ndarray, normal_actions:np.ndarray, log_probs:np.ndarray, 
            rewards:np.ndarray, costs:np.ndarray, dones:np.ndarray, fails:np.ndarray, 
            next_states:np.ndarray, stat_variables:np.ndarray, next_stat_variables:np.ndarray) -> None:
        for env_idx in range(self.n_envs):
            self.storage[env_idx].append([
                states[env_idx], actions[env_idx], normal_actions[env_idx], log_probs[env_idx], rewards[env_idx], 
                costs[env_idx], dones[env_idx], fails[env_idx], next_states[env_idx], stat_variables[env_idx], next_stat_variables[env_idx],
            ])

    @torch.no_grad()
    def getBatches(
        self, obs_rms, reward_rms, actor:torch.nn.Module, 
        reward_critic:torch.nn.Module, cost_critics:torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        states_list = []
        actions_list = []
        normal_actions_list = []
        reward_targets_list = []
        reward_gaes_list = []
        cost_targets_lists = [[] for _ in range(len(cost_critics))]
        cost_gaes_lists = [[] for _ in range(len(cost_critics))]
        for env_idx in range(self.n_envs):
            env_trajs = list(self.storage[env_idx])
            states = np.array([traj[0] for traj in env_trajs])
            actions = np.array([traj[1] for traj in env_trajs])
            normal_actions = np.array([traj[2] for traj in env_trajs])
            log_probs = np.array([traj[3] for traj in env_trajs])
            rewards = np.array([traj[4] for traj in env_trajs]) # (n_steps_per_env,)
            costs = np.array([traj[5] for traj in env_trajs]) # (n_steps_per_env, cost_dim)
            dones = np.array([traj[6] for traj in env_trajs])
            fails = np.array([traj[7] for traj in env_trajs])
            next_states = np.array([traj[8] for traj in env_trajs])
            stat_variables = np.array([traj[9] for traj in env_trajs]) # (n_steps_per_env, cost_dim)
            next_stat_variables = np.array([traj[10] for traj in env_trajs]) # (n_steps_per_env, cost_dim)

            # normalize 
            states = obs_rms.normalize(states)
            next_states = obs_rms.normalize(next_states)
            rewards = reward_rms.normalize(rewards)
            costs = (1.0 - fails.reshape((-1, 1)))*costs + fails.reshape((-1, 1))*costs/(1.0 - self.discount_factor)

            states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
            normal_actions_tensor = torch.tensor(normal_actions, device=self.device, dtype=torch.float32)
            mu_log_probs_tensor = torch.tensor(log_probs, device=self.device, dtype=torch.float32)
            stat_variables_tensor = torch.tensor(stat_variables, device=self.device, dtype=torch.float32)
            next_stat_variables_tensor = torch.tensor(next_stat_variables, device=self.device, dtype=torch.float32)

            # for rho
            means_tensor, _, stds_tensor = actor(states_tensor)
            old_log_probs_tensor = actor.getLogProbJIT(normal_actions_tensor, means_tensor, stds_tensor)
            rhos_tensor = torch.clamp(torch.exp(old_log_probs_tensor - mu_log_probs_tensor), 0.0, 100.0)
            rhos = rhos_tensor.detach().cpu().numpy()

            # get targets
            next_reward_quantiles_tensor = reward_critic(next_states_tensor).reshape(actions_tensor.shape[0], -1)
            next_reward_quantiles = torch.sort(next_reward_quantiles_tensor, dim=-1)[0].detach().cpu().numpy()
            reward_targets_list.append(self._getQuantileTargets(rewards, dones, fails, rhos, next_reward_quantiles))
            next_cost_quantiles_tensor_list = []
            for cost_idx in range(len(cost_critics)):
                next_cost_quantiles_tensor = cost_critics[cost_idx](next_states_tensor).reshape(actions_tensor.shape[0], -1)
                next_cost_quantiles_tensor_list.append(next_cost_quantiles_tensor)
                next_cost_quantiles = torch.sort(next_cost_quantiles_tensor, dim=-1)[0].detach().cpu().numpy()
                cost_targets_lists[cost_idx].append(self._getQuantileTargets(costs[:, cost_idx], dones, fails, rhos, next_cost_quantiles))

            # get GAEs
            next_reward_values = next_reward_quantiles_tensor.mean(dim=-1)
            reward_values = reward_critic(states_tensor).reshape(actions_tensor.shape[0], -1).mean(dim=-1).detach().cpu().numpy()
            next_cost_values_list = []
            cost_values_list = []
            for cost_idx in range(len(cost_critics)):
                next_cost_values = ((next_cost_quantiles_tensor_list[cost_idx] - next_stat_variables_tensor[:, cost_idx:(cost_idx+1)]) \
                                    *(next_cost_quantiles_tensor_list[cost_idx] >= next_stat_variables_tensor[:, cost_idx:(cost_idx+1)]).float()).mean(dim=-1).detach().cpu().numpy()
                cost_quantiles_tensor = cost_critics[cost_idx](states_tensor).reshape(actions_tensor.shape[0], -1)
                cost_values = ((cost_quantiles_tensor - stat_variables_tensor[:, cost_idx:(cost_idx+1)])*(cost_quantiles_tensor >= stat_variables_tensor[:, cost_idx:(cost_idx+1)]).float()
                                    ).mean(dim=-1).detach().cpu().numpy()
                next_cost_values_list.append(next_cost_values)
                cost_values_list.append(cost_values)
            reward_delta = 0.0
            cost_delta = np.zeros(len(cost_critics))
            reward_targets = np.zeros_like(rewards)
            cost_targets = np.zeros_like(costs)
            for t in reversed(range(len(reward_targets))):
                reward_targets[t] = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_reward_values[t] \
                                + self.discount_factor*(1.0 - dones[t])*reward_delta
                reward_delta = self.gae_coeff*(reward_targets[t] - reward_values[t])
                for cost_idx in range(len(cost_critics)):
                    cost_targets[t, cost_idx] = self.discount_factor*(1.0 - fails[t])*next_cost_values_list[cost_idx][t] \
                                    + self.discount_factor*(1.0 - dones[t])*cost_delta[cost_idx]
                    cost_delta[cost_idx] = self.gae_coeff*(cost_targets[t, cost_idx] - cost_values_list[cost_idx][t])
            reward_gaes = reward_targets - reward_values

            # append
            states_list.append(states)
            actions_list.append(actions)
            normal_actions_list.append(normal_actions)
            reward_gaes_list.append(reward_gaes)
            for cost_idx in range(len(cost_critics)):
                cost_gaes_lists[cost_idx].append(cost_targets[:, cost_idx] - cost_values_list[cost_idx])

        states_tensor = torch.tensor(np.concatenate(states_list, axis=0), device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(np.concatenate(actions_list, axis=0), device=self.device, dtype=torch.float32)
        normal_actions_tensor = torch.tensor(np.concatenate(normal_actions_list, axis=0), device=self.device, dtype=torch.float32)
        reward_targets_tensor = torch.tensor(np.concatenate(reward_targets_list, axis=0), device=self.device, dtype=torch.float32)
        reward_gaes_tensor = torch.tensor(np.concatenate(reward_gaes_list, axis=0), device=self.device, dtype=torch.float32)
        cost_targets_tensor_list = []
        cost_gaes_tensor_list = []
        for cost_idx in range(len(cost_critics)):
            cost_targets_tensor = torch.tensor(np.concatenate(cost_targets_lists[cost_idx], axis=0), device=self.device, dtype=torch.float32)
            cost_gaes_tensor = torch.tensor(np.concatenate(cost_gaes_lists[cost_idx], axis=0), device=self.device, dtype=torch.float32)
            cost_targets_tensor_list.append(cost_targets_tensor)
            cost_gaes_tensor_list.append(cost_gaes_tensor)
        return states_tensor, actions_tensor, normal_actions_tensor, \
            reward_targets_tensor, reward_gaes_tensor, cost_targets_tensor_list, cost_gaes_tensor_list

    #################
    # private methods
    #################

    def _projection(self, quantiles1:np.ndarray, weight1:float, quantiles2:np.ndarray, weight2:float) -> np.ndarray:
        n_quantiles1 = len(quantiles1)
        n_quantiles2 = len(quantiles2)
        assert n_quantiles1 <= n_quantiles2
        n_quantiles3 = self.n_target_quantiles
        cpp_quantiles1 = ctypeArrayConvert(quantiles1)
        cpp_quantiles2 = ctypeArrayConvert(quantiles2)
        cpp_new_quantiles = ctypeArrayConvert(np.zeros(n_quantiles3))
        self._lib.projection.argtypes = [
            ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double*n_quantiles1), ctypes.c_int, ctypes.c_double, 
            ctypes.POINTER(ctypes.c_double*n_quantiles2), ctypes.c_int, ctypes.POINTER(ctypes.c_double*n_quantiles3)
        ]
        self._lib.projection(n_quantiles1, weight1, cpp_quantiles1, n_quantiles2, 
                             weight2, cpp_quantiles2, n_quantiles3, cpp_new_quantiles)
        new_quantiles = np.array(cpp_new_quantiles)
        return new_quantiles

    def _getQuantileTargets(self, rewards:np.ndarray, dones:np.ndarray, fails:np.ndarray, 
                            rhos:np.ndarray, next_quantiles:np.ndarray) -> np.ndarray:
        """
        inputs:
            rewards: (batch_size,)
            dones: (batch_size,)
            fails: (batch_size,)
            rhos: (batch_size,)
            next_quantiles: (batch_size, n_critics*n_quantiles)
        outputs:
            target_quantiles: (batch_size, n_target_quantiles)
        """
        target_quantiles = np.zeros((next_quantiles.shape[0], self.n_target_quantiles))
        gae_target = rewards[-1] + self.discount_factor*(1.0 - fails[-1])*next_quantiles[-1] # (n_critics*n_quantiles,)
        gae_weight = self.gae_coeff
        for t in reversed(range(len(target_quantiles))):
            target = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_quantiles[t] # (n_critics*n_quantiles,)
            target = self._projection(target, 1.0 - self.gae_coeff, gae_target, gae_weight) # (n_target_quantiles,)
            target_quantiles[t, :] = target[:]
            if t != 0:
                if self.gae_coeff != 1.0:
                    gae_weight = self.gae_coeff*rhos[t]*(1.0 - dones[t-1])*(1.0 - self.gae_coeff + gae_weight)
                gae_target = rewards[t-1] + self.discount_factor*(1.0 - fails[t-1])*target # (n_target_quantiles,)
        return target_quantiles
    