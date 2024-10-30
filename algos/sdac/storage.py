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
            n_steps:int,
            n_update_steps:int,
            len_replay_buffer:int) -> None:
        self.device = device
        self.discount_factor = discount_factor
        self.gae_coeff = gae_coeff
        self.n_target_quantiles = n_target_quantiles
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_update_steps = n_update_steps
        self.len_replay_buffer = len_replay_buffer
        self.n_steps_per_env = int(self.n_steps/self.n_envs)
        self.n_update_steps_per_env = int(self.n_update_steps/self.n_envs)
        self.len_replay_buffer_per_env = int(self.len_replay_buffer/self.n_envs)
        self.storage = [deque(maxlen=self.len_replay_buffer_per_env) for _ in range(self.n_envs)]

        # projection operator
        self._lib = ctypes.cdll.LoadLibrary(f'{os.path.dirname(os.path.abspath(__file__))}/cpp_modules/main.so')
        self._lib.projection.restype = None

    ################
    # Public Methods
    ################

    def getLen(self) -> int:
        return np.sum([len(self.storage[i]) for i in range(self.n_envs)])

    def addTransition(self, states:np.ndarray, actions:np.ndarray, normal_actions:np.ndarray, log_probs:np.ndarray, 
                      rewards:np.ndarray, costs:np.ndarray, dones:np.ndarray, 
                      fails:np.ndarray, next_states:np.ndarray) -> None:
        for env_idx in range(self.n_envs):
            self.storage[env_idx].append([
                states[env_idx], actions[env_idx], normal_actions[env_idx], log_probs[env_idx], rewards[env_idx], 
                costs[env_idx], dones[env_idx], fails[env_idx], next_states[env_idx]
            ])

    @torch.no_grad()    
    def getBatches(self, obs_rms, reward_rms, actor:torch.nn.Module, reward_critic:torch.nn.Module, 
                   cost_critics:torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        state_len = len(self.storage[0])
        n_latest_steps = min(state_len, self.n_steps_per_env)
        n_update_steps = min(state_len, self.n_update_steps_per_env)

        # process the latest trajectories
        start_idx = state_len - n_latest_steps
        end_idx = start_idx + n_latest_steps
        states_list, actions_list, reward_targets_list, cost_targets_lists \
            = self._processBatches(obs_rms, reward_rms, actor, reward_critic, cost_critics, start_idx, end_idx)

        # process the rest trajectories
        if n_update_steps > n_latest_steps:
            start_idx = np.random.randint(0, state_len - n_update_steps + 1)
            end_idx = start_idx + n_update_steps - n_latest_steps
            temp_states_list, temp_actions_list, temp_reward_targets_list, temp_cost_targets_lists \
                = self._processBatches(obs_rms, reward_rms, actor, reward_critic, cost_critics, start_idx, end_idx)
            states_list.extend(temp_states_list)
            actions_list.extend(temp_actions_list)
            reward_targets_list.extend(temp_reward_targets_list)
            for cost_idx in range(len(cost_critics)):
                cost_targets_lists[cost_idx].extend(temp_cost_targets_lists[cost_idx])

        # convert to tensor
        states_tensor = torch.tensor(np.concatenate(states_list, axis=0), device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(np.concatenate(actions_list, axis=0), device=self.device, dtype=torch.float32)
        reward_targets_tensor = torch.tensor(np.concatenate(reward_targets_list, axis=0), device=self.device, dtype=torch.float32)
        cost_targets_tensor_list = []
        for cost_idx in range(len(cost_critics)):
            cost_targets_tensor_list.append(
                torch.tensor(np.concatenate(cost_targets_lists[cost_idx], axis=0), device=self.device, dtype=torch.float32)
            )
        return states_tensor, actions_tensor, reward_targets_tensor, cost_targets_tensor_list

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
    
    def _processBatches(self, obs_rms, reward_rms, actor:torch.nn.Module, 
                        reward_critic:torch.nn.Module, cost_critics:torch.nn.Module, 
                        start_idx:int, end_idx:int) -> Tuple[list, list, list, list]:
        cost_dim = len(cost_critics)
        batch_size = end_idx - start_idx
        states_list = []
        actions_list = []
        reward_targets_list = []
        cost_targets_lists = [[] for _ in range(cost_dim)]

        for env_idx in range(self.n_envs):
            env_trajs = list(self.storage[env_idx])[start_idx:end_idx]
            states = np.array([traj[0] for traj in env_trajs])
            actions = np.array([traj[1] for traj in env_trajs])
            normal_actions = np.array([traj[2] for traj in env_trajs])
            log_probs = np.array([traj[3] for traj in env_trajs])
            rewards = np.array([traj[4] for traj in env_trajs])
            costs = np.array([traj[5] for traj in env_trajs])
            dones = np.array([traj[6] for traj in env_trajs])
            fails = np.array([traj[7] for traj in env_trajs])
            next_states = np.array([traj[8] for traj in env_trajs])

            # normalize 
            states = obs_rms.normalize(states)
            next_states = obs_rms.normalize(next_states)
            rewards = reward_rms.normalize(rewards)
            costs = (1.0 - fails.reshape(-1, 1))*costs + fails.reshape(-1, 1)*costs/(1.0 - self.discount_factor)

            states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
            normal_actions_tensor = torch.tensor(normal_actions, device=self.device, dtype=torch.float32)
            mu_log_probs_tensor = torch.tensor(log_probs, device=self.device, dtype=torch.float32)

            # for rho
            means_tensor, _, stds_tensor = actor(states_tensor)
            old_log_probs_tensor = actor.getLogProbJIT(normal_actions_tensor, means_tensor, stds_tensor)
            rhos_tensor = torch.clamp(torch.exp(old_log_probs_tensor - mu_log_probs_tensor), 0.0, 100.0)
            rhos = rhos_tensor.detach().cpu().numpy()

            # get next actions
            epsilons_tensor = torch.randn_like(actions_tensor)
            actor.updateActionDist(next_states_tensor, epsilons_tensor)
            next_actions_tensor = actor.sample(deterministic=False)[0]

            # get targets
            next_reward_quantiles_tensor = reward_critic(next_states_tensor, next_actions_tensor).reshape(batch_size, -1)
            next_reward_quantiles = torch.sort(next_reward_quantiles_tensor, dim=-1)[0].detach().cpu().numpy()
            reward_targets = self._getQuantileTargets(rewards, dones, fails, rhos, next_reward_quantiles)
            reward_targets_list.append(reward_targets)
            for cost_idx in range(cost_dim):
                next_cost_quantiles_tensor = cost_critics[cost_idx](next_states_tensor, next_actions_tensor).reshape(batch_size, -1)
                next_cost_quantiles = torch.sort(next_cost_quantiles_tensor, dim=-1)[0].detach().cpu().numpy()
                cost_targets = self._getQuantileTargets(costs[:, cost_idx], dones, fails, rhos, next_cost_quantiles)
                cost_targets_lists[cost_idx].append(cost_targets)

            # append
            states_list.append(states)
            actions_list.append(actions)

        return states_list, actions_list, reward_targets_list, cost_targets_lists
