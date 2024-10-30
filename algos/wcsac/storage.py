from collections import deque
from copy import deepcopy
import numpy as np
import random
import torch
import os

EPS = 1e-8 

class ReplayBuffer:
    def __init__(self, len_replay_buffer:int, batch_size:int, discount_factor:float, device:torch.device) -> None:
        self.len_replay_buffer = len_replay_buffer
        self.batch_size = batch_size
        self.device = device
        self.discount_factor = discount_factor
        self.storage = deque(maxlen=self.len_replay_buffer)

    ################
    # Public Methods
    ################

    def getLen(self):
        return len(self.storage)

    def addTransition(self, states, actions, rewards, costs, dones, fails, next_states):
        for env_idx in range(len(states)):
            self.storage.append([
                states[env_idx], actions[env_idx], rewards[env_idx], costs[env_idx], 
                dones[env_idx], fails[env_idx], next_states[env_idx]])
    
    def getBatches(self, obs_rms, reward_rms):
        sampled_trajectory = random.sample(self.storage, min(self.batch_size, len(self.storage)))
        states = np.array([t[0] for t in sampled_trajectory])
        actions = np.array([t[1] for t in sampled_trajectory])
        rewards = np.array([t[2] for t in sampled_trajectory])
        costs = np.array([t[3] for t in sampled_trajectory])
        dones = np.array([t[4] for t in sampled_trajectory])
        fails = np.array([t[5] for t in sampled_trajectory])
        next_states = np.array([t[6] for t in sampled_trajectory])

        states = obs_rms.normalize(states)
        rewards = reward_rms.normalize(rewards)
        next_states = obs_rms.normalize(next_states)
        costs = (1.0 - fails.reshape(-1, 1))*costs + fails.reshape(-1, 1)*costs/(1.0 - self.discount_factor)
        
        # convert to tensor
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, device = self.device, dtype = torch.float32)
        costs_tensor = torch.tensor(costs, device = self.device, dtype = torch.float32)
        dones_tensor = torch.tensor(dones, device = self.device, dtype = torch.float32)
        fails_tensor = torch.tensor(fails, device = self.device, dtype = torch.float32)
        next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)

        return states_tensor, actions_tensor, rewards_tensor, \
            costs_tensor, dones_tensor, fails_tensor, next_states_tensor

