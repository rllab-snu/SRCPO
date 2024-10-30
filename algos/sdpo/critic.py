from algos.common.critic_dist import CriticSDist
from algos.common.network_base import MLP

import numpy as np
import torch

class Critic(CriticSDist):
    def __init__(
            self, device:torch.device, 
            obs_dim:int, 
            action_dim:int, 
            critic_cfg:dict) -> None:
        self.action_dim = action_dim
        super().__init__(device, obs_dim, critic_cfg)

    ################
    # public methods
    ################

    def build(self) -> None:
        for critic_idx in range(self.n_critics):
            self.add_module(f"critic_{critic_idx}", MLP(
                input_size=(self.obs_dim + 2*self.action_dim), 
                output_size=self.n_quantiles,
                shape=self.critic_cfg['mlp']['shape'], 
                activation=self.activation,
                layer_norm=self.layer_norm,
                crelu=self.crelu,
            ))

    def forward(self, obs:torch.Tensor, mean:torch.Tensor, std:torch.Tensor) -> torch.Tensor:
        '''
        outputs: 
            batch_size x n_critics x n_quantiles
            or 
            n_critics x n_quantiles
        '''
        obs = torch.cat([obs, mean, std], dim=-1)
        quantiles = []
        for critic_idx in range(self.n_critics):
            x = self._modules[f"critic_{critic_idx}"](obs)
            x = self.rescale*self.last_activation(x)
            x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
            quantiles.append(x)
        x = torch.stack(quantiles, dim=-2)
        return x

    def getLoss(self, obs:torch.Tensor, mean:torch.Tensor, std:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        '''
        obs: batch_size x obs_dim
        target: batch_size x n_target_quantiles
        '''
        batch_size = obs.shape[0]
        n_target_quantiles = target.shape[-1]
        target = target.view(
            batch_size, 1, 1, n_target_quantiles)
        current_quantiles = self.forward(obs, mean, std).unsqueeze(-1) # (batch=B, n_critics=N, n_quantiles=M, 1)
        pairwise_delta = target - current_quantiles # B x N x M x kN
        critic_loss = torch.mean(
            pairwise_delta*(self.cdf - (pairwise_delta.detach() < 0).float()))
        return critic_loss

    def getLoss2(self, obs:torch.Tensor, mean:torch.Tensor, std:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        '''
        obs: batch_size x obs_dim
        target: batch_size x n_target_quantiles
        '''
        assert target.shape[-1] == 2*self.n_quantiles
        adjusted_target = 0.5*(target[:, ::2] + target[:, 1::2]).unsqueeze(-2) # (batch_size, 1, n_quantiles)
        current_quantiles = self.forward(obs, mean, std) # (batch_size, n_critics, n_quantiles)
        critic_loss = torch.mean(torch.square(adjusted_target - current_quantiles))
        return critic_loss
