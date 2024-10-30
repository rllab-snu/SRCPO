from algos.common.critic_base import CriticS, CriticSA
from algos.common.network_base import MLP, initWeights

import numpy as np
import torch

class CriticSADist(CriticSA):
    def __init__(
            self, device:torch.device, 
            obs_dim:int, 
            action_dim:int, 
            critic_cfg:dict) -> None:
        self.n_critics = critic_cfg['n_critics']
        self.n_quantiles = critic_cfg['n_quantiles']
        self.layer_norm = critic_cfg['mlp'].get('layer_norm', False)
        self.crelu = critic_cfg['mlp'].get('crelu', False)
        self.rescale = critic_cfg.get('rescale', 1.0)
        super().__init__(device, obs_dim, action_dim, critic_cfg)

        # calculate cdf
        with torch.no_grad():
            cdf = (torch.arange(
                self.n_quantiles, device=self.device, dtype=torch.float32)
                + 0.5)/self.n_quantiles
            self.cdf = cdf.view(1, 1, -1, 1) # 1 x 1 x M x 1

    ################
    # public methods
    ################

    def build(self) -> None:
        for critic_idx in range(self.n_critics):
            self.add_module(f"critic_{critic_idx}", MLP(
                input_size=self.obs_dim + self.action_dim, 
                output_size=self.n_quantiles,
                shape=self.critic_cfg['mlp']['shape'], 
                activation=self.activation,
                layer_norm=self.layer_norm,
                crelu=self.crelu,
            ))

    def forward(self, obs:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        '''
        outputs: 
            batch_size x n_critics x n_quantiles
            or 
            n_critics x n_quantiles
        '''
        concat_x = torch.cat([obs, action], dim=-1)
        quantiles = []
        for critic_idx in range(self.n_critics):
            x = self._modules[f"critic_{critic_idx}"](concat_x)
            x = self.rescale*self.last_activation(x)
            x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
            quantiles.append(x)
        x = torch.stack(quantiles, dim=-2)
        return x

    def getLoss(self, obs:torch.Tensor, action:torch.Tensor, target:torch.Tensor, huber_coeff=0.0) -> torch.Tensor:
        '''
        obs: batch_size x obs_dim
        action: batch_size x action_dim
        target: batch_size x n_target_quantiles
        '''
        batch_size = obs.shape[0]
        n_target_quantiles = target.shape[-1]
        target = target.view(
            batch_size, 1, 1, n_target_quantiles)
        current_quantiles = self.forward(obs, action).unsqueeze(-1) # (batch=B, n_critics=N, n_quantiles=M, 1)
        pairwise_delta = target - current_quantiles # B x N x M x kN
        if huber_coeff <= 0:
            critic_loss = torch.mean(
                pairwise_delta*(self.cdf - (pairwise_delta.detach() < 0).float())
            )
        else:
            abs_pairwise_delta = torch.abs(pairwise_delta)
            huber_terms = torch.where(
                abs_pairwise_delta > huber_coeff, 
                huber_coeff*(abs_pairwise_delta - huber_coeff/2.0), 
                pairwise_delta**2 * 0.5
            )
            critic_loss = torch.mean(
                torch.abs(self.cdf - (pairwise_delta.detach() < 0).float()) * huber_terms
            )
        return critic_loss

    def getLoss2(self, obs:torch.Tensor, action:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        '''
        obs: batch_size x obs_dim
        action: batch_size x action_dim
        target: batch_size x n_target_quantiles
        '''
        assert target.shape[-1] == 2*self.n_quantiles
        adjusted_target = 0.5*(target[:, ::2] + target[:, 1::2]).unsqueeze(-2) # (batch_size, 1, n_quantiles)
        current_quantiles = self.forward(obs, action) # (batch_size, n_critics, n_quantiles)
        critic_loss = torch.mean(torch.square(adjusted_target - current_quantiles))
        return critic_loss


class CriticSDist(CriticS):
    def __init__(
            self, device:torch.device, 
            obs_dim:int, 
            critic_cfg:dict) -> None:
        self.n_critics = critic_cfg['n_critics']
        self.n_quantiles = critic_cfg['n_quantiles']
        self.layer_norm = critic_cfg['mlp'].get('layer_norm', False)
        self.crelu = critic_cfg['mlp'].get('crelu', False)
        self.rescale = critic_cfg.get('rescale', 1.0)
        super().__init__(device, obs_dim, critic_cfg)

        # calculate cdf
        with torch.no_grad():
            cdf = (torch.arange(
                self.n_quantiles, device=self.device, dtype=torch.float32)
                + 0.5)/self.n_quantiles
            self.cdf = cdf.view(1, 1, -1, 1) # 1 x 1 x M x 1

    ################
    # public methods
    ################

    def build(self) -> None:
        for critic_idx in range(self.n_critics):
            self.add_module(f"critic_{critic_idx}", MLP(
                input_size=self.obs_dim, 
                output_size=self.n_quantiles,
                shape=self.critic_cfg['mlp']['shape'], 
                activation=self.activation,
                layer_norm=self.layer_norm,
                crelu=self.crelu,
            ))

    def forward(self, obs:torch.Tensor) -> torch.Tensor:
        '''
        outputs: 
            batch_size x n_critics x n_quantiles
            or 
            n_critics x n_quantiles
        '''
        quantiles = []
        for critic_idx in range(self.n_critics):
            x = self._modules[f"critic_{critic_idx}"](obs)
            x = self.rescale*self.last_activation(x)
            x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
            quantiles.append(x)
        x = torch.stack(quantiles, dim=-2)
        return x

    def getLoss(self, obs:torch.Tensor, target:torch.Tensor, huber_coeff=0.0) -> torch.Tensor:
        '''
        obs: batch_size x obs_dim
        target: batch_size x n_target_quantiles
        '''
        batch_size = obs.shape[0]
        n_target_quantiles = target.shape[-1]
        target = target.view(
            batch_size, 1, 1, n_target_quantiles)
        current_quantiles = self.forward(obs).unsqueeze(-1) # (batch=B, n_critics=N, n_quantiles=M, 1)
        pairwise_delta = target - current_quantiles # B x N x M x kN
        if huber_coeff <= 0:
            critic_loss = torch.mean(
                pairwise_delta*(self.cdf - (pairwise_delta.detach() < 0).float())
            )
        else:
            abs_pairwise_delta = torch.abs(pairwise_delta)
            huber_terms = torch.where(
                abs_pairwise_delta > huber_coeff, 
                huber_coeff*(abs_pairwise_delta - huber_coeff/2.0), 
                pairwise_delta**2 * 0.5
            )
            critic_loss = torch.mean(
                torch.abs(self.cdf - (pairwise_delta.detach() < 0).float()) * huber_terms
            )
        return critic_loss

    def getLoss2(self, obs:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        '''
        obs: batch_size x obs_dim
        target: batch_size x n_target_quantiles
        '''
        assert target.shape[-1] == 2*self.n_quantiles
        adjusted_target = 0.5*(target[:, ::2] + target[:, 1::2]).unsqueeze(-2) # (batch_size, 1, n_quantiles)
        current_quantiles = self.forward(obs) # (batch_size, n_critics, n_quantiles)
        critic_loss = torch.mean(torch.square(adjusted_target - current_quantiles))
        return critic_loss
