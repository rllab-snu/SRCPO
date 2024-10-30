from algos.common.actor_squash import ActorSquash as Actor
from algos.common.critic_dist import CriticSDist as Critic
from algos.common.agent_base import AgentBase
from utils import cprint

from .storage import ReplayBuffer

from typing import Tuple
import numpy as np
import pickle
import torch
import os

EPS = 1e-8

class Agent(AgentBase):
    def __init__(self, args) -> None:
        super().__init__(
            name=args.name,
            device=args.device,
            # obs_dim=args.obs_dim + 2,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_reward,
        )

        # base
        self.save_dir = args.save_dir
        self.checkpoint_dir=f'{self.save_dir}/checkpoint'
        self.action_bound_min = args.action_bound_min
        self.action_bound_max = args.action_bound_max
        self.n_steps = args.n_steps
        self.n_envs = args.n_envs

        # for RL
        self.discount_factor = args.discount_factor
        self.critic_lr = args.critic_lr
        self.n_critic_iters = args.n_critic_iters
        self.max_grad_norm = args.max_grad_norm
        self.gae_coeff = args.gae_coeff
        self.n_target_quantiles = args.n_target_quantiles
        self.model_cfg = args.model

        # for trust region
        self.max_kl = args.max_kl
        self.kl_tolerance = args.kl_tolerance
        self.adaptive_lr_ratio = args.adaptive_lr_ratio
        self.clip_ratio = args.clip_ratio

        # for constraints
        self.con_threshold = args.con_threshold
        self.con_threshold /= (1.0 - self.discount_factor)
        self.con_alpha = args.con_alpha

        # for CVaR-CPO
        self.actor_lr = args.actor_lr
        self.n_actor_iters = args.n_actor_iters
        self.var_lr = args.var_lr
        self.con_lambda_lr = args.con_lambda_lr
        self.con_lambda_max = args.con_lambda_max

        # for model
        self.actor = Actor(
            self.device, self.obs_dim, self.action_dim, self.action_bound_min, 
            self.action_bound_max, self.model_cfg['actor']).to(self.device)
        self.reward_critic = Critic(
            self.device, self.obs_dim, self.model_cfg['reward_critic']).to(self.device)
        self.cost_critic = Critic(
            self.device, self.obs_dim, self.model_cfg['cost_critic']).to(self.device)
        self.var = 0.0
        self.con_lambda = 0.0
        self.stat_variables = self.var*np.ones(self.n_envs)

        # for replay buffer
        self.replay_buffer = ReplayBuffer(
            self.device, self.discount_factor, self.gae_coeff, 
            self.n_target_quantiles, self.n_envs, self.n_steps)
        # self.init_observation_list = []

        # for optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.reward_critic_optimizer = torch.optim.Adam(
            self.reward_critic.parameters(), lr=self.critic_lr)
        self.cost_critic_optimizer = torch.optim.Adam(
                self.cost_critic.parameters(), lr=self.critic_lr)

    ################
    # Public Methods
    ################

    @torch.no_grad()
    def getAction(self, observation:np.ndarray, discounted_cost_sums:np.ndarray, discounts:np.ndarray, deterministic:bool) -> np.ndarray:
        # state = np.concatenate([observation, discounted_cost_sums, discounts.reshape((-1, 1))], axis=-1)
        state = observation.copy()
        state_tensor = torch.tensor(self.obs_rms.normalize(state), dtype=torch.float32, device=self.device)
        epsilon_tensor = torch.randn(state_tensor.shape[:-1] + (self.action_dim,), device=self.device)
        self.actor.updateActionDist(state_tensor, epsilon_tensor)
        norm_action_tensor, unnorm_action_tensor = self.actor.sample(deterministic)
        normal_action_tensor = self.actor.getNormalAction()
        log_prob_tensor = self.actor.getLogProb()

        self.state = state.copy()
        self.action = norm_action_tensor.detach().cpu().numpy()
        self.normal_action = normal_action_tensor.detach().cpu().numpy()
        self.log_prob = log_prob_tensor.detach().cpu().numpy()
        return unnorm_action_tensor.detach().cpu().numpy()
    
    def reset(self, observations:np.ndarray, env_idx:int=None) -> None:
        if env_idx is None:
            self.stat_variables[:] = self.var*np.ones(self.n_envs)
            # for env_idx in range(self.n_envs):
            #     self.init_observation_list.append(observations[env_idx].copy())
        else:
            self.stat_variables[env_idx] = self.var
            # self.init_observation_list.append(observations[env_idx].copy())

    def step(self, rewards:np.ndarray, costs:np.ndarray, dones:np.ndarray, 
             fails:np.ndarray, next_states:np.ndarray, 
             discounted_cost_sums:np.ndarray, discounts:np.ndarray) -> None:

        # add transition to replay buffer
        next_stat_variables = (self.stat_variables - costs.flatten())/self.discount_factor
        self.replay_buffer.addTransition(
            self.state, self.action, self.normal_action, self.log_prob,
            rewards, costs.flatten(), dones, fails, 
            # np.concatenate([next_states, discounted_cost_sums, discounts.reshape((-1, 1))], axis=-1), 
            next_states,
            self.stat_variables.copy(), next_stat_variables.copy(),
        )

        # update statistic variables
        self.stat_variables[:] = next_stat_variables

        # update statistics
        if self.norm_obs:
            self.obs_rms.update(self.state)
        if self.norm_reward:
            self.reward_rms.update(rewards)

    def readyToTrain(self) -> bool:
        return self.replay_buffer.getLen() >= self.n_steps

    def train(self) -> dict:
        # get batches
        states_tensor, actions_tensor, normal_actions_tensor, \
        reward_targets_tensor, cost_targets_tensor, reward_gaes_tensor, cost_gaes_tensor = \
            self.replay_buffer.getBatches(
                self.obs_rms, self.reward_rms, self.actor, self.reward_critic, self.cost_critic)
        # init_states = np.array(self.init_observation_list)
        # init_states = np.concatenate([init_states, np.zeros((init_states.shape[0], 1)), np.ones((init_states.shape[0], 1))], axis=-1)
        # init_states_tensor = torch.tensor(self.obs_rms.normalize(init_states), dtype=torch.float32, device=self.device)
        

        # ================== Critic Update ================== #
        for _ in range(self.n_critic_iters):
            reward_critic_loss = self.reward_critic.getLoss(
                states_tensor, reward_targets_tensor)
            self.reward_critic_optimizer.zero_grad()
            reward_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.reward_critic.parameters(), self.max_grad_norm)
            self.reward_critic_optimizer.step()

            cost_critic_loss = self.cost_critic.getLoss(
                states_tensor, cost_targets_tensor)
            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.cost_critic.parameters(), self.max_grad_norm)
            self.cost_critic_optimizer.step()
        # ================================================== #

        # ================= Policy Update ================= #
        # backup old policy
        with torch.no_grad():
            epsilons_tensor = torch.randn_like(actions_tensor)
            self.actor.updateActionDist(states_tensor, epsilons_tensor)
            old_action_dists = self.actor.getDist()
            old_log_probs_tensor = self.actor.getLogProb(normal_actions_tensor)
            reduced_gaes_tensor = reward_gaes_tensor - self.con_lambda*cost_gaes_tensor
            reduced_gaes_tensor = (reduced_gaes_tensor - reduced_gaes_tensor.mean())/(reduced_gaes_tensor.std() + EPS)

        for _ in range(self.n_actor_iters):
            self.actor.updateActionDist(states_tensor, epsilons_tensor)
            cur_action_dists = self.actor.getDist()
            kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_action_dists, cur_action_dists), dim=-1))
            if kl > self.max_kl*self.kl_tolerance: break
            cur_log_probs_tensor = self.actor.getLogProb(normal_actions_tensor)
            prob_ratios_tensor = torch.exp(cur_log_probs_tensor - old_log_probs_tensor)
            clipped_ratios_tensor = torch.clamp(prob_ratios_tensor, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio)
            actor_loss = -torch.mean(torch.minimum(reduced_gaes_tensor*prob_ratios_tensor, reduced_gaes_tensor*clipped_ratios_tensor))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        self.actor.updateActionDist(states_tensor, epsilons_tensor)
        cur_action_dists = self.actor.getDist()
        kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_action_dists, cur_action_dists), dim=-1))
        entropy = self.actor.getEntropy()

        # adjust learning rate based on KL divergence
        if kl > self.max_kl*self.kl_tolerance:
            self.actor_lr /= self.adaptive_lr_ratio
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.actor_lr
        elif kl < self.max_kl/self.kl_tolerance:
            self.actor_lr *= self.adaptive_lr_ratio
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.actor_lr
        # ================================================= #

        # update con lambda
        # cost_quantiles_tensor = self.cost_critic(init_states_tensor).view(init_states_tensor.shape[0], -1)
        cost_quantiles_tensor = self.cost_critic(states_tensor).view(states_tensor.shape[0], -1)
        stat_variables_tensor = torch.ones_like(cost_quantiles_tensor)*self.var
        constraint = ((cost_quantiles_tensor - stat_variables_tensor)*(cost_quantiles_tensor >= stat_variables_tensor).float()).mean()
        con_val = self.var + constraint.item()/self.con_alpha
        self.con_lambda += self.con_lambda_lr*(con_val - self.con_threshold)
        self.con_lambda = np.clip(self.con_lambda, 0.0, self.con_lambda_max)

        # update VaR
        self.var -= self.var_lr*(self.con_alpha - (cost_quantiles_tensor >= self.var).float().mean().item())
        self.var = np.clip(self.var, 0.0, 1.0/(1.0 - self.discount_factor))

        # return
        train_results = {
            'actor_loss': actor_loss.item(),
            'constraints': [con_val],
            'reward_critic_loss': reward_critic_loss.item(),
            'cost_critic_loss': cost_critic_loss.item(),
            'entropy': entropy.item(),
            'kl': kl.item(),
            'VaR': self.var,
            'con_lambda': self.con_lambda,
        }
        return train_results

    def save(self, model_num):
        # save rms
        self.obs_rms.save(self.save_dir, model_num)
        self.reward_rms.save(self.save_dir, model_num)

        # save network models
        save_dict = {
            'actor': self.actor.state_dict(),
            'reward_critic': self.reward_critic.state_dict(),
            'reward_critic_optimizer': self.reward_critic_optimizer.state_dict(),            
        }
        save_dict['cost_critic'] = self.cost_critic.state_dict()
        save_dict['cost_critic_optimizer'] = self.cost_critic_optimizer.state_dict()
        torch.save(save_dict, f"{self.checkpoint_dir}/model_{model_num}.pt")
        with open(f"{self.save_dir}/others_{model_num}.pkl", 'wb') as f:
            pickle.dump([self.var, self.con_lambda], f)
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def load(self, model_num):
        # load rms
        self.obs_rms.load(self.save_dir, model_num)
        self.reward_rms.load(self.save_dir, model_num)

        # load network models
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model_{model_num}.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.reward_critic.load_state_dict(checkpoint['reward_critic'])
            self.reward_critic_optimizer.load_state_dict(checkpoint['reward_critic_optimizer'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic'])
            self.cost_critic_optimizer.load_state_dict(checkpoint['cost_critic_optimizer'])
            with open(f"{self.save_dir}/others_{model_num}.pkl", 'rb') as f:
                self.var, self.con_lambda = pickle.load(f)
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
            return int(model_num)
        else:
            self.actor.initialize()
            self.reward_critic.initialize()
            self.cost_critic.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
            return 0
