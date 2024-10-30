from algos.common.actor_squash import ActorSquash as Actor
from algos.common.critic_dist import CriticSADist as Critic
from algos.common.agent_base import AgentBase
from utils import cprint

from .optimizer import SDACOptimizer
from .storage import ReplayBuffer

from scipy.stats import norm
from typing import Tuple
import numpy as np
import torch
import os

EPS = 1e-8

class Agent(AgentBase):
    def __init__(self, args) -> None:
        super().__init__(
            name=args.name,
            device=args.device,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_reward,
        )

        # base
        self.save_dir = args.save_dir
        self.checkpoint_dir=f'{self.save_dir}/checkpoint'
        self.cost_dim = args.cost_dim
        self.action_bound_min = args.action_bound_min
        self.action_bound_max = args.action_bound_max
        self.n_steps = args.n_steps
        self.n_envs = args.n_envs
        self.n_cons = self.cost_dim

        # for RL
        self.discount_factor = args.discount_factor
        self.n_update_steps = args.n_update_steps
        self.critic_lr = args.critic_lr
        self.n_critic_iters = args.n_critic_iters
        self.max_grad_norm = args.max_grad_norm
        self.gae_coeff = args.gae_coeff
        self.len_replay_buffer = args.len_replay_buffer
        self.n_target_quantiles = args.n_target_quantiles
        self.model_cfg = args.model

        # for trust region
        self.max_kl = args.max_kl
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.line_decay = args.line_decay

        # for constraints
        self.con_thresholds = np.array(args.con_thresholds)
        self.con_thresholds /= (1.0 - self.discount_factor)
        self.con_zeta = np.min(self.con_thresholds)
        self.con_alphas = np.array(args.con_alphas)
        self.con_sigmas = norm.pdf(norm.ppf(self.con_alphas))/self.con_alphas
        assert len(self.con_thresholds) == self.cost_dim
        assert len(self.con_alphas) == self.cost_dim

        # for entropy
        self.con_entropy = args.con_entropy
        if self.con_entropy:
            self.con_ent_thresh = -args.con_ent_thresh*self.action_dim
            self.con_thresholds = np.append(self.con_thresholds, self.con_ent_thresh)
            self.n_cons += 1

        # for model
        self.actor = Actor(
            self.device, self.obs_dim, self.action_dim, self.action_bound_min, 
            self.action_bound_max, self.model_cfg['actor']).to(self.device)
        self.reward_critic = Critic(
            self.device, self.obs_dim, self.action_dim, 
            self.model_cfg['reward_critic']).to(self.device)
        self.cost_critics = []
        for _ in range(self.cost_dim):
            self.cost_critics.append(Critic(
                self.device, self.obs_dim, self.action_dim, 
                self.model_cfg['cost_critic']).to(self.device))

        # for replay buffer
        self.replay_buffer = ReplayBuffer(
            self.device, self.discount_factor, self.gae_coeff, self.n_target_quantiles, 
            self.n_envs, self.n_steps, self.n_update_steps, self.len_replay_buffer)

        # for optimizers
        self.actor_optimizer = SDACOptimizer(
            self.device, self.actor, self.damping_coeff, self.num_conjugate, 
            self.line_decay, self.max_kl, self.con_thresholds, 
            self.con_zeta, self.con_entropy)
        self.reward_critic_optimizer = torch.optim.Adam(
            self.reward_critic.parameters(), lr=self.critic_lr)
        self.cost_critic_optimizers = []
        for cost_idx in range(self.cost_dim):
            self.cost_critic_optimizers.append(
                torch.optim.Adam(
                    self.cost_critics[cost_idx].parameters(), 
                    lr=self.critic_lr))

    ################
    # Public Methods
    ################

    @torch.no_grad()
    def getAction(self, state:np.ndarray, discounted_cost_sums:np.ndarray, discounts:np.ndarray, deterministic:bool) -> np.ndarray:
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

    def step(self, rewards:np.ndarray, costs:np.ndarray, dones:np.ndarray, 
             fails:np.ndarray, next_states:np.ndarray, 
             discounted_cost_sums:np.ndarray, discounts:np.ndarray) -> None:

        self.replay_buffer.addTransition(
            self.state, self.action, self.normal_action, self.log_prob,
            rewards, costs, dones, fails, next_states)

        # update statistics
        if self.norm_obs:
            self.obs_rms.update(self.state)
        if self.norm_reward:
            self.reward_rms.update(rewards)

    def readyToTrain(self) -> bool:
        return self.replay_buffer.getLen() >= self.n_update_steps

    def train(self) -> dict:
        # get batches
        states_tensor, actions_tensor, reward_targets_tensor, cost_targets_tensor_list = \
            self.replay_buffer.getBatches(
                self.obs_rms, self.reward_rms, self.actor, self.reward_critic, self.cost_critics)

        # ================== Critic Update ================== #
        for _ in range(self.n_critic_iters):
            reward_critic_loss = self.reward_critic.getLoss(
                states_tensor, actions_tensor, reward_targets_tensor)
            self.reward_critic_optimizer.zero_grad()
            reward_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.reward_critic.parameters(), self.max_grad_norm)
            self.reward_critic_optimizer.step()

            cost_critic_losses = []
            for cost_idx in range(self.cost_dim):
                cost_critic_loss = self.cost_critics[cost_idx].getLoss(
                    states_tensor, actions_tensor, cost_targets_tensor_list[cost_idx])
                self.cost_critic_optimizers[cost_idx].zero_grad()
                cost_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.cost_critics[cost_idx].parameters(), self.max_grad_norm)
                self.cost_critic_optimizers[cost_idx].step()
                cost_critic_losses.append(cost_critic_loss)
            cost_critic_loss = torch.mean(torch.stack(cost_critic_losses, dim=0))
        # ================================================== #

        # ================= Policy Update ================= #
        # backup old policy
        with torch.no_grad():
            epsilons_tensor = torch.randn_like(actions_tensor)
            self.actor.updateActionDist(states_tensor, epsilons_tensor)
            old_action_dists = self.actor.getDist()
            entropy_tensor = self.actor.getEntropy()

        def get_obj_cons_kl() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.actor.updateActionDist(states_tensor, epsilons_tensor)
            cur_actions = self.actor.sample(deterministic=False)[0]
            cur_action_dists = self.actor.getDist()

            obj = self._getObjective(states_tensor, cur_actions)
            cons = self._getConstraints(states_tensor, cur_actions)
            if self.con_entropy:
                entropy = self.actor.getEntropy()
                cons = torch.cat((cons, -entropy.unsqueeze(0)), dim=0)
            kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(
                old_action_dists, cur_action_dists), dim=-1))
            return obj, cons, kl

        objective, constraints, kl, safety_mode = \
            self.actor_optimizer.step(get_obj_cons_kl)
        # ================================================= #

        # return
        train_results = {
            'objective': objective.item(),
            'constraints': constraints.detach().cpu().numpy(),
            'reward_critic_loss': reward_critic_loss.item(),
            'cost_critic_loss': cost_critic_loss.item(),
            'entropy': entropy_tensor.item(),
            'kl': kl.item(),
            'safety_mode': safety_mode,
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
        for cost_idx in range(self.cost_dim):
            save_dict[f'cost_critic_{cost_idx}'] = self.cost_critics[cost_idx].state_dict()
            save_dict[f'cost_critic_optimizer_{cost_idx}'] = self.cost_critic_optimizers[cost_idx].state_dict()
        torch.save(save_dict, f"{self.checkpoint_dir}/model_{model_num}.pt")
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
            for cost_idx in range(self.cost_dim):
                self.cost_critics[cost_idx].load_state_dict(checkpoint[f'cost_critic_{cost_idx}'])
                self.cost_critic_optimizers[cost_idx].load_state_dict(checkpoint[f'cost_critic_optimizer_{cost_idx}'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
            return int(model_num)
        else:
            self.actor.initialize()
            self.reward_critic.initialize()
            for cost_idx in range(self.cost_dim):
                self.cost_critics[cost_idx].initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
            return 0

    #################
    # Private Methods
    #################

    def _getObjective(self, states:torch.Tensor, cur_actions:torch.Tensor) -> torch.Tensor:
        objective = torch.mean(self.reward_critic(states, cur_actions))
        return objective

    def _getConstraints(self, states:torch.Tensor, cur_actions:torch.Tensor) -> torch.Tensor:
        constraints = []
        for cost_idx in range(self.cost_dim):
            cost_mean = torch.mean(self.cost_critics[cost_idx](states, cur_actions))
            cost_std = torch.std(self.cost_critics[cost_idx](states, cur_actions))
            constraint = cost_mean + self.con_sigmas[cost_idx]*cost_std
            constraints.append(constraint)
        return torch.stack(constraints, dim=0)
