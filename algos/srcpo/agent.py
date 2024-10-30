from algos.common.agent_base import AgentBase
from utils import cprint

from .optimizer import MultiTROptimizer
from .storage import ReplayBuffer
from .dist import TruncNormal
from .critic import Critic
from .actor import Actor

from typing import Tuple
import numpy as np
import torch
import os

EPS = 1e-8

def NormalKLDivLoss(mu1, log_std1, mu2, log_std2):
    std1 = torch.exp(log_std1)
    std2 = torch.exp(log_std2)
    kl_div = 2.0*(std1 - std2)**2 + (mu1 - mu2)**2 # approximated version
    return kl_div.mean()


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

        # for RL
        self.discount_factor = args.discount_factor
        self.n_update_steps = args.n_update_steps
        self.critic_lr = args.critic_lr
        self.beta_lr = args.beta_lr
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
        assert self.con_thresholds.shape[0] == self.cost_dim
        self.con_delta_etas = np.array(args.con_delta_etas) # (cost_dim, N)
        assert self.con_delta_etas.shape[0] == self.cost_dim
        self.con_alphas = np.array(args.con_alphas) # (cost_dim, N-1)
        assert self.con_alphas.shape[0] == self.cost_dim
        assert self.con_delta_etas.shape[1] == self.con_alphas.shape[1] + 1
        self.con_delta_alphas = np.concatenate(
            [self.con_alphas[:, 1:], np.ones((self.cost_dim, 1))], axis=-1) - self.con_alphas

        # to sample betas
        self.n_con_betas = self.con_alphas.shape[1] # N-1
        self.con_beta_dim = self.cost_dim*self.n_con_betas
        self.con_beta_scales = self.con_thresholds/np.sum(
            self.con_delta_alphas*np.cumsum(self.con_delta_etas[:, 1:], axis=-1), axis=-1)
        self.con_beta_clip_penalty = args.con_beta_clip_penalty
        self.con_beta_fix_std = args.__dict__.get('con_beta_fix_std', None)

        # for entropy
        self.con_entropy = args.con_entropy
        if self.con_entropy:
            self.con_ent_thresh = -args.con_ent_thresh*self.action_dim
            self.con_thresholds = np.append(self.con_thresholds, self.con_ent_thresh)

        # for model
        self.actor = Actor(
            self.device, self.obs_dim, self.con_beta_dim, self.action_dim, 
            self.action_bound_min, self.action_bound_max, self.model_cfg['actor']).to(self.device)
        self.reward_critic = Critic(
            self.device, self.obs_dim, self.con_beta_dim, 
            self.action_dim, self.model_cfg['reward_critic']).to(self.device)
        self.cost_critics = []
        for _ in range(self.cost_dim):
            self.cost_critics.append(Critic(
                self.device, self.obs_dim, self.con_beta_dim, 
                self.action_dim, self.model_cfg['cost_critic']).to(self.device))
        self.delta_beta_sampler = TruncNormal(
            self.con_beta_dim, self.con_beta_fix_std).to(self.device)

        # for replay buffer
        self.replay_buffer = ReplayBuffer(
            self.device, self.discount_factor, self.gae_coeff, self.n_target_quantiles, 
            self.n_envs, self.n_steps, self.n_update_steps, self.len_replay_buffer)

        # for optimizers
        self.intermediate_actor_optimizer = MultiTROptimizer(
            self.device, self.actor, self.damping_coeff, self.num_conjugate, 
            self.line_decay, self.max_kl, 1, self.con_thresholds, self.con_entropy)
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=self.critic_lr)
        self.cost_critic_optimizers = []
        for cost_idx in range(self.cost_dim):
            self.cost_critic_optimizers.append(
                torch.optim.Adam(self.cost_critics[cost_idx].parameters(), lr=self.critic_lr))
        self.con_beta_optimizer = torch.optim.Adam(self.delta_beta_sampler.parameters(), lr=self.beta_lr)

    ################
    # Public Methods
    ################

    def reset(self, observations:np.ndarray, env_idx:int=None) -> None:
        if env_idx is None:
            with torch.no_grad():
                sampled_delta_betas = self.delta_beta_sampler.sample()
                self.con_delta_betas_tensor = \
                    sampled_delta_betas.view((1, -1)).repeat((self.n_envs, 1)) # (n_envs, con_beta_dim)

    @torch.no_grad()
    def getAction(
        self, observation:np.ndarray, 
        discounted_cost_sums:np.ndarray, 
        discounts:np.ndarray, 
        deterministic:bool,
    ) -> np.ndarray:
        obs_tensor = torch.tensor(self.obs_rms.normalize(observation), dtype=torch.float32, device=self.device)
        epsilon_tensor = torch.randn(obs_tensor.shape[:-1] + (self.action_dim,), device=self.device)
        self.actor.updateActionDist(obs_tensor, self.con_delta_betas_tensor, epsilon_tensor)
        norm_action_tensor, unnorm_action_tensor = self.actor.sample(deterministic)
        normal_action_tensor = self.actor.getNormalAction()
        log_prob_tensor = self.actor.getLogProb()

        self.obs = observation.copy()
        self.con_delta_beta = self.con_delta_betas_tensor.detach().cpu().numpy()
        self.action = norm_action_tensor.detach().cpu().numpy()
        self.normal_action = normal_action_tensor.detach().cpu().numpy()
        self.log_prob = log_prob_tensor.detach().cpu().numpy()
        return unnorm_action_tensor.detach().cpu().numpy()

    def step(self, rewards:np.ndarray, costs:np.ndarray, dones:np.ndarray, 
             fails:np.ndarray, next_states:np.ndarray, 
             discounted_cost_sums:np.ndarray, discounts:np.ndarray) -> None:

        self.replay_buffer.addTransition(
            self.obs, self.action, self.normal_action, self.log_prob,
            rewards, costs, dones, fails, next_states, 
            discounted_cost_sums, discounts, self.con_delta_beta)

        # update statistics
        if self.norm_obs:
            self.obs_rms.update(self.obs)
        if self.norm_reward:
            self.reward_rms.update(rewards)

    def readyToTrain(self) -> bool:
        # update con_delta_beta
        with torch.no_grad():
            self.con_delta_betas_tensor[:] = \
                self.delta_beta_sampler.sample().view((1, -1)).repeat((self.n_envs, 1))
        return self.replay_buffer.getLen() >= self.n_update_steps

    def train(self) -> dict:
        # get batches
        states_tensor, betas_tensor, actions_tensor, reward_targets_tensor, \
        cost_targets_tensor_list, discounted_cost_sums_tensor_list, discounts_tensor = \
                self.replay_buffer.getBatches(
                    self.obs_rms, self.reward_rms, self.actor, 
                    self.reward_critic, self.cost_critics)

        # ================== Critic Update ================== #
        for _ in range(self.n_critic_iters):
            reward_critic_loss = self.reward_critic.getLoss(
                states_tensor, betas_tensor, actions_tensor, reward_targets_tensor)
            self.reward_critic_optimizer.zero_grad()
            reward_critic_loss.backward()
            self.reward_critic_optimizer.step()

            cost_critic_losses = []
            for cost_idx in range(self.cost_dim):
                cost_critic_loss = self.cost_critics[cost_idx].getLoss(
                    states_tensor, betas_tensor, actions_tensor, cost_targets_tensor_list[cost_idx])
                self.cost_critic_optimizers[cost_idx].zero_grad()
                cost_critic_loss.backward()
                self.cost_critic_optimizers[cost_idx].step()
                cost_critic_losses.append(cost_critic_loss)
            cost_critic_loss = torch.mean(torch.stack(cost_critic_losses, dim=0))
        # ================================================== #

        # ================= Policy Update ================= #
        objectives_tensor_list = []
        constraints_tensor_list = []
        delta_theta_list = []
        for beta_idx in range(self.n_update_steps//self.n_steps):
            # prepare for update
            with torch.no_grad():
                start_idx = beta_idx*self.n_steps
                end_idx = (beta_idx + 1)*self.n_steps
                actor_states_tensor = states_tensor[start_idx:end_idx] # (n_steps, obs_dim)
                delta_betas_tensor = betas_tensor[start_idx:end_idx] # (n_steps, con_beta_dim)

                # calculate con_beta and integral
                con_delta_betas = delta_betas_tensor[0].detach().cpu().numpy() \
                                    .reshape((self.cost_dim, self.n_con_betas))
                con_betas = np.cumsum(con_delta_betas, axis=-1) \
                                *self.con_beta_scales.reshape((-1, 1))
                conj_integral = np.zeros(self.cost_dim)
                for beta_idx in range(self.n_con_betas):
                    temp_sum = np.sum(self.con_delta_etas[:, 1:(beta_idx+2)]*con_betas[:, :(beta_idx+1)], axis=-1)
                    conj_integral += temp_sum*self.con_delta_alphas[:, beta_idx]

                # backup old policy
                epsilons_tensor = torch.randn_like(actions_tensor[start_idx:end_idx])
                self.actor.updateActionDist(actor_states_tensor, delta_betas_tensor, epsilons_tensor)
                old_actions_tensor = self.actor.sample(deterministic=False)[0]
                old_action_dists = self.actor.getDist()

                # get the current risk measure
                init_constraints_list = []
                for cost_idx in range(self.cost_dim):
                    cost_quantiles_tensor = self.cost_critics[cost_idx](
                        actor_states_tensor, delta_betas_tensor, old_actions_tensor).view((-1,))
                    con_val = torch.mean(cost_quantiles_tensor)*self.con_delta_etas[cost_idx, 0] + conj_integral[cost_idx]
                    for beta_idx in range(self.n_con_betas):
                        con_val += torch.mean(
                                torch.clamp(
                                    cost_quantiles_tensor - con_betas[cost_idx, beta_idx], min=0.0
                                )
                            )*self.con_delta_etas[cost_idx, 1 + beta_idx]
                    init_constraints_list.append(con_val)

            def get_obj_cons_kl() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                self.actor.updateActionDist(actor_states_tensor, delta_betas_tensor, epsilons_tensor)
                cur_actions = self.actor.sample(deterministic=False)[0]
                cur_action_dists = self.actor.getDist()
                objs = self.reward_critic(actor_states_tensor, delta_betas_tensor, cur_actions).mean().unsqueeze(0)
                cons = []
                batch_size = actor_states_tensor.shape[0]
                for cost_idx in range(self.cost_dim):
                    cost_quantiles_tensor = self.cost_critics[cost_idx](
                        actor_states_tensor, delta_betas_tensor, cur_actions).view((batch_size, -1)) # (B, M*N)
                    shifted_cost_quantiles_tensor = discounted_cost_sums_tensor_list[cost_idx][start_idx:end_idx].unsqueeze(-1) \
                                                        + discounts_tensor[start_idx:end_idx].unsqueeze(-1)*cost_quantiles_tensor
                    constraint = torch.mean(cost_quantiles_tensor)*self.con_delta_etas[cost_idx, 0]
                    for beta_idx in range(self.n_con_betas):
                        constraint += torch.mean(
                            (shifted_cost_quantiles_tensor >= con_betas[cost_idx, beta_idx]).detach().float() \
                            *cost_quantiles_tensor*self.con_delta_etas[cost_idx, 1 + beta_idx])
                    con_advantage = constraint - constraint.detach()
                    constraint = con_advantage/(1.0 - self.discount_factor) + init_constraints_list[cost_idx]
                    cons.append(constraint)
                cons = torch.stack(cons, dim=0)
                if self.con_entropy:
                    entropy = self.actor.getEntropy()
                    cons = torch.cat((cons, -entropy.unsqueeze(0)), dim=0)
                kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(
                    old_action_dists, cur_action_dists), dim=-1))
                return objs, cons, kl

            # update actor's gradient
            objectives, constraints, delta_theta \
                = self.intermediate_actor_optimizer.step(get_obj_cons_kl, actor_states_tensor, delta_betas_tensor)
            objectives_tensor_list.append(objectives)
            constraints_tensor_list.append(constraints)
            delta_theta_list.append(delta_theta)

        # get old actions and entropy
        with torch.no_grad():
            epsilons_tensor = torch.randn_like(actions_tensor)
            self.actor.updateActionDist(states_tensor, betas_tensor, epsilons_tensor)
            old_action_dists = self.actor.getDist()
            entropy = self.actor.getEntropy()

            # update actor's parameters
            init_theta = torch.cat([t.view(-1) for t in self.actor.parameters()]).clone().detach()
            delta_theta = torch.stack(delta_theta_list).mean(dim=0)
            self._applyParams(self.actor, init_theta - delta_theta)

        # get objectives and constraints
        with torch.no_grad():
            self.actor.updateActionDist(states_tensor, betas_tensor, epsilons_tensor)
            cur_action_dists = self.actor.getDist()
            kl = torch.mean(torch.sum(
                torch.distributions.kl.kl_divergence(
                    old_action_dists, cur_action_dists
                ), 
            dim=-1))
            obj_val = torch.stack(objectives_tensor_list, dim=0).mean()
            con_vals = torch.stack(constraints_tensor_list, dim=0).mean(dim=0)
        # ================================================= #

        # =========== beta distribution update =========== #
        with torch.no_grad():
            con_beta_scales_tensor = torch.tensor(self.con_beta_scales, dtype=torch.float32, device=self.device).view((-1, 1))
            con_delta_etas_tensor = torch.tensor(self.con_delta_etas, dtype=torch.float32, device=self.device)
            con_delta_alphas_tensor = torch.tensor(self.con_delta_alphas, dtype=torch.float32, device=self.device)

        beta_losses = []
        for beta_idx in range(self.n_update_steps//self.n_steps):
            with torch.no_grad():
                start_idx = beta_idx*self.n_steps
                end_idx = (beta_idx + 1)*self.n_steps
                beta_states_tensor = states_tensor[start_idx:end_idx] # (n_steps, obs_dim)
                beta_actions_tensor = actions_tensor[start_idx:end_idx] # (n_steps, action_dim)

            # sample beta
            sampled_beta = self.delta_beta_sampler.sample()
            sampled_betas = sampled_beta.unsqueeze(0).expand(beta_states_tensor.shape[0], -1)

            # calculate con_beta and integral
            con_delta_betas = sampled_beta.reshape((self.cost_dim, self.n_con_betas))
            con_betas = torch.cumsum(con_delta_betas, dim=-1)*con_beta_scales_tensor
            conj_integral = []
            for beta_idx in range(self.n_con_betas):
                temp_sum = (con_delta_etas_tensor[:, 1:(beta_idx+2)]*con_betas[:, :(beta_idx+1)]).sum(dim=-1)
                conj_integral.append(temp_sum*con_delta_alphas_tensor[:, beta_idx])
            conj_integral = torch.stack(conj_integral, dim=0).sum(dim=0)

            # calculate loss
            beta_loss = -self.reward_critic(beta_states_tensor, sampled_betas, beta_actions_tensor).mean()
            for cost_idx in range(self.cost_dim):
                cost_quantiles_tensor = self.cost_critics[cost_idx](
                    beta_states_tensor, sampled_betas, beta_actions_tensor).view((-1,))
                con_val = torch.mean(cost_quantiles_tensor)*self.con_delta_etas[cost_idx, 0] + conj_integral[cost_idx]
                for beta_idx in range(self.n_con_betas):
                    con_val += torch.mean(torch.clamp(
                        cost_quantiles_tensor - con_betas[cost_idx, beta_idx], min=0.0,
                        ))*self.con_delta_etas[cost_idx, 1 + beta_idx]
                beta_loss += self.con_beta_clip_penalty*torch.clamp(con_val - self.con_thresholds[cost_idx], min=0.0)
            beta_losses.append(beta_loss)

        # update
        beta_loss = torch.stack(beta_losses, dim=0).mean()
        self.con_beta_optimizer.zero_grad()
        beta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.delta_beta_sampler.parameters(), self.max_grad_norm)
        self.con_beta_optimizer.step()

        risk_beta_means, risk_beta_stds = self.delta_beta_sampler.getMeanStd()
        risk_beta_means = risk_beta_means.detach().cpu().numpy()
        risk_beta_stds = risk_beta_stds.detach().cpu().numpy()
        # ================================================ #

        # return
        train_results = {
            'objective': obj_val.item(),
            'constraints': con_vals.cpu().numpy(),
            'reward_critic_loss': reward_critic_loss.item(),
            'cost_critic_loss': cost_critic_loss.item(),
            'entropy': entropy.item(),
            'kl': kl.item(),
            'beta_loss': beta_loss.item(),
        }
        for beta_idx in range(self.con_beta_dim):
            train_results[f'risk_beta_mean{(beta_idx+1)}'] = risk_beta_means[beta_idx]
            train_results[f'risk_beta_std{(beta_idx+1)}'] = risk_beta_stds[beta_idx]
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
            'delta_beta_sampler': self.delta_beta_sampler.state_dict(),
            'con_beta_optimizer': self.con_beta_optimizer.state_dict(),
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
            self.delta_beta_sampler.load_state_dict(checkpoint['delta_beta_sampler'])
            self.con_beta_optimizer.load_state_dict(checkpoint['con_beta_optimizer'])
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

    def _applyGradParams(self, net, params):
        n = 0
        for p in net.parameters():
            numel = p.numel()
            p.grad = params[n:n + numel].view(p.shape)
            n += numel

    def _applyParams(self, net, params) -> None:
        n = 0
        for p in net.parameters():
            numel = p.numel()
            p.data.copy_(params[n:n + numel].view(p.shape))
            n += numel