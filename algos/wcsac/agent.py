from algos.common.actor_squash import ActorSquash as Actor
from algos.common.critic_dist import CriticSADist as Critic
from algos.common.agent_base import AgentBase
from utils import cprint

from .storage import ReplayBuffer

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

        # for RL
        self.discount_factor = args.discount_factor
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.batch_size = args.batch_size
        self.n_update_iters = args.n_update_iters
        self.max_grad_norm = args.max_grad_norm
        self.len_replay_buffer = args.len_replay_buffer
        self.soft_update_ratio = args.soft_update_ratio
        self.target_entropy = self.action_dim*args.entropy_threshold
        self.model_cfg = args.model

        # for constraints
        self.con_thresholds = np.array(args.con_thresholds)/(1.0 - self.discount_factor)
        self.con_alphas = np.array(args.con_alphas)
        assert len(self.con_thresholds) == self.cost_dim
        assert len(self.con_alphas) == self.cost_dim
        self.con_thresholds = torch.tensor(self.con_thresholds, dtype=torch.float32, device=self.device)
        self.pre_con_lambdas = torch.zeros(
            self.cost_dim, dtype=torch.float32, device=self.device, requires_grad=True)
        # self.getConLambdas = lambda: torch.nn.functional.softplus(self.pre_con_lambdas)
        self.getConLambdas = lambda: torch.exp(self.pre_con_lambdas)
        self.con_lambdas_lr = args.con_lambdas_lr
        self.log_std_fix = args.model['actor']['log_std_fix']
        self.pre_entropy_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.getEntropyAlpha = lambda: torch.exp(self.pre_entropy_alpha)
        self.ent_alpha_lr = args.ent_alpha_lr

        # for replay buffer
        self.replay_buffer = ReplayBuffer(
            self.len_replay_buffer, self.batch_size, self.discount_factor, self.device)

        # for model
        self.actor = Actor(
            self.device, self.obs_dim, self.action_dim, self.action_bound_min, 
            self.action_bound_max, self.model_cfg['actor']).to(self.device)
        self.reward_critic = Critic(
            self.device, self.obs_dim, self.action_dim, 
            self.model_cfg['reward_critic']).to(self.device)
        self.target_reward_critic = Critic(
            self.device, self.obs_dim, self.action_dim, 
            self.model_cfg['reward_critic']).to(self.device)
        self.cost_critics = []
        self.target_cost_critics = []
        for _ in range(self.cost_dim):
            self.cost_critics.append(Critic(
                self.device, self.obs_dim, self.action_dim, 
                self.model_cfg['cost_critic']).to(self.device))
            self.target_cost_critics.append(Critic(
                self.device, self.obs_dim, self.action_dim, 
                self.model_cfg['cost_critic']).to(self.device))

        # for optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.reward_critic_optimizer = torch.optim.Adam(
            self.reward_critic.parameters(), lr=self.critic_lr)
        self.cost_critic_optimizers = []
        for cost_idx in range(self.cost_dim):
            self.cost_critic_optimizers.append(
                torch.optim.Adam(
                    self.cost_critics[cost_idx].parameters(), 
                    lr=self.critic_lr))
        self.entropy_alpha_optimizer = torch.optim.Adam([self.pre_entropy_alpha], lr=self.ent_alpha_lr)
        self.con_lambdas_optimizer = torch.optim.Adam([self.pre_con_lambdas], lr=self.con_lambdas_lr)

    ################
    # Public Methods
    ################

    @torch.no_grad()
    def getAction(self, observation:np.ndarray, discounted_cost_sums:np.ndarray, discounts:np.ndarray, deterministic:bool) -> np.ndarray:
        state_tensor = torch.tensor(self.obs_rms.normalize(observation), dtype=torch.float32, device=self.device)
        epsilon_tensor = torch.randn(state_tensor.shape[:-1] + (self.action_dim,), device=self.device)
        self.actor.updateActionDist(state_tensor, epsilon_tensor)
        norm_action_tensor, unnorm_action_tensor = self.actor.sample(deterministic)

        self.state = observation.copy()
        self.action = norm_action_tensor.detach().cpu().numpy()
        return unnorm_action_tensor.detach().cpu().numpy()

    def step(self, rewards:np.ndarray, costs:np.ndarray, dones:np.ndarray, 
             fails:np.ndarray, next_states:np.ndarray, 
             discounted_cost_sums:np.ndarray, discounts:np.ndarray) -> None:

        self.replay_buffer.addTransition(
            self.state, self.action, rewards, costs, dones, fails, next_states)

        # update statistics
        if self.norm_obs:
            self.obs_rms.update(self.state)
        if self.norm_reward:
            self.reward_rms.update(rewards)

    def readyToTrain(self) -> bool:
        return self.replay_buffer.getLen() >= self.batch_size

    def train(self) -> dict:
        for _ in range(self.n_update_iters):
            states_tensor, actions_tensor, rewards_tensor, costs_tensor, \
            dones_tensor, fails_tensor, next_states_tensor = \
                self.replay_buffer.getBatches(self.obs_rms, self.reward_rms)

            # ================== Critic Update ================== #
            # calculate critic targets
            with torch.no_grad():
                batch_size = states_tensor.shape[0]
                entropy_alpha_tensor = self.getEntropyAlpha()
                epsilons_tensor = torch.randn_like(actions_tensor)
                self.actor.updateActionDist(next_states_tensor, epsilons_tensor)
                next_actions_tensor = self.actor.sample(deterministic=False)[0]
                next_log_probs_tensor = self.actor.getLogProb() # (batch_size,)
                next_reward_quantiles_tensor = self.target_reward_critic(
                    next_states_tensor, next_actions_tensor).view(batch_size, -1) # (batch_size, n_critics*n_quantiles)
                next_reward_quantiles_tensor = torch.sort(next_reward_quantiles_tensor, dim=-1)[0]
                if self.log_std_fix:
                    reward_targets_tensor = rewards_tensor.view(-1, 1) + self.discount_factor*(1.0 - fails_tensor.view(-1, 1)) * \
                            next_reward_quantiles_tensor # (batch_size, n_critic*n_quantiles)
                else:
                    reward_targets_tensor = rewards_tensor.view(-1, 1) + self.discount_factor*(1.0 - fails_tensor.view(-1, 1)) * \
                            (next_reward_quantiles_tensor - entropy_alpha_tensor*next_log_probs_tensor.view(-1, 1)) # (batch_size, n_critic*n_quantiles)
                cost_targets_tensor_list = []
                for cost_idx in range(self.cost_dim):
                    next_cost_quantiles_tensor = self.target_cost_critics[cost_idx](
                        next_states_tensor, next_actions_tensor).view(batch_size, -1) # (batch_size, n_critics*n_quantiles)
                    next_cost_quantiles_tensor = torch.sort(next_cost_quantiles_tensor, dim=-1)[0]
                    cost_targets_tensor = costs_tensor[:, cost_idx:cost_idx+1] \
                        + self.discount_factor*(1.0 - fails_tensor.view(-1, 1))*next_cost_quantiles_tensor
                    cost_targets_tensor_list.append(cost_targets_tensor)

            # reward critic update
            reward_critic_loss = self.reward_critic.getLoss(states_tensor, actions_tensor, reward_targets_tensor)
            # reward_critic_loss = self.reward_critic.getLoss(
            #     states_tensor, actions_tensor, reward_targets_tensor, huber_coeff=1.0)
            self.reward_critic_optimizer.zero_grad()
            reward_critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.reward_critic.parameters(), self.max_grad_norm)
            self.reward_critic_optimizer.step()

            # soft update
            self._softUpdate(self.target_reward_critic, self.reward_critic, self.soft_update_ratio)

            for cost_idx in range(self.cost_dim):
                # cost critic update
                cost_critic_loss = self.cost_critics[cost_idx].getLoss(states_tensor, actions_tensor, cost_targets_tensor_list[cost_idx])
                # cost_critic_loss = self.cost_critics[cost_idx].getLoss(
                #     states_tensor, actions_tensor, cost_targets_tensor_list[cost_idx], huber_coeff=1.0)
                self.cost_critic_optimizers[cost_idx].zero_grad()
                cost_critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.cost_critics[cost_idx].parameters(), self.max_grad_norm)
                self.cost_critic_optimizers[cost_idx].step()

                # soft update
                self._softUpdate(self.target_cost_critics[cost_idx], self.cost_critics[cost_idx], self.soft_update_ratio)
            # ================================================== #

            # ================= Policy Update ================= #
            with torch.no_grad():
                batch_size = states_tensor.shape[0]
                con_lambdas_tensor = self.getConLambdas()
                entropy_alpha_tensor = self.getEntropyAlpha()
                epsilons_tensor = torch.randn_like(actions_tensor)

            self.actor.updateActionDist(states_tensor, epsilons_tensor)
            sampled_actions_tensor = self.actor.sample(deterministic=False)[0]
            entropy = self.actor.getEntropy()
            objective = self.reward_critic(states_tensor, sampled_actions_tensor).mean()
            constraints = []
            for cost_idx in range(self.cost_dim):
                cost_qunatiles_tensor = self.cost_critics[cost_idx](states_tensor, sampled_actions_tensor).view(batch_size, -1)
                with torch.no_grad():
                    sorted_cost_qunatiles_tensor = torch.sort(cost_qunatiles_tensor, dim=-1)[0]
                    VaR_idx = int(np.clip(
                        (1.0 - self.con_alphas[cost_idx])*cost_qunatiles_tensor.shape[-1], 
                        0, cost_qunatiles_tensor.shape[-1] - 1))
                    VaR = sorted_cost_qunatiles_tensor[:, VaR_idx:VaR_idx+1]
                constraint = (VaR + (cost_qunatiles_tensor - VaR).clamp(min=0.0)/self.con_alphas[cost_idx]).mean()
                constraints.append(constraint)
            constraints = torch.stack(constraints, dim=-1)
            if self.log_std_fix:
                actor_loss = -objective + torch.dot(con_lambdas_tensor, constraints)
            else:
                actor_loss = -(objective + entropy_alpha_tensor*entropy) + torch.dot(con_lambdas_tensor, constraints)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            # ================================================= #

            # entropy alpha update
            entropy_alpha_tensor = self.getEntropyAlpha()
            if not self.log_std_fix:
                entropy_alpha_loss = torch.mean(entropy_alpha_tensor*(entropy - self.target_entropy).detach())
                self.entropy_alpha_optimizer.zero_grad()
                entropy_alpha_loss.backward()
                self.entropy_alpha_optimizer.step()
                self.pre_entropy_alpha.data.copy_(self.pre_entropy_alpha.data.clamp(-8.0, 8.0))

            # constraint lambda update
            con_lambdas_tensor = self.getConLambdas()
            con_lambdas_loss = torch.dot(con_lambdas_tensor, self.con_thresholds - constraints.detach())
            self.con_lambdas_optimizer.zero_grad()
            con_lambdas_loss.backward()
            self.con_lambdas_optimizer.step()
            self.pre_con_lambdas.data.copy_(self.pre_con_lambdas.data.clamp(-8.0, 8.0))

        # return
        train_results = {
            'objective': objective.item(),
            'constraints': constraints.detach().cpu().numpy(),
            'reward_critic_loss': reward_critic_loss.item(),
            'cost_critic_loss': cost_critic_loss.item(),
            'entropy': entropy.item(),
            'con_lambdas': con_lambdas_tensor.detach().cpu().numpy(),
            'entropy_alpha': entropy_alpha_tensor.item(),
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
            'target_reward_critic': self.target_reward_critic.state_dict(),
            'reward_critic_optimizer': self.reward_critic_optimizer.state_dict(),            
            'pre_entropy_alpha': self.pre_entropy_alpha.data,
            'entropy_alpha_optimizer': self.entropy_alpha_optimizer.state_dict(),
            'pre_con_lambdas': self.pre_con_lambdas.data,
            'con_lambdas_optimizer': self.con_lambdas_optimizer.state_dict(),
       }
        for cost_idx in range(self.cost_dim):
            save_dict[f'cost_critic_{cost_idx}'] = self.cost_critics[cost_idx].state_dict()
            save_dict[f'target_cost_critic_{cost_idx}'] = self.target_cost_critics[cost_idx].state_dict()
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
            self.target_reward_critic.load_state_dict(checkpoint['target_reward_critic'])
            self.reward_critic_optimizer.load_state_dict(checkpoint['reward_critic_optimizer'])
            self.pre_entropy_alpha.data = checkpoint['pre_entropy_alpha']
            self.entropy_alpha_optimizer.load_state_dict(checkpoint['entropy_alpha_optimizer'])
            self.pre_con_lambdas.data = checkpoint['pre_con_lambdas']
            self.con_lambdas_optimizer.load_state_dict(checkpoint['con_lambdas_optimizer'])
            for cost_idx in range(self.cost_dim):
                self.cost_critics[cost_idx].load_state_dict(
                    checkpoint[f'cost_critic_{cost_idx}'])
                self.target_cost_critics[cost_idx].load_state_dict(
                    checkpoint[f'target_cost_critic_{cost_idx}'])
                self.cost_critic_optimizers[cost_idx].load_state_dict(
                    checkpoint[f'cost_critic_optimizer_{cost_idx}'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
            return int(model_num)
        else:
            self.actor.initialize()
            self.reward_critic.initialize()
            self._softUpdate(self.target_reward_critic, self.reward_critic, 0.0)
            for cost_idx in range(self.cost_dim):
                self.cost_critics[cost_idx].initialize()
                self._softUpdate(self.target_cost_critics[cost_idx], self.cost_critics[cost_idx], 0.0)
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
            return 0

    #################
    # Private Methods
    #################

    def _softUpdate(self, target, source, polyak):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * polyak + param.data * (1.0 - polyak))
