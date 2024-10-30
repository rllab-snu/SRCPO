# env
from utils.vectorize import ConAsyncVectorEnv
from tasks.safety_gym import SafetyGymEnv
import gymnasium as gym
import tasks.register

# utils
from utils import setSeed, cprint
from utils.logger import Logger

# algorithm
from algos import algo_dict

# base
from ruamel.yaml import YAML
from copy import deepcopy
import numpy as np
import argparse
import pickle
import torch
import wandb
import time
import os

def getParser():
    parser = argparse.ArgumentParser(description='RL')
    # common
    parser.add_argument('--wandb', action='store_true', help='use wandb?')
    parser.add_argument('--test', action='store_true', help='test or train?')
    parser.add_argument('--eval', action='store_true', help='evaluate?')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    parser.add_argument('--model_num', type=int, default=0, help='num model.')
    parser.add_argument('--save_freq', type=int, default=int(5e5), help='# of time steps for save.')
    parser.add_argument('--wandb_freq', type=int, default=int(1e3), help='# of time steps for wandb logging.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--task_cfg_path', type=str, help='cfg.yaml file location for task.')
    parser.add_argument('--algo_cfg_path', type=str, help='cfg.yaml file location for algorithm.')
    parser.add_argument('--project_name', type=str, default="RL", help='wandb project name.')
    parser.add_argument('--comment', type=str, default=None, help='wandb comment saved in run name.')
    return parser

def train(args, task_cfg, algo_cfg):
    # set seed
    setSeed(args.seed)

    # set arguments
    args.n_envs = task_cfg['n_envs']
    args.n_steps = algo_cfg['n_steps']
    args.n_total_steps = task_cfg['n_total_steps']
    args.max_episode_len = task_cfg['max_episode_len']

    # create environments
    if 'safe' in args.task_name.lower():
        if task_cfg['make_args']:
            env_id = lambda: SafetyGymEnv(args.task_name, **task_cfg['make_args'])
        else:
            env_id = lambda: SafetyGymEnv(args.task_name)
    else:
        env_id = lambda: gym.make(
            args.task_name, 
            max_episode_length=args.max_episode_len, 
            is_earlystop=task_cfg['is_earlystop']
        )
    vec_env = ConAsyncVectorEnv([env_id for _ in range(args.n_envs)])
    args.obs_dim = vec_env.single_observation_space.shape[0]
    args.action_dim = vec_env.single_action_space.shape[0]
    args.cost_dim = vec_env.single_cost_space.shape[0]
    args.action_bound_min = vec_env.single_action_space.low
    args.action_bound_max = vec_env.single_action_space.high
    args.cost_names = task_cfg["costs"]
    assert len(args.cost_names) == args.cost_dim

    # declare agent
    agent_args = deepcopy(args)
    for key in algo_cfg.keys():
        agent_args.__dict__[key] = algo_cfg[key]
    agent = algo_dict[args.algo_name.lower()](agent_args)
    initial_step = agent.load(args.model_num)

    # wandb
    if args.wandb:
        wandb.init(project=args.project_name, config=args)
        if args.comment is not None:
            wandb.run.name = f"{args.name}/{args.comment}"
        else:
            wandb.run.name = f"{args.name}"

    # logger
    log_name_list = deepcopy(agent_args.logging['cost_indep'])
    for log_name in agent_args.logging['cost_dep']:
        log_name_list += [f"{log_name}_{cost_name}" for cost_name in args.cost_names]
    logger = Logger(log_name_list, f"{args.save_dir}/logs")

    # set train parameters
    reward_sums = np.zeros(args.n_envs)
    cost_sums = np.zeros((args.n_envs, args.cost_dim))
    discounted_cost_sums = np.zeros((args.n_envs, args.cost_dim))
    discounts = np.ones(args.n_envs)
    env_cnts = np.zeros(args.n_envs)
    total_step = initial_step
    wandb_step = initial_step
    save_step = initial_step

    # initialize environments
    n_actions_per_env = (args.max_episode_len*np.arange(args.n_envs)/args.n_envs).astype(int)
    observations, infos = vec_env.reset(n_actions_per_env=n_actions_per_env)
    agent.reset(observations)

    # start training
    for _ in range(int(initial_step/args.n_steps), int(args.n_total_steps/args.n_steps)):
        start_time = time.time()

        for _ in range(int(args.n_steps/args.n_envs)):
            env_cnts += 1
            total_step += args.n_envs

            # ======= collect trajectories & training ======= #
            actions = agent.getAction(observations, None, None, False)
            observations, rewards, terminates, truncates, infos = vec_env.step(actions)
            costs = rewards[..., 1:]
            rewards = rewards[..., 0]

            discounted_cost_sums += costs*discounts[:, None]
            discounts *= agent.discount_factor

            reward_sums += rewards
            cost_sums += costs
            temp_fails = []
            temp_dones = []
            temp_observations = []
            temp_discounted_cost_sums = discounted_cost_sums.copy()
            temp_discounts = discounts.copy()

            for env_idx in range(args.n_envs):
                fail = (not truncates[env_idx]) and terminates[env_idx]
                done = terminates[env_idx] or truncates[env_idx]
                temp_observations.append(
                    infos['final_observation'][env_idx] 
                    if done else observations[env_idx])
                temp_fails.append(fail)
                temp_dones.append(done)

                if done:
                    agent.reset(observations, env_idx)
                    eplen = env_cnts[env_idx]
                    if 'eplen' in logger.log_name_list: 
                        logger.write('eplen', [eplen, eplen])
                    if 'reward_sum' in logger.log_name_list:
                        logger.write('reward_sum', [eplen, reward_sums[env_idx]])
                    for cost_idx in range(args.cost_dim):
                        log_name = f'cost_sum_{args.cost_names[cost_idx]}'
                        if log_name in logger.log_name_list: 
                            logger.write(log_name, [eplen, cost_sums[env_idx, cost_idx]])
                        log_name = f'discounted_cost_sum_{args.cost_names[cost_idx]}'
                        if log_name in logger.log_name_list: 
                            logger.write(log_name, [eplen, discounted_cost_sums[env_idx, cost_idx]])
                    discounted_cost_sums[env_idx] = 0.0
                    discounts[env_idx] = 1.0
                    reward_sums[env_idx] = 0
                    cost_sums[env_idx, :] = 0
                    env_cnts[env_idx] = 0

            temp_dones = np.array(temp_dones)
            temp_fails = np.array(temp_fails)
            temp_observations = np.array(temp_observations)
            agent.step(rewards, costs, temp_dones, temp_fails, temp_observations, temp_discounted_cost_sums, temp_discounts)
            # =============================================== #

            # wandb logging
            if total_step - wandb_step >= args.wandb_freq and args.wandb:
                wandb_step += args.wandb_freq
                log_data = {"step": total_step}
                print_len_episode = max(int(args.wandb_freq/args.max_episode_len), args.n_envs)
                print_len_step = max(int(args.wandb_freq/args.n_steps), args.n_envs)
                for cost_idx, cost_name in enumerate(args.cost_names):
                    for log_name in agent_args.logging['cost_dep']:
                        if 'sum' in log_name:                        
                            log_data[f'{log_name}/{cost_name}'] = logger.get_avg(f'{log_name}_{cost_name}', print_len_episode)
                        else:
                            log_data[f'{log_name}/{cost_name}'] = logger.get_avg(f'{log_name}_{cost_name}', print_len_step)
                for log_name in agent_args.logging['cost_indep']:
                    if ('sum' in log_name) or ('eplen' in log_name):
                        log_data[f"metric/{log_name}"] = logger.get_avg(log_name, print_len_episode)
                    else:
                        log_data[f"metric/{log_name}"] = logger.get_avg(log_name, print_len_step)
                wandb.log(log_data)
                print(log_data)

            # save
            if total_step - save_step >= args.save_freq:
                save_step += args.save_freq
                agent.save(total_step)
                logger.save()

        # train
        if agent.readyToTrain():
            train_results = agent.train()
            for key, value in train_results.items():
                if key in agent_args.logging['cost_indep']:
                    logger.write(key, [args.n_steps, value])
                if key in agent_args.logging['cost_dep']:
                    for cost_idx, cost_name in enumerate(args.cost_names):
                        logger.write(f"{key}_{cost_name}", [args.n_steps, value[cost_idx]])

        # calculate FPS
        end_time = time.time()
        fps = args.n_steps/(end_time - start_time)
        if 'fps' in logger.log_name_list:
            logger.write('fps', [args.n_steps, fps])

    # final save
    agent.save(total_step)
    logger.save()

    # terminate
    vec_env.close()

def test(args, task_cfg, algo_cfg):
    # set arguments
    args.n_envs = 1
    args.n_steps = algo_cfg['n_steps']
    args.n_total_steps = task_cfg['n_total_steps']
    args.max_episode_len = task_cfg['max_episode_len']

    # create environments
    if 'safe' in args.task_name.lower():
        if task_cfg['make_args']:
            env = SafetyGymEnv(args.task_name, render_mode="human", **task_cfg['make_args'])
        else:
            env = SafetyGymEnv(args.task_name, render_mode="human")
    else:
        env = gym.make(
            args.task_name, 
            max_episode_length=args.max_episode_len, 
            is_earlystop=task_cfg['is_earlystop']
        )
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.cost_dim = env.cost_space.shape[0]
    args.action_bound_min = env.action_space.low
    args.action_bound_max = env.action_space.high
    args.cost_names = task_cfg["costs"]
    assert len(args.cost_names) == args.cost_dim

    # declare agent
    agent_args = deepcopy(args)
    for key in algo_cfg.keys():
        agent_args.__dict__[key] = algo_cfg[key]
    agent = algo_dict[args.algo_name.lower()](agent_args)
    agent.load(args.model_num)

    # start rollouts
    for _ in range(100):
        # initialize environments
        observation, info = env.reset()
        agent.reset(observation.reshape(1, -1))
        reward_sum = 0.0
        cost_sum = np.zeros(args.cost_dim)

        for _ in range(args.max_episode_len):
            # ======= collect trajectories & training ======= #
            actions = agent.getAction(observation.reshape(1, -1), None, None, True)
            observation, reward, terminate, truncate, info = env.step(actions[0])
            cost = reward[..., 1:]
            reward = reward[..., 0]
            time.sleep(0.01)
            env.render()

            reward_sum += reward
            cost_sum += cost
            if terminate or truncate:
                break
            # =============================================== #
        print(reward_sum, cost_sum)


def eval(args, task_cfg, algo_cfg):
    # set arguments
    args.n_envs = 1
    args.n_steps = algo_cfg['n_steps']
    args.n_total_steps = task_cfg['n_total_steps']
    args.max_episode_len = task_cfg['max_episode_len']

    # create environments
    if 'safe' in args.task_name.lower():
        if task_cfg['make_args']:
            env = SafetyGymEnv(args.task_name, **task_cfg['make_args'])
        else:
            env = SafetyGymEnv(args.task_name)
    else:
        env = gym.make(
            args.task_name, 
            max_episode_length=args.max_episode_len, 
            is_earlystop=task_cfg['is_earlystop']
        )
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.cost_dim = env.cost_space.shape[0]
    args.action_bound_min = env.action_space.low
    args.action_bound_max = env.action_space.high
    args.cost_names = task_cfg["costs"]
    assert len(args.cost_names) == args.cost_dim

    # declare agent
    agent_args = deepcopy(args)
    for key in algo_cfg.keys():
        agent_args.__dict__[key] = algo_cfg[key]
    agent = algo_dict[args.algo_name.lower()](agent_args)
    initial_step = agent.load(args.model_num)

    reward_sum_list = []
    cost_sum_list = []
    for ep_idx in range(100):
        # initialize environments
        observation, info = env.reset()
        agent.reset(observation.reshape(1, -1))
        reward_sum = 0.0
        cost_sum = np.zeros(args.cost_dim)
        for _ in range(args.max_episode_len):
            # ======= collect trajectories & training ======= #
            actions = agent.getAction(observation.reshape(1, -1), None, None, True)
            observation, reward, terminate, truncate, info = env.step(actions[0])
            cost = reward[..., 1:]
            reward = reward[..., 0]
            reward_sum += reward
            cost_sum += cost
            if terminate or truncate:
                break
            # =============================================== #
        print(ep_idx, reward_sum, cost_sum)
        reward_sum_list.append(reward_sum)
        cost_sum_list.append(cost_sum)
    reward_sum_list = np.array(reward_sum_list)
    cost_sum_list = np.array(cost_sum_list)

    # save results
    eval_log_dir = f"{agent.save_dir}/eval"
    if not os.path.exists(eval_log_dir):
        os.makedirs(eval_log_dir)
    with open(f"{eval_log_dir}/result.pkl", 'wb') as f:
        pickle.dump([reward_sum_list, cost_sum_list], f)

    # terminate
    env.close()


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    # ==== processing args ==== #
    # load configuration file
    with open(args.task_cfg_path, 'r') as f:
        task_cfg = YAML().load(f)
    args.task_name = task_cfg['name']
    with open(args.algo_cfg_path, 'r') as f:
        algo_cfg = YAML().load(f)
    args.algo_name = algo_cfg['name']
    args.postfix = algo_cfg.get('postfix', None)
    if args.postfix is not None:
        args.name = f"{(args.task_name.lower())}_{(args.algo_name.lower())}_{(args.postfix.lower())}"
    else:
        args.name = f"{(args.task_name.lower())}_{(args.algo_name.lower())}"
    # save_dir
    args.save_dir = f"results/{args.name}/seed_{args.seed}"
    # device
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device(f'cuda:{args.gpu_idx}')
        cprint('[torch] cuda is used.', bold=True, color='cyan')
    else:
        device = torch.device('cpu')
        cprint('[torch] cpu is used.', bold=True, color='cyan')
    args.device = device
    # ========================= #

    if args.test:
        test(args, task_cfg, algo_cfg)
    elif args.eval:
        eval(args, task_cfg, algo_cfg)
    else:
        train(args, task_cfg, algo_cfg)
