# Spectral-Risk-Constrained Policy Optimization (SRCPO)

## Requirements

- Python 3.8 or greater
- torch==1.12.1
- gymnasium
- safety_gymnasium
- scikit-learn
- pandas
- qpsolvers=1.9.0
- ruamel.yaml

## How to train

`python main.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/{algo_name}/{env_name}.yaml --seed {seed_idx} --gpu_idx {gpu_idx} --wandb`

or  

`bash scripts/train.sh`

## How to eval

`python main.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/{algo_name}/{env_name}.yaml --seed {seed_idx} --gpu_idx {gpu_idx} --model_num {model_num} --eval`

or  

`bash scripts/eval.sh`

## How to test (render)

`python main.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/{algo_name}/{env_name}.yaml --seed {seed_idx} --gpu_idx {gpu_idx} --model_num {model_num} --test`
