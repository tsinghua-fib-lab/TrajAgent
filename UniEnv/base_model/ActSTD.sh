#!/bin/bash
export PYTHONPATH=./UniEnv/model_lib/Activity-Trajectory-Generation/src
BASE_ENV_PATH="your_base_env_path"
cd ./UniEnv/model_lib/Activity-Trajectory-Generation/src

# 默认参数
AUGMENTATION="--use_aug"

${BASE_ENV_PATH}/STAN_py37_cu101_torch171/bin/python app.py \
  --data_path $3 \
  --aug_name $2 \
  --dataset $7 \
  --city $8 \
  --result_path $6 \
  --gpu_id $1 \
  --max_step "${10}" \
  --epochs $5 \
  --num_iterations 3 \
  --model "jumpcnf" \
  --tpp "neural" \
  --solve_reverse \
  --ode_method "scipy_solver" \
  --ode_solver "RK45" \
  --tpp_style "gru" \
  $AUGMENTATION
