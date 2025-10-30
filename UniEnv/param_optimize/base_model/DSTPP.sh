#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/DSTPP
cd UniEnv/model_lib/DSTPP
BASE_ENV_PATH="your_base_env_path"
PARAM_OP="--param_op"

# 默认参数

${BASE_ENV_PATH}/STAN_py37_cu101_torch171/bin/python app.py \
  --total_epochs $5 \
  --data_path $3 \
  --aug_name $2 \
  --dataset "Earthquake" \
  --mode "train" \
  --timesteps 50 \
  --samplingsteps 50 \
  --batch_size 64 \
  --city $8 \
  --result_path $6 \
  --cuda_id $1 \
  --config_path "${12}" \
  $PARAM_OP --optim_path "${11}" \
  --max_step "${13}" 
