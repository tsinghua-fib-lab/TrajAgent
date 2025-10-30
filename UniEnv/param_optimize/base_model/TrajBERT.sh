#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/TrajBERT/
cd UniEnv/model_lib/TrajBERT/
PARAM_OP="--param_op"
BASE_ENV_PATH="your_base_env_path"
# 默认参数

${BASE_ENV_PATH}/traj_agent/bin/python run_model.py \
  --max_epoch $5 \
  --data_path $3 \
  --dataset $7 \
  --result_path $6 \
  --gpu_id $1 \
  --config_path "${12}" \
  $PARAM_OP --optim_path "${11}" \
  --aug_name $2 \
  --max_step "${13}" 

