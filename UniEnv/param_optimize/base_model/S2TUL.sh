#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/S2TUL
cd UniEnv/model_lib/S2TUL

PARAM_OP="--param_op"
BASE_ENV_PATH="your_base_env_path"
# 默认参数

${BASE_ENV_PATH}/torch-1.9.0-py38-pri/bin/python batched_main.py \
  --epochs $5 \
  --data_path $3 \
  --aug_name $2 \
  --dataset $7 \
  --city $8 \
  --result_path $6 \
  --gpu_id $1 \
  --config_path "${12}" \
  $PARAM_OP --optim_path "${11}" \
  --max_step "${13}" 


