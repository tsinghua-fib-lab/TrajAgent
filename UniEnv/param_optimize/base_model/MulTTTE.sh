#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/MulT-TTE
cd UniEnv/model_lib/MulT-TTE

PARAM_OP="--param_op"
BASE_ENV_PATH="your_base_env_path"
# 默认参数

${BASE_ENV_PATH}/torch_1.13.1_py39_pri/bin/python main.py \
  --epochs $5 \
  --data_path $3 \
  --aug_name $2 \
  --dataset $7 \
  --city $8 \
  --result_path $6 \
  --device "cuda:${1}" \
  --config_path "${12}" \
  --identify "example_0" \
  $PARAM_OP --optim_path "${11}" \
  --max_step "${13}" \
  --model "MulT_TTE" 
 

