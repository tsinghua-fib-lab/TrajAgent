#!/bin/bash

export PYTHONPATH="UniEnv/model_lib/MainTUL2"
cd UniEnv/model_lib/MainTUL2
BASE_ENV_PATH="your_base_env_path"

# 默认参数

${BASE_ENV_PATH}/dplink/bin/python project/main.py \
  --epochs $5 \
  --data_path $3 \
  --aug_name $2 \
  --dataset $7 \
  --city $8 \
  --result_path $6 \
  --gpu_id $1 \
  --max_step "${10}" 
