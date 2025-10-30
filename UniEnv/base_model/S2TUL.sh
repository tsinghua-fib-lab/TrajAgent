#!/bin/bash

export PYTHONPATH=./UniEnv/model_lib/S2TUL
cd ./UniEnv/model_lib/S2TUL
BASE_ENV_PATH="your_base_env_path"
# 默认参数
AUGMENTATION="--use_aug"

${BASE_ENV_PATH}/torch-1.9.0-py38-pri/bin/python batched_main.py \
  --data_path $3 \
  --aug_name $2 \
  --dataset $7 \
  --city $8 \
  --result_path $6 \
  --gpu_id $1 \
  --max_step "${10}" \
  --epochs $5 \
  $AUGMENTATION