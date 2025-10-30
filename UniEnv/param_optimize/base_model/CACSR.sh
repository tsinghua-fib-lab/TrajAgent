#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/CACSR
cd UniEnv/model_lib/CACSR
PARAM_OP="--param_op"
BASE_ENV_PATH="your_base_env_path"

# 默认参数

${BASE_ENV_PATH}/torch-1.10.2-py39/bin/python train_CACSR.py \
  --max_epoch $5 \
  --dataroot $3 \
  --aug_name $2 \
  --dataset $7 \
  --city $8 \
  --result_path $6 \
  --gpu_id $1 \
  --config_path "${12}" \
  $PARAM_OP --optim_path "${11}" \
  --max_step "${13}" 

