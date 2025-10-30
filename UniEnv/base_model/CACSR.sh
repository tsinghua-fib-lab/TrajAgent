#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/CACSR
BASE_ENV_PATH="your_base_env_path"
cd UniEnv/model_lib/CACSR

# 默认参数
AUGMENTATION="--use_aug"

${BASE_ENV_PATH}/torch-1.10.2-py39/bin/python train_CACSR.py \
  --dataroot $3 \
  --aug_name $2 \
  --dataset $7 \
  --city $8 \
  --result_path $6 \
  --gpu_id $1 \
  --max_step "${10}" \
  --max_epoch $5 \
  $AUGMENTATION