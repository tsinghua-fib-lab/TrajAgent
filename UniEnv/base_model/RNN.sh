#!/bin/bash

# 设置工作目录
export PYTHONPATH="UniEnv/model_lib:${PYTHONPATH}"
cd "UniEnv/model_lib/libcity"
BASE_ENV_PATH="your_base_env_path"
# 默认参数
BATCH_SIZE=32
LEARNING_RATE=0.001
AUGMENTATION="--use_aug"

${BASE_ENV_PATH}/libcity_py39_torch231_cu121/bin/python run_model.py \
  --task traj_loc_pred --model $4 \
  --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --max_epoch $5 \
  --gpu_id $1 $AUGMENTATION --aug_name $2 --data_path $6 \
  --input_session_path $3 \
  --dataset $7 --city $8 \
  --max_step "${10}" 