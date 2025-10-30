#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/GETNext
cd UniEnv/model_lib/GETNext
BASE_ENV_PATH="your_base_env_path"
# 默认参数
BATCH_SIZE=32
LEARNING_RATE=0.001
AUGMENTATION="--use_aug"
${BASE_ENV_PATH}/STAN_py37_cu101_torch171/bin/python build_graph.py \
  --input_session_path $3 \
  --dataset_name $7 --city $8 \
  --aug_name $2 \
  $AUGMENTATION && \ 


${BASE_ENV_PATH}/STAN_py37_cu101_torch171/bin/python train.py \
  --epochs $5 \
  --gpu_id $1 $AUGMENTATION --aug_name $2 --result_path $6 \
  --input_session_path $3 \
  --dataset_name $7 --city $8 \
  --max_step "${10}" 