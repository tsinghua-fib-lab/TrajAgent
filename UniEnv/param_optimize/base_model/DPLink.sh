#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/DPLink
cd UniEnv/model_lib/DPLink
BASE_ENV_PATH="your_base_env_path"
# 默认参数
AUGMENTATION="--use_aug"
REPEAT=1
PARAM_OP="--param_op"

${BASE_ENV_PATH}/dplink/bin/python codes/run.py \
  --data_path $3 \
  --repeat $REPEAT \
  --epoch $5 $AUGMENTATION \
  --dataset $7 \
  --city $8 \
  --task $9 \
  --config_path "${12}" \
  $PARAM_OP --optim_path "${11}" \
  --max_step "${13}" 
