#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/GraphMM-Master
cd UniEnv/model_lib/GraphMM-Master
BASE_ENV_PATH="your_base_env_path"

PARAM_OP="--param_op"

# 默认参数
#  torch-1.10.2-cu111-py37
${BASE_ENV_PATH}/graphmm/bin/python train_gmm.py \
  --epochs $5 \
  --root_path $3 \
  --aug_name $2 \
  --dataset "tencent" \
  --city $8 \
  --result_path $6 \
  --dev_id $1 \
  --config_path "${12}" \
  $PARAM_OP --optim_path "${11}" \
  --max_step "${13}" 


