#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/DeepMapMatching
cd UniEnv/model_lib/DeepMapMatching

PARAM_OP="--param_op"
BASE_ENV_PATH="your_base_env_path"
# 默认参数

${BASE_ENV_PATH}/traj_agent/bin/python DeepMM/seq2seq.py \
  --epoch $5 \
  --data_path $3 \
  --aug_name $2 \
  --dataset "tencent" \
  --city $8 \
  --result_path $6 \
  --config "./DeepMM/configs/config_best.json" \
  --gpu $1 \
  --config_path "${12}" \
  $PARAM_OP --optim_path "${11}" \
  --max_step "${13}" 


