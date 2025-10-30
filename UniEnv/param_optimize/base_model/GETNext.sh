#!/bin/bash

export PYTHONPATH=UniEnv/model_lib/GETNext
cd UniEnv/model_lib/GETNext
BASE_ENV_PATH="your_base_env_path"

PARAM_OP="--param_op"

# 默认参数
${BASE_ENV_PATH}/STAN_py37_cu101_torch171/bin/python build_graph.py \
  --input_session_path $3 \
  --dataset_name $7 --city $8 \
  --config_path "${12}" \
  $PARAM_OP && \ 
  
${BASE_ENV_PATH}/STAN_py37_cu101_torch171/bin/python train.py \
  --epochs $5 \
  --city $8 \
  --max_step "${13}" \
  --input_session_path $3 \
  --dataset_name $7 \
  --result_path $6 \
  --gpu_id $1 \
  --config_path "${12}" \
  $PARAM_OP --optim_path "${11}" \
  --max_step "${13}" 