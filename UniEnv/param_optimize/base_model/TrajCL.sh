#!/bin/bash
export PYTHONPATH=./UniEnv/model_lib/TrajCL
cd ./UniEnv/model_lib/TrajCL

PARAM_OP="--param_op"
BASE_ENV_PATH="your_base_env_path"
${BASE_ENV_PATH}/torch-181-py37/bin/python train.py \
  --root_dir $3 \
  --dataset $7 \
  --trajcl_training_epochs 7 \
  --max_step "${13}" \
  --result_path $6 \
  --aug_name $2 \
  --city $8 \
  --gpu_id $1 \
  --optim_path "${11}"
  $PARAM_OP \
  --config_file "${12}" 

if [ $? -eq 0 ]; then
    ${BASE_ENV_PATH}/torch-181-py37/bin/python train_trajsimi.py \
    --root_dir $3 \
    --dataset $7 \
    --trajcl_training_epochs 7 \
    --max_step "${13}" \
    --result_path $6 \
    --aug_name $2 \
    --city $8 \
    --gpu_id $1 \
    --optim_path "${11}"
    $PARAM_OP \
    --config_file "${12}" \
    --trajsimi_measure_fn_name hausdorff

