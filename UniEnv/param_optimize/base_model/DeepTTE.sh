#!/bin/bash
export MKL_THREADING_LAYER=GNU

PARAM_OP="--param_op"
BASE_ENV_PATH="your_base_env_path"
${BASE_ENV_PATH}/traj_agent/bin/python UniEnv/model_lib/DeepTTE/run_model.py \
--data $3 \
--max_epoch $5 \
--result_path $6 \
--dataset $7 \
--city $8 \
--task $9 \
$PARAM_OP 1 \
--config_path "${12}" \
--optim_path "${11}" \
--max_step "${13}" 