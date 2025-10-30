#!/bin/bash
export MKL_THREADING_LAYER=GNU
BASE_ENV_PATH="your_base_env_path"


${BASE_ENV_PATH}/tensorflow-gpu-1.15.0/bin/python UniEnv/model_lib/GMVSAE/run_model.py \
--data $3 \
--max_epoch 2 \
--result_path $6 \
--dataset $7 \
--city $8 \
--task $9 \
--mode train

${BASE_ENV_PATH}/tensorflow-gpu-1.15.0/bin/python UniEnv/model_lib/GMVSAE/run_model.py \
--data $3 \
--max_epoch 2 \
--result_path $6 \
--dataset $7 \
--city $8 \
--task $9 \
--mode eval