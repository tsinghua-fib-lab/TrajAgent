#!/bin/bash

# 目标服务器的 SSH 登录信息
SSH_HOST="duyuwei@101.6.69.55"
SSH_KEY="~/.ssh/id_rsa.pub"
SSH_PORT="35167"

# 远程服务器上的 Python 脚本路径和工作目录
WORKING_DIR="MainTUL"
REMOTE_SCRIPT_PATH="project/main.py"
ENVIRONMENT="/usr/local/anaconda/envs/torch-1.9.1-py39/bin/python"

# 使用 SSH 执行远程 Python 脚本，并传递参数
# -i 参数允许您指定 SSH 私钥
# cd 命令切换到相应目录
# conda activate 激活指定的环境
# _switch_cuda 设置CUDA版本
# python3 运行Python脚本并传递参数
# ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_HOST" << EOF
ssh -p "$SSH_PORT" "$SSH_HOST" << EOF
    cd $WORKING_DIR
    export PATH=/usr/local/cuda-10.2/bin:\$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:\$LD_LIBRARY_PATH
    $ENVIRONMENT $REMOTE_SCRIPT_PATH 
EOF
