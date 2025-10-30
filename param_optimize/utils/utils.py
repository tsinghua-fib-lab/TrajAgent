import copy
import os
import yaml
import time
import json
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
import subprocess
# from data_augmentation.utils.data_transform import *
from yaml.loader import SafeLoader
from collections import Counter
from data_augmentation.utils import (
    operator_dict,
)
from datetime import datetime, timedelta
from UniEnv.etc.settings import *
from data_augmentation.utils.llm_da_utils import ParallelDA

def get_config_param(args):  #读取超参数配置文件
    if args.model in ['GRU','LSTM']:
        config_file = os.path.join(args.config_path,"RNN.json")
    else:
        config_file = os.path.join(args.config_path,f"{args.model}.json")

    with open(config_file,"r") as f:
        model_config = json.load(f)
    return model_config
def save_config_param(args, model_config): #将生成的配置文件写入
    with open(os.path.join(args.config_path, f"{args.model}.json"), "w") as f:
        json.dump(model_config, f)
    

def train_model(args, indexes_str):
    aug_data_path = args.aug_data_path
    file_path = args.result_path
    
    if args.model in ['GRU','LSTM']:
        config_file = os.path.join(args.config_path,"RNN.json")
    else:
        config_file = os.path.join(args.config_path,f"{args.model}.json")
    if DATA_TYPE[args.dataset] == 'checkin':
        indexes = [int(item) for item in indexes_str.split("_")]
        file_name = f"{indexes_str}.json"
        if file_name not in os.listdir(aug_data_path):
            da_module = ParallelDA(aug_data_path, args.da_config)
            da_module.generate_all(
                args=args,
                indexes=indexes,
                traj=True
            )   
    base_model_file = os.path.join(BASE_MODEL_PARAM_PATH, f"{args.model}.sh")
    subprocess.call(['sh', '-x', base_model_file, str(args.gpu_id), str(indexes_str), str(aug_data_path),
                     args.model, str(args.max_epoch), str(args.result_path), str(args.dataset), str(args.city),
                     str(args.task), "--param_op", str(args.result_path), str(config_file), str(args.max_step)])
    
def get_model_result(args, indexes):
    # if isinstance(indexes,list):
    #     indexes = [str(index) for index in indexes]
    #     indexes = "_".join(indexes)
    with open(os.path.join(args.config_path, "uuid.json"),"r") as f:
        uuid = json.load(f)
    for filename in os.listdir(args.result_path):
        if filename.split("_")[-1] == f"{uuid}.json":
            filename_LLM = filename
    with open(os.path.join(args.result_path, filename_LLM),"r") as f:
        result = json.load(f)
    return result[EVALUATE_METRIC[MODEL_TYPE[args.model]]], result['config']
        
    