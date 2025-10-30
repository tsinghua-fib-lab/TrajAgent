import argparse
import json
import os
import jsonlines
from typing import List
from data_augmentation.utils.llm_da_utils import copy_allfiles
from param_optimize.utils.utils import train_model, get_model_result, get_config_param
from UniEnv.etc.settings import *
from param_optimize.pa_agent import ParamReflectAgent, ParamAgent
from UniEnv.model_lib.llm_methods import LLMZSMethod, LLMMoveMethod, LIMPMethod
from data_augmentation.utils.base_llm import LLMWrapper

def pa_main(sample_one_traj_per_user, prompt_num, traj_min_len, traj_max_len, memory_length, trial_num, XR, base_model, city, task, model, dataset, seed=42, gpu_id=0,n_cpu=3, da_config=DA_CONFIG_FILE, time_sample="maximum", max_epoch=4, max_step=4, max_memory=30, enhance=0.3, aug_methods_name="1000_1000", aug_data_path=None, user_num=150):
    # 重置LLMWrapper会话ID，确保每次运行都有独立的日志文件
    LLMWrapper.reset_session()
    
    cwd = os.getcwd()
    # /{args.base_model}/{args.memory_length}
    if DATA_TYPE[dataset] == 'checkin':
        aug_data_path = os.path.join(cwd, PROCESS_DATA_OUTPUT_PATH, f"{dataset}/{MODEL_TYPE[model]}/{city}/{base_model}/{memory_length}")  #增强后的session训练数据存储地址
    elif DATA_TYPE[dataset] in ['gps','map']:
        aug_data_path = os.path.join(cwd, PROCESS_DATA_INPUT_PATH, dataset, MODEL_TYPE[model])  #暂不支持地图数据增强处理
    else:
        aug_data_path = os.path.join(cwd, PROCESS_DATA_INPUT_PATH, dataset)
    config_path = os.path.join(cwd, PARAM_CONFIG_FILE_PATH, base_model, dataset, model,city,str(memory_length))
    result_path = os.path.join(cwd, PARAM_OP_RESULT_PATH, task, base_model)
    if XR:
        result_path = os.path.join(result_path, "xr")
    da_pa_config_file_path = os.path.join(DA_PA_CONFIG_FILE_PATH, base_model, dataset, model, city, str(memory_length)) #增强算子参数配置文件存储位置,以防ug_methods_name出现未增强过的方法组合
    
    if memory_length>1:
       result_path = os.path.join(result_path,f"memory_{memory_length}") 
    
    if not os.path.exists(da_pa_config_file_path):
        os.makedirs(da_pa_config_file_path)
    if not os.path.exists(result_path):
       os.makedirs(result_path)
    if not os.path.exists(config_path):
       os.makedirs(config_path)
    if not os.path.exists(aug_data_path):
       os.makedirs(aug_data_path)
    copy_allfiles(PARAM_CONFIG_FILE_PATH,config_path) #每次实验开始时，将配置文件复原
    class Args:
        def __init__(self,
                    sample_one_traj_per_user: bool,
                    prompt_num: int,
                    city: str,
                    trial_num: int,
                    model: str,
                    dataset: str,
                    seed: int,
                    gpu_id: int,
                    n_cpu: int,
                    result_path: str,
                    da_config: str,
                    time_sample: str, 
                    max_epoch: int, 
                    max_step: int, 
                    max_memory: int, 
                    enhance: float,
                    aug_data_path: str,
                    base_model: str,
                    task: str,
                    aug_methods_name: str,     
                    config_path: str,  
                    da_pa_config_file_path: str,  
                    XR: bool,
                    memory_length: int,
                    traj_min_len: int,
                    traj_max_len: int,
                    user_num: int
                    ):
            self.sample_one_traj_per_user=sample_one_traj_per_user
            self.prompt_num=prompt_num
            self.traj_min_len=traj_min_len
            self.traj_max_len=traj_max_len
            self.user_num=user_num
            self.city=city
            self.model=model
            self.dataset=dataset
            self.trial_num=trial_num
            self.base_model=base_model
            self.seed=seed 
            self.gpu_id=gpu_id 
            self.n_cpu=n_cpu
            self.result_path=result_path
            self.aug_data_path=aug_data_path
            self.da_config=da_config
            self.time_sample=time_sample
            self.max_epoch=max_epoch 
            self.max_step=max_step
            self.max_memory=max_memory
            self.enhance=enhance
            self.config_path=config_path
            self.task=task
            self.aug_methods_name = aug_methods_name
            self.XR = XR
            self.da_pa_config_file_path = da_pa_config_file_path
            self.memory_length = memory_length
    args = Args(
            sample_one_traj_per_user=sample_one_traj_per_user,
            city=city,
            traj_min_len=traj_min_len,
            traj_max_len=traj_max_len,
            model=model, 
            user_num=user_num,
            dataset=dataset,
            seed=seed, 
            prompt_num=prompt_num,
            gpu_id=gpu_id, 
            n_cpu=n_cpu,
            result_path=result_path,
            da_config=da_config,
            time_sample=time_sample,
            max_epoch=max_epoch, 
            max_step=max_step,
            XR=XR,
            max_memory=max_memory,
            enhance=enhance,
            task=task,
            base_model=base_model,
            aug_methods_name = aug_methods_name,
            aug_data_path=aug_data_path,
            config_path=config_path,
            da_pa_config_file_path=da_pa_config_file_path,
            trial_num=trial_num,
            memory_length=memory_length
            )
    
    if MODEL_TYPE[args.model] == "LLM":
        # 创建ParamAgent实例来调用get_pred_llm方法
        temp_agent = ParamAgent(
            score=0.25,  # 初始分数
            index=[],
            n_cpu=args.n_cpu,
            args=args,
            enhance=args.enhance,
            max_steps=args.max_step
        )
        
        # 从配置文件读取参数
        with open(os.path.join(PARAM_CONFIG_FILE_PATH, f"{model}.json"), "r") as f:
            result_dict = json.load(f)
        # 调用ParamAgent的get_pred_llm方法
        score = temp_agent.get_pred_llm(result_dict)
        index = []
    else:
        train_model(args, "1000_1000")   #获得DL Model的原始数据训练结果准确率
        score, index = get_model_result(args, "1000_1000")
    # TODO: ./libcity/config/config_parser.py 115行数据集、默认配置要修改
    print("init score:",score)
    agents = ParamReflectAgent(score = score, args = args, n_cpu = args.n_cpu, enhance = args.enhance, max_steps = args.max_step, index = index) 
    result_indexes, result_score = agents.run()
    
    # 打印日志文件路径信息和token统计
    session_id = LLMWrapper.get_session_id()
    if session_id:
        print(f"\n=== LLM交互日志文件 ===")
        print(f"会话ID: {session_id}")
        print(f"详细日志: {DIAL_RESULT_PATH}/llama3-70b/{session_id}_detailed_log.jsonl")
        print(f"预测日志: {args.result_path}/prediction_logs/predictions_{session_id}.jsonl")
        print(f"聊天历史: {DIAL_RESULT_PATH}/llama3-70b/{session_id}_history.jsonl")
        
        # 显示token统计信息
        print(f"\n=== Token使用统计 ===")
        # 从日志文件中读取token统计信息
        detailed_log_file = f"{DIAL_RESULT_PATH}/{base_model}/{session_id}_detailed_log.jsonl"
        if os.path.exists(detailed_log_file):
            total_input_tokens = 0
            total_output_tokens = 0
            total_tokens = 0
            interaction_count = 0
            
            try:
                with jsonlines.open(detailed_log_file, 'r') as reader:
                    for line in reader:
                        if line.get("type") == "interaction" and "prompt_tokens" in line:
                            interaction_count += 1
                            total_input_tokens += int(line.get("prompt_tokens", 0))
                            total_output_tokens += int(line.get("response_tokens", 0))
                            total_tokens += int(line.get("total_tokens", 0))
                
                print(f"总交互次数: {interaction_count}")
                print(f"总输入Token: {total_input_tokens:,}")
                print(f"总输出Token: {total_output_tokens:,}")
                print(f"总Token消耗: {total_tokens:,}")
                if interaction_count > 0:
                    print(f"平均每次交互输入Token: {total_input_tokens/interaction_count:.1f}")
                    print(f"平均每次交互输出Token: {total_output_tokens/interaction_count:.1f}")
                    print(f"平均每次交互总Token: {total_tokens/interaction_count:.1f}")
                else:
                    print("平均每次交互Token: 0.0")
            except Exception as e:
                print(f"读取token统计失败: {e}")
        else:
            print("详细日志文件不存在，无法获取token统计")
    
    return result_indexes, result_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--city", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--aug_methods_name", type=str, default="1000_1000")
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--max_step", type=int)
    parser.add_argument("--max_epoch", type=int)
    parser.add_argument("--base_model", type=str, default="llama3-70b")
    parser.add_argument("--XR", action='store_true')
    parser.add_argument("--trial_num", type=int, default=1)
    parser.add_argument("--memory_length", type=int, default=1)
    parser.add_argument("--user_num", type=int, default=200)
    parser.add_argument("--prompt_num", type=int, default=200)
    parser.add_argument("--traj_min_len", type=int, default=3)
    parser.add_argument("--traj_max_len", type=int, default=100)
    parser.add_argument("--sample_one_traj_per_user", action='store_true')

    args = parser.parse_args()

    result_param, result_score = pa_main(sample_one_traj_per_user=args.sample_one_traj_per_user, traj_min_len=args.traj_min_len, traj_max_len=args.traj_max_len, prompt_num=args.prompt_num, memory_length=args.memory_length, trial_num = args.trial_num, XR=args.XR, gpu_id = args.gpu_id, model = args.model, dataset = args.dataset, city = args.city, task = args.task, max_step=args.max_step, base_model=args.base_model, aug_methods_name=args.aug_methods_name, max_epoch=args.max_epoch, user_num=args.user_num)
    print(result_param, result_score)
