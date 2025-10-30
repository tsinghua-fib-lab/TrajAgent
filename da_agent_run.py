import argparse
import re
import os
from data_augmentation.da_agent import ReactReflectAgent
from data_augmentation.utils.llm_da_utils import train_model, get_model_result, copy_allfiles
from UniEnv.etc.settings import *

def da_main(trial_num, memory_length, city, task, XR, base_model, model, dataset, pa_da, seed=42, gpu_id=3,n_cpu=3, da_config=DA_CONFIG_FILE, time_sample="maximum", max_epoch=2, max_step=5, max_memory=30, enhance=0.5):
    cwd = os.getcwd()
    aug_data_path = os.path.join(cwd, PROCESS_DATA_OUTPUT_PATH, f"{dataset}/{MODEL_TYPE[model]}/{city}/{base_model}/{memory_length}")  #增强后的session训练数据存储地址
    da_pa_config_file_path = os.path.join(DA_PA_CONFIG_FILE_PATH, base_model, dataset, model, city, str(memory_length)) #增强算子参数配置文件存储位置
    
    if not os.path.exists(da_pa_config_file_path):
        os.makedirs(da_pa_config_file_path)
    if not os.path.exists(aug_data_path):
        os.makedirs(aug_data_path)
    
    copy_allfiles(DA_PA_CONFIG_FILE_PATH,da_pa_config_file_path)

    if pa_da:
        result_path = os.path.join(cwd, DA_PA_RESULT_PATH, task, base_model)
        
    else:
        result_path = os.path.join(cwd, RESULT_PATH, task)
        
    if XR:
        result_path = os.path.join(result_path,"xr")
    
      
    if memory_length>1:
       result_path = os.path.join(result_path,f"memory_{memory_length}") 
        
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    class Args:
        def __init__(self,
                    pa_da: bool,
                    city: str,
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
                    task: str,
                    da_pa_config_file_path: str,
                    base_model:str,
                    XR: bool,
                    memory_length:int,
                    trial_num: int
                    ):
            self.pa_da=pa_da
            self.XR=XR
            self.city=city
            self.model=model  
            self.dataset=dataset
            self.seed=seed
            self.base_model=base_model
            self.gpu_id=gpu_id 
            self.n_cpu=n_cpu
            self.result_path=result_path
            self.da_config=da_config
            self.time_sample=time_sample
            self.max_epoch=max_epoch 
            self.max_step=max_step
            self.max_memory=max_memory
            self.enhance=enhance
            self.memory_length=memory_length
            self.aug_data_path=aug_data_path
            self.task=task
            self.da_pa_config_file_path=da_pa_config_file_path
            self.trial_num=trial_num
    args = Args(
            pa_da=pa_da,
            city=city,
            model=model,  
            dataset=dataset,
            memory_length=memory_length,
            seed=seed, 
            gpu_id=gpu_id, 
            n_cpu=n_cpu,
            result_path=result_path,
            da_config=da_config,
            time_sample=time_sample,
            max_epoch=max_epoch, 
            max_step=max_step,
            max_memory=max_memory,
            enhance=enhance,
            aug_data_path=aug_data_path,
            task=task,
            da_pa_config_file_path=da_pa_config_file_path,
            base_model=base_model,
            XR=XR,
            trial_num=trial_num
            )
    agent_cls = ReactReflectAgent 

    train_model(args, [1000,1000])   #获得DL Model的原始数据训练结果准确率
    score = get_model_result(args, [1000,1000])
    # TODO: ./libcity/config/config_parser.py 115行数据集、默认配置要修改
    print(score)
    agents = agent_cls(score = score, index = [], args = args, n_cpu = args.n_cpu, enhance = args.enhance, max_steps = args.max_step) 
    result_indexes, result_score = agents.run()
    return result_indexes, result_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--city", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--gpu_id", default=0)
    parser.add_argument("--max_epoch", default=4)
    parser.add_argument("--pa_da", action='store_true')
    parser.add_argument("--base_model", default="llama3-70b")
    parser.add_argument("--max_step", default=10)
    parser.add_argument("--XR", action='store_true')
    parser.add_argument("--memory_length", type=int, default=1)
    parser.add_argument("--trial_num", type=int, default=1)

    args = parser.parse_args()
    result_indexes, result_score = da_main(trial_num=args.trial_num, memory_length=args.memory_length, XR=args.XR, model = args.model, base_model=args.base_model, dataset = args.dataset, city = args.city, task = args.task, gpu_id=args.gpu_id, max_epoch=args.max_epoch, pa_da=args.pa_da, max_step=args.max_step)
    print(result_indexes, result_score)
