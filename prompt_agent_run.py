import argparse
import os
import random
import json
import pandas as pd
from UniEnv.etc.settings import LIMP_DATA_PATH, MODEL_TYPE, PROCESS_DATA_OUTPUT_PATH
from prompt_optimize.prompt_agent import PromptReflectAgent

SFT_DATA_PATH = os.path.join(LIMP_DATA_PATH, "SFT/")  # 存储模型结果
gpt_ft_path = os.path.join(SFT_DATA_PATH, 'finetune_trajectory_annotated.csv')
random.seed(114514)

def prompt_main(shot_length, dataset, memory_length, city, user_num, trial_num, base_model, task, model, seed=42, n_cpu=3,  max_step=4, enhance=0.3):
    cwd = os.getcwd()
    aug_data_path = os.path.join(cwd, PROCESS_DATA_OUTPUT_PATH, f"{dataset}/{MODEL_TYPE[model]}/{city}/{base_model}/{memory_length}")  #增强后的session训练数据存储地址
    
    class Args:
        def __init__(self,
                    trial_num: int,
                    model: str,
                    seed: int,
                    n_cpu: int,
                    max_step: int, 
                    enhance: float,
                    base_model: str,
                    task: str,
                    user_num: int,
                    shot_length: int,
                    memory_length: int,
                    city: str,
                    dataset: str,
                    aug_data_path: str
                    ):
            self.model=model
            self.trial_num=trial_num
            self.base_model=base_model
            self.user_num=user_num
            self.seed=seed 
            self.n_cpu=n_cpu
            self.shot_length=shot_length
            self.max_step=max_step
            self.enhance=enhance
            self.task=task
            self.memory_length=memory_length
            self.city=city
            self.dataset=dataset
            self.aug_data_path=aug_data_path

    args = Args(
            model=model,  
            seed=seed, 
            n_cpu=n_cpu,
            max_step=max_step,
            enhance=enhance,
            shot_length=shot_length,
            task=task,
            base_model=base_model,
            trial_num=trial_num,
            user_num=user_num,
            memory_length=memory_length,
            city=city,
            dataset=dataset,
            aug_data_path=aug_data_path
            )
    agent_cls = PromptReflectAgent 
    
    if MODEL_TYPE[model] == "LLM":
        # 读取traj_preprocess.py生成的数据
        # 数据格式：[traj_id, type, hour_24h, weekday, entity_id, location, longitude, latitude, type_name]
        test_data_path = os.path.join(aug_data_path, "test_1000_1000.json")
        train_data_path = os.path.join(aug_data_path, "train_1000_1000.json")
        
        # 检查文件是否存在
        if not os.path.exists(test_data_path):
            print(f"Error: Test data file not found at {test_data_path}")
            print("Please run traj_preprocess.py first to generate the required data files.")
            return 0.0
            
        if not os.path.exists(train_data_path):
            print(f"Error: Train data file not found at {train_data_path}")
            print("Please run traj_preprocess.py first to generate the required data files.")
            return 0.0
        
        # 读取测试数据
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # 读取训练数据
        with open(train_data_path, 'r') as f:
            train_data = json.load(f)
        
        # 获取所有用户
        all_users = list(test_data.keys())
        all_users.sort()
        
        # 如果指定了用户数量，则随机选择用户
        if user_num > 0 and user_num < len(all_users):
            select_users = random.sample(all_users, user_num)
        else:
            select_users = all_users
        
        # 设置默认准确率
        accuracy = 0.375
        
        print(f"Loaded {len(test_data)} users from test data")
        print(f"Loaded {len(train_data)} users from train data")
        print(f"Selected {len(select_users)} users for processing")
        
        # 验证数据格式
        if len(select_users) > 0:
            sample_user = select_users[0]
            if sample_user in test_data and len(test_data[sample_user]) > 0:
                sample_trajectory = test_data[sample_user][0]
                if len(sample_trajectory) > 0:
                    sample_point = sample_trajectory[0]
                    print(f"Sample data point format: {len(sample_point)} fields")
                    print(f"Sample point: {sample_point}")
                    
                    # 验证数据格式是否符合预期
                    if len(sample_point) >= 9:
                        traj_id, poi_type, hour, weekday, user_id, location, longitude, latitude, poi_name = sample_point[:9]
                        print(f"Data format validation:")
                        print(f"  - traj_id: {traj_id} (type: {type(traj_id)})")
                        print(f"  - poi_type: {poi_type} (type: {type(poi_type)})")
                        print(f"  - hour: {hour} (type: {type(hour)})")
                        print(f"  - weekday: {weekday} (type: {type(weekday)})")
                        print(f"  - user_id: {user_id} (type: {type(user_id)})")
                        print(f"  - location: {location} (type: {type(location)})")
                        print(f"  - longitude: {longitude} (type: {type(longitude)})")
                        print(f"  - latitude: {latitude} (type: {type(latitude)})")
                        print(f"  - poi_name: {poi_name} (type: {type(poi_name)})")
                    else:
                        print(f"Warning: Data point has {len(sample_point)} fields, expected at least 9")
        
    else:
        # 对于非LLM模型，使用原有的CSV数据读取逻辑
        try:
            data = pd.read_csv(gpt_ft_path)
            all_users = list(data['user_id'].values)
            all_users.sort()
            select_users = random.choices(all_users, k=args.user_num)
            target_data = data[data['user_id'].isin(select_users)]
            acc = 0
            for idx, row in target_data.iterrows():
                if row['predicted_intent'] == row['intent']:
                    acc += 1
            accuracy = acc/target_data.shape[0]
        except FileNotFoundError:
            print(f"Warning: CSV file not found at {gpt_ft_path}")
            print("Using default accuracy for non-LLM models")
            accuracy = 0.375
            select_users = []
    
    agents = agent_cls(select_users = select_users, score = accuracy, args = args, n_cpu = args.n_cpu, enhance = args.enhance, max_steps = args.max_step) 
    result_score = agents.run()
    return result_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str, default="TUL")
    parser.add_argument("--max_step", type=int)
    parser.add_argument("--base_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--trial_num", type=int, default=2)
    parser.add_argument("--shot_length", type=int, default=10)
    parser.add_argument("--user_num", type=int, default=2)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--memory_length", type=int)
    parser.add_argument("--city", type=str)

    args = parser.parse_args()
    result_score = prompt_main(shot_length = args.shot_length, dataset = args.dataset, city = args.city, memory_length = args.memory_length, user_num = args.user_num, trial_num = args.trial_num, model = args.model, task = args.task, max_step=args.max_step, base_model=args.base_model)
    print(result_score)
