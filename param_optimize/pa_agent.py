import os
import random
import time
import json
import numpy as np
import multiprocessing as mp
from typing import List, Any
import tqdm
import datetime
import jsonlines
import uuid
from param_optimize.utils.utils import train_model, get_model_result, get_config_param, save_config_param
from param_optimize.utils.prompt import THOUGHT, QUESTION, ACTION, XR_THOUGHT, XR_QUESTION, XR_ACTION
from param_optimize.utils.params import PARAMS_DESCRIPTION
from data_augmentation.utils.base_llm import LLMWrapper
from UniEnv.etc.settings import *
from prompt_optimize.prompt_agent import PromptAgent
from prompt_optimize.utils.utils import *
from UniEnv.model_lib.llm_methods import LLMZSMethod, LLMMoveMethod, LIMPMethod
from prompt_optimize.utils.utils import extract_json, evaluate

class ParamAgent:
    def __init__(self,
                 score: float,
                 index: list,
                 n_cpu: int,
                 args: Any,
                 enhance: float,
                 max_steps: int,
                 ) -> None:
        self.args = args
        self.n_cpu = n_cpu        
        self.max_steps = max_steps
        self.enhance = enhance    #停止条件，(step_score - self.threshold)/self.threshold >= self.enhance
        self.max_memory = args.max_memory  #记忆中存储记录超过10条，或者每轮运行超过10分钟仍未达到停止条件（即达到预期的提高效果）时，开始反思
        self.reset_period = 1800   #反思间隔是30min 
        self.threshold = score  #初始值为未经增强的DL模型训练结果
        self.threshold_index = index
        self.aug_methods_name = args.aug_methods_name  #传入的最佳增强策略
        self._last_reflect_time = time.time()
        # Initialize LLM method based on model type
        self._initialize_llm_method()
        
        self.__reset_agent()

    def _initialize_llm_method(self):
        """Initialize the appropriate LLM method based on model type"""
        config_path = os.path.join(os.getcwd(), "config", "llm_methods")
        os.makedirs(config_path, exist_ok=True)
        if MODEL_TYPE[self.args.model] == 'LLM':
            # 数据格式：[traj_id, type, hour_24h, weekday, entity_id, location, longitude, latitude, type_name]
            test_data_path = os.path.join(self.args.aug_data_path, "test_1000_1000.json")
            train_data_path = os.path.join(self.args.aug_data_path, "train_1000_1000.json")
            
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
            if self.args.user_num > 0 and self.args.user_num < len(all_users):
                self.select_users = random.sample(all_users, self.args.user_num)
            else:
                self.select_users = all_users
                    
            if self.args.model == "LLMZS":
                self.llm_method = LLMZSMethod(config_path)
            elif self.args.model == "LLMMove":
                self.llm_method = LLMMoveMethod(config_path)
            elif self.args.model == "LIMP":
                self.llm_method = LIMPMethod(config_path)
            else:
                # For other models, return None to use original logic
                self.llm_method = None

    def run(self, reset = True) :
        if reset:
            self.__reset_agent()   #重置agent,从step=1开始,重置memory
        self.inter_turn = 0
        self.memory_part = []   #效果好的记忆
        self.extract_llm = LLMWrapper(
                                    temperature=0,
                                    max_tokens=1000,
                                    model_name='llama3-70b',
                                    model_kwargs={"stop": "\n"},
                                    platform="DeepInfra")

        self.llm = LLMWrapper(
                                temperature=0,
                                max_tokens=1000,
                                model_name=self.args.base_model,
                                model_kwargs={"stop": "\n"},
                                platform="DeepInfra")

        while not self.is_halted() and not self.is_finished() and int(self.step_n) <= int(self.max_steps):
            step_index, step_score = self.step()
            self.scratchpad = '<SCRATCHPAD>:\n'  #清空短期记忆
            self.step_n += 1
            if EVALUATE_METRIC[MODEL_TYPE[self.args.model]] not in ['MAE']:
                if (step_score - self.threshold)/self.threshold >= self.enhance:  #若某步优化效果超过self.enhance,则可以停止优化，提前结束
                    self.finished = True
                if step_score > self.threshold: # 更新threshold
                    self.threshold = step_score
                    self.threshold_index = step_index
            else:
                if (self.threshold - step_score)/self.threshold >= self.enhance:  #若某步优化效果超过self.enhance,则可以停止优化，提前结束
                    self.finished = True
                if step_score < self.threshold:
                    self.threshold = step_score
                    self.threshold_index = step_index
        return step_index, step_score
    
    def parse_choice(self, choice):
        # 适用于各个模型的答案提取方法
        if "gpt" in self.args.base_model:
            config_str = choice.replace("'", '"')
            try:
                result_dict = json.loads(config_str)
            except json.JSONDecodeError as e:
                result_dict={}
            parse_dict_prompt = f"""
            Please check if the dictionary in SOURCE DICT is in the correct format.The dict should have the same keys as CONFIG PARAMS DICT.
            1. You can adapt the value by following methods:
               - Convert strings to integers if CONFIG expects integers
               - Convert strings to floats if CONFIG expects floats  
               - Keep strings as strings if CONFIG expects strings
               If the format is correct,please output the dictionary.
            2. If not, output an empty dict.Do not output other contents.\n
            <CONFIG PARAMS DICT>
            {self.model_config}
            <SOURCE DICT>
            {result_dict}
            """
        else:
            parse_dict_prompt = f"""
            Please check if the dictionary in SOURCE DICT is in the correct format.The dict should have the same keys as CONFIG PARAMS DICT.
            1. You can adapt the value by following methods:
               - Convert strings to integers if CONFIG expects integers
               - Convert strings to floats if CONFIG expects floats  
               - Keep strings as strings if CONFIG expects strings
               If the format is correct,please output the dictionary.
            2. If not, output an empty dict.
            Please only output the dict.Do not output other contents.\n
            <CONFIG PARAMS DICT>
            {self.model_config}
            <SOURCE DICT>
            {choice}
            """
        parse_dict = self.extract_llm.get_response(prompt_text=parse_dict_prompt)
        # 将字符串转换为JSON格式
        config_str = parse_dict.replace("'", '"')
        try:
            result_dict = json.loads(config_str)
        except json.JSONDecodeError as e:
            result_dict={}
        return result_dict
    
    def parse_thought(self, thought):
        
        # parse_thought_prompt = """
        # Please remove the part after "Action:" of the above sentence and output:\n
        # """
        # parse_thought = self.llm(thought + parse_thought_prompt)
        action_index = thought.find("Action:")# Action:
        if action_index!= -1:
            parse_thought = thought[:action_index]
        else:
            parse_thought = thought
        return parse_thought
    
    def get_trajectories(self, user_segments, train_user_segments, params):
        """Get trajectories for prediction with consistent sampling"""
        self.test_dictionary = {}
        self.true_locations = {}
        # for each trajectory id in test trajectory ids, we retrieve the historical stays
        # (location ids and time from training) and the context stays (location ids and time
        # from testing with specific trajectory id)
        counter = 0
        context_len = getattr(params, 'context_len', 6)
        history_len = getattr(params, 'history_len', 40)  # Default history length
        traj_min_num = 3  # Default history length
        traj_max_num = 10000  # Default history length
        sample_one_traj_per_user = self.args.sample_one_traj_per_user  # 控制每个用户是否只采样一个轨迹
        sample_traj_count = 1  # 每个用户采样的轨迹数量
        # 保证user是有序的，便于重复试验
        user_list = [str(y) for y in sorted([int(x) for x in list(user_segments.keys())])]
        
        for user in user_list:
            trajectories = user_segments[user]
            traj_count = 0
            
            if len(trajectories) < traj_min_num:
                continue 
            elif len(trajectories) > traj_max_num:
                continue 
                
            if user not in self.test_dictionary:
                self.test_dictionary[str(user)] = {}
            if user not in self.true_locations:
                self.true_locations[str(user)] = {}
            
            # 保证traj_id是有序的，便于重复试验
            traj_ids = [str(y) for y in sorted([int(x) for x in list(trajectories.keys())])]
            
            # 如果启用每个用户采样指定数量的轨迹，则随机选择轨迹
            if sample_one_traj_per_user and len(traj_ids) > sample_traj_count:
                import random
                # 使用固定的随机种子确保结果可重现
                random.seed(50)
                # 随机选择指定数量的轨迹
                selected_traj_ids = random.sample(traj_ids, min(sample_traj_count, len(traj_ids)))
                traj_ids = selected_traj_ids
            elif sample_one_traj_per_user and len(traj_ids) <= sample_traj_count:
                # 如果轨迹数量不足，使用所有可用轨迹
                print(f"用户 {user} 轨迹数量不足，使用所有轨迹: {traj_ids}")
            
            # Process each trajectory (list of location points)
            for traj_id in traj_ids:
                trajectory = trajectories[traj_id]
   
                if len(trajectory) < 2:  # Need at least 2 points for context and target
                    continue
                traj_count += 1    
                counter += 1
                
                if traj_id not in self.test_dictionary[str(user)]:
                    self.test_dictionary[str(user)][traj_id] = {}
                if traj_id not in self.true_locations[str(user)]:
                    self.true_locations[str(user)][traj_id] = {}
                
                # Convert trajectory points to the expected format
                # Each point: [traj_id, type, hour_24h, weekday, entity_id, location, longitude, latitude, type_name]
                # Convert to: [hour, weekday, type_name, location] for historical/context stays
                # And: [longitude, latitude] for positions
                
                # Get context length from parameters, default to trajectory length - 1
                
                # Split trajectory into context and target based on context_len parameter
                if context_len >= len(trajectory):
                    # If context_len is too large, use all but last point as context
                    context_points = trajectory[:-1]
                    target_point = trajectory[-1]
                else:
                    # Use specified context length
                    context_points = trajectory[-context_len-1:-1]  # Last context_len points before target
                    target_point = trajectory[-1]
                
                # Convert context points to the expected format
                context_stays = []
                context_pos = []
                for point in context_points:
                    # point: [traj_id, type, hour_24h, weekday, entity_id, location, longitude, latitude, type_name]
                    context_stays.append([point[2], point[3], point[8], point[5]])  # [hour, weekday, type_name, location]
                    context_pos.append([point[6], point[7]])  # [longitude, latitude]
                
                # Convert target point
                target_stay = [target_point[2], target_point[3], target_point[8], target_point[5]]  # [hour, weekday, type_name, location]
                target_pos = [target_point[6], target_point[7]]  # [longitude, latitude]
                
                # Get historical stays from training data (if available)
                historical_stays = []
                if user in train_user_segments and traj_id in train_user_segments[user]:
                    train_trajectory = train_user_segments[user][traj_id]
                    for point in train_trajectory:
                        historical_stays.append([point[2], point[3], point[8], point[5]])  # [hour, weekday, type_name, location]
                
                # Also add remaining trajectory points to historical_stays if available
                remaining_points = trajectory[:-context_len-1] if context_len < len(trajectory) - 1 else []
                for point in remaining_points:
                    historical_stays.append([point[2], point[3], point[8], point[5]])  # [hour, weekday, type_name, location]
                
                # Limit historical_stays length based on history_len parameter
                if len(historical_stays) > history_len:
                    historical_stays = historical_stays[-history_len:]
                
                # Store in test dictionary
                self.test_dictionary[str(user)][traj_id] = {
                    'historical_stays': historical_stays,
                    'historical_pos': [],  # Will be filled if needed
                    'historical_stays_long': historical_stays,
                    'context_stays': context_stays,
                    'context_pos': context_pos,
                    'target_stay': target_stay,
                }
                
                # Store true location (target point)
                self.true_locations[str(user)][traj_id] = {
                    'ground_stay': str(target_point[5]),  # location
                    'ground_addr': target_point[8],  # type_name
                    'ground_pos': target_pos  # [longitude, latitude]
                }
                
            # 总数控制
            if counter >= self.args.prompt_num:
                print("已经采集到足够的数据，要求轨迹:{} 实际轨迹:{}".format(self.args.prompt_num, counter))
                break
                
        if counter < self.args.prompt_num:
            print("数据不足，要求轨迹:{} 实际轨迹:{}".format(self.args.prompt_num, counter))

    def trajs_sampling(self, test_dataset):
        """Sample trajectories for testing"""
        traj_min_len = self.args.traj_min_len  # Default history length
        traj_max_len = self.args.traj_max_len  # Default history length
        user_segments = {}
        for user_id, user_trajectories in test_dataset.items():
            if str(user_id) not in self.select_users:
                continue
            if user_id not in user_segments:
                user_segments[user_id] = {}
            
            # Each user has a list of trajectories
            for traj_idx, trajectory in enumerate(user_trajectories):
                if len(trajectory) < traj_min_len or len(trajectory) > traj_max_len:  # Need at least 2 points
                    continue
                    
                # Extract trajectory ID from the first point
                # Data format: [traj_id, type, hour_24h, weekday, entity_id, location, longitude, latitude, type_name]
                traj_id = str(trajectory[0][0]) if trajectory else str(traj_idx)
                
                # Store the trajectory as-is for processing in get_trajectories
                user_segments[user_id][traj_id] = trajectory
                
        return user_segments
    
    def get_prediction(self, user_id, traj_id, prompt_text):
        """Get prediction using LLM method if available, otherwise use original logic"""
        output = dict()
        true_value = self.true_locations[user_id][traj_id]
        tar_poi = int(true_value['ground_stay'])
        
        # 记录预测开始信息
        prediction_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_id": user_id,
            "traj_id": traj_id,
            "model": self.args.model,
            "prompt_text": prompt_text,
            "true_value": true_value
        }
        
        try:
            pre_text = self.llm_method.llm.get_response(prompt_text=prompt_text)
            
            # 记录LLM响应
            prediction_log.update({
                "llm_response": pre_text,
                "status": "success"
            })
            
            if self.llm_method is not None:
                # Use LLM method for prediction parsing
                prediction, reason = self.llm_method.parse_prediction(pre_text)
                output['prediction'] = prediction
                output['true'] = tar_poi
                output['input'] = prompt_text
                output['reason'] = reason
                predictions = output
            else:
                # Use original logic for other models
                if self.args.model == "LLMMove":
                    try:            
                        res_content = eval(pre_text)
                        prediction = res_content["recommendation"]
                        output['prediction'] = prediction
                        output['true'] = tar_poi
                        output['input'] = prompt_text
                        predictions = output
                    except:
                        output_json, prediction, reason = extract_json(pre_text)
                        predictions = {
                        'input': prompt_text,
                        'output': output_json,
                        'prediction': prediction,
                        'reason': reason,
                        'true': tar_poi 
                        }
                else:
                    output_json, prediction, reason = extract_json(pre_text)
                    true_venue = true_value["ground_stay"]
                    predictions = {
                        'input': prompt_text,
                        'output': output_json,
                        'prediction': prediction,
                        'reason': reason,
                        'true': true_venue  
                    }
            
            # 记录最终预测结果
            prediction_log.update({
                "final_prediction": predictions,
                "parsing_status": "success"
            })
            
        except Exception as e:
            # 记录错误信息
            prediction_log.update({
                "status": "error",
                "error_message": str(e),
                "llm_response": "",
                "final_prediction": {}
            })
            print(f"Error in prediction for user {user_id}, traj {traj_id}: {e}")
            raise e
        
        # 保存预测日志到文件
        try:
            log_dir = os.path.join(self.args.result_path, "prediction_logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # 使用统一的会话ID，与LLMWrapper保持一致
            if not hasattr(self, '_session_id'):
                self._session_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            
            log_file = os.path.join(log_dir, f"predictions_{self._session_id}.jsonl")
            
            with jsonlines.open(log_file, 'a') as writer:
                writer.write(prediction_log)
        except Exception as e:
            print(f"Warning: Failed to save prediction log: {e}")
        
        return predictions
    
    def get_pred_llm(self, params):
        # Load test data
        test_dataset = json.load(open(os.path.join(self.args.aug_data_path, "test_1000_1000.json")))
        user_segments = self.trajs_sampling(test_dataset)
        
        # Load training data for historical context
        train_dataset = json.load(open(os.path.join(self.args.aug_data_path, "train_1000_1000.json")))
        train_user_segments = self.trajs_sampling(train_dataset)
        
        # Get trajectories
        self.get_trajectories(user_segments, train_user_segments, params)
        
        # Load stay points for LLMMove
        if self.args.model == "LLMMove":
            stay_points = json.load(open(os.path.join(self.args.aug_data_path, "stay_points.json")))
        else:
            stay_points = None
        
        # Initialize LLMWrapper with optimized parameters
        if self.llm_method is not None:
            # Extract LLMWrapper parameters from params
            llm_params = {
                'temperature': getattr(params, 'temperature', 0.01),
                'max_tokens': getattr(params, 'max_new_tokens', 200),
                'top_p': getattr(params, 'top_p', 0.95),
                'top_k': getattr(params, 'top_k', 250),
                'length_penalty': getattr(params, 'length_penalty', 1),
                'presence_penalty': getattr(params, 'presence_penalty', 0)
            }
            
            # Update LLMWrapper with new parameters
            self.llm_method.llm = LLMWrapper(
                temperature=llm_params['temperature'],
                max_tokens=llm_params['max_tokens'],
                model_name=self.args.base_model,
                model_kwargs={
                    "stop": "\n",
                    "top_p": llm_params['top_p'],
                    "top_k": llm_params['top_k'],
                    "length_penalty": llm_params['length_penalty'],
                    "presence_penalty": llm_params['presence_penalty']
                },
                platform="DeepInfra"
            )
        
        """First step prediction using LLM methods"""
        candidate_poi_info = dict()
        self.all_predictions = dict()
        
        # 计算总预测数量用于进度条
        total_predictions = sum(len(user_data) for user_data in self.test_dictionary.values())
        
        # 外层进度条：用户进度
        user_pbar = tqdm.tqdm(
            self.test_dictionary.items(), 
            desc="Processing Users", 
            total=len(self.test_dictionary),
            position=0,
            leave=True
        )
        
        processed_predictions = 0
        
        for user_id, user_data in user_pbar:
            user_id = str(user_id)
            if user_id not in self.all_predictions:
                 self.all_predictions[user_id] = {}
            
            # 内层进度条：轨迹预测进度
            traj_pbar = tqdm.tqdm(
                user_data.items(),
                desc=f"User {user_id} Predictions",
                total=len(user_data),
                position=1,
                leave=False
            )
            
            for traj_id, traj_seqs in traj_pbar:
                 # [hour, weekday, type_name, location]
                context_stays = traj_seqs['context_stays']
                target_stay = traj_seqs['target_stay']
                historical_stays = traj_seqs['historical_stays']  # [hour, weekday, type_name, location]
                if traj_id not in self.all_predictions[user_id]:
                    self.all_predictions[user_id][traj_id] = None
                true_value = self.true_locations[user_id][traj_id]
                
                if self.llm_method is not None and isinstance(self.llm_method, LLMMoveMethod):
                    # Use LLMMove method for candidate filtering
                    all_pois = list(stay_points.keys())
                    tar_poi = int(true_value['ground_stay'])
                    # Use the last context position for candidate filtering
                    if traj_seqs['context_pos']:
                        last_context_pos = traj_seqs['context_pos'][-1]
                        candidate_poi_info = self.llm_method.filter_candidate_pois(
                            all_pois, last_context_pos, max_distance=5.0, max_candidates=99
                        )
                    print(len(candidate_poi_info))
                
                # Generate prompt using LLM method if available
                if self.llm_method is not None:
                    trajectory_data = {
                        'historical_stays': historical_stays,
                        'context_stays': context_stays,
                        'target_stay': target_stay
                    }
                    
                    if isinstance(self.llm_method, LLMMoveMethod):
                        prompt_text = self.llm_method.generate_prompt(trajectory_data, candidate_poi_info)
                    elif isinstance(self.llm_method, LLMZSMethod):
                        prompt_text = self.llm_method.generate_prompt(trajectory_data)
                    else:
                        # Fallback to original prompt generation
                        prompt_text = prompt_generator(traj_seqs, self.args.model, candidate_poi_info, origin_pred=None, shot=None)
                else:
                    # Use original prompt generation
                    prompt_text = prompt_generator(traj_seqs, self.args.model, candidate_poi_info, origin_pred=None, shot=None)
                
                # 更新内层进度条描述，显示当前预测状态
                traj_pbar.set_postfix({
                    'Predicting': f"{user_id}-{traj_id}",
                    'Model': self.args.model
                })
                
                predictions = self.get_prediction(user_id, traj_id, prompt_text)
                if not isinstance(predictions['prediction'], list):
                    print(predictions['prediction'])
                    continue
                self.all_predictions[user_id][traj_id] = predictions
                
                processed_predictions += 1
                # 更新内层进度条描述
                traj_pbar.set_postfix({
                    'Processed': f"{processed_predictions}/{total_predictions}",
                    'User': user_id,
                    'Traj': traj_id,
                    'Model': self.args.model
                })
            
            # 关闭内层进度条
            traj_pbar.close()
            
            # 更新外层进度条描述
            user_pbar.set_postfix({
                'Users': f"{len(self.all_predictions)}/{len(self.test_dictionary)}",
                'Predictions': processed_predictions,
                'Model': self.args.model
            })
        
        # 关闭外层进度条
        user_pbar.close()
                
        mrr,map_score, ndcg_score, accuracy_top_1, accuracy_top_3, accuracy_top_5 = evaluate(self.all_predictions)
        trial_result = {
            "mrr": mrr,
            "map_score": map_score,
            "ndcg_score": ndcg_score,
            "accuracy_top_1": accuracy_top_1,
            "accuracy_top_3": accuracy_top_3,
            "accuracy_top_5": accuracy_top_5,
            "params": params
        }
        # 获取当前实验的token统计
        token_stats = self._get_experiment_token_stats()
        
        # 将token统计添加到实验结果中
        trial_result.update({
            "token_statistics": token_stats,
            "experiment_info": {
                "session_id": LLMWrapper.get_session_id(),
                "model": self.args.model,
                "dataset": self.args.dataset,
                "city": self.args.city,
                "timestamp": datetime.datetime.now().isoformat()
            }
        })
        
        flag = ''.join(str(uuid.uuid4()).split('-'))
        max_epoch_str = str(int(self.args.max_epoch)) if self.args.max_epoch is not None else "0"
        max_step_str = str(int(self.args.max_step)) if self.args.max_step is not None else "0"
        filename = f"{self.args.model}_{self.args.dataset}_{self.args.city}_{self.args.traj_min_len}_{self.args.traj_max_len}_epoch_{max_epoch_str}_step_{max_step_str}_{self.args.prompt_num}_{flag}"
        with open(os.path.join(self.args.result_path, filename), "w") as f:
            json.dump(trial_result, f)
        
        # 打印实验token统计
        print(f"\n=== 实验Token统计 ===")
        print(f"会话ID: {token_stats.get('session_id', 'N/A')}")
        print(f"总交互次数: {token_stats.get('interaction_count', 0)}")
        print(f"总输入Token: {token_stats.get('total_input_tokens', 0):,}")
        print(f"总输出Token: {token_stats.get('total_output_tokens', 0):,}")
        print(f"总Token消耗: {token_stats.get('total_tokens', 0):,}")
        if token_stats.get('interaction_count', 0) > 0:
            print(f"平均每次交互输入Token: {token_stats.get('average_input_tokens_per_interaction', 0):.1f}")
            print(f"平均每次交互输出Token: {token_stats.get('average_output_tokens_per_interaction', 0):.1f}")
            print(f"平均每次交互总Token: {token_stats.get('average_total_tokens_per_interaction', 0):.1f}")
        
        return accuracy_top_5

    def _get_experiment_token_stats(self):
        """获取当前实验的token统计信息"""
        import jsonlines
        from data_augmentation.utils.base_llm import LLMWrapper
        
        session_id = LLMWrapper.get_session_id()
        if not session_id:
            return {
                "session_id": None,
                "interaction_count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "average_input_tokens_per_interaction": 0,
                "average_output_tokens_per_interaction": 0,
                "average_total_tokens_per_interaction": 0,
                "error": "No session ID found"
            }
        
        # 构建日志文件路径
        model_short_name = self.args.base_model.split("/")[-1] if "/" in self.args.base_model else self.args.base_model
        detailed_log_file = f"{DIAL_RESULT_PATH}/{model_short_name}/{session_id}_detailed_log.jsonl"
        
        if not os.path.exists(detailed_log_file):
            return {
                "session_id": session_id,
                "interaction_count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "average_input_tokens_per_interaction": 0,
                "average_output_tokens_per_interaction": 0,
                "average_total_tokens_per_interaction": 0,
                "error": "Log file not found"
            }
        
        try:
            total_input_tokens = 0
            total_output_tokens = 0
            total_tokens = 0
            interaction_count = 0
            
            with jsonlines.open(detailed_log_file, 'r') as reader:
                for line in reader:
                    if line.get("type") == "interaction" and "prompt_tokens" in line:
                        interaction_count += 1
                        total_input_tokens += int(line.get("prompt_tokens", 0))
                        total_output_tokens += int(line.get("response_tokens", 0))
                        total_tokens += int(line.get("total_tokens", 0))
            
            return {
                "session_id": session_id,
                "interaction_count": interaction_count,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "average_input_tokens_per_interaction": total_input_tokens / max(interaction_count, 1),
                "average_output_tokens_per_interaction": total_output_tokens / max(interaction_count, 1),
                "average_total_tokens_per_interaction": total_tokens / max(interaction_count, 1),
                "log_file": detailed_log_file
            }
            
        except Exception as e:
            return {
                "session_id": session_id,
                "interaction_count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "average_input_tokens_per_interaction": 0,
                "average_output_tokens_per_interaction": 0,
                "average_total_tokens_per_interaction": 0,
                "error": str(e)
            }

    def step(self):
        # Think
        self.inter_turn += 1
        # self.scratchpad += f'\nThought{self.inter_turn}:'
        thought = self.prompt_agent(self.thought)  #从历史交互记录中获得反思，存储在scratchpad中
        parse_thought = self.parse_thought(thought)
        self.scratchpad += f'\nThought{self.inter_turn}:' + parse_thought

        for i in range(self.args.memory_length):
            choices = self.prompt_agent(self.action)  
            result_dict = self.parse_choice(choices) 
            if len(result_dict)>0:
                print(result_dict)
                save_config_param(self.args, result_dict)

                self.scratchpad += f'\nAction{self.inter_turn}:' + str(result_dict)

                # Observe
                self.scratchpad += f'\nObservation{self.inter_turn}: '  #各种index以及效果组合
                index = self.aug_methods_name
                if MODEL_TYPE[self.args.model] == "LLM":
                    score = self.get_pred_llm(result_dict)
                    self.get_memory(self.inter_turn, result_dict, score)
                else:
                    train_model(self.args, index)
                    score, model_config = get_model_result(self.args, index)
                    self.get_memory(self.inter_turn, model_config, score)
                self.format_memory()
            else:
                print("Invalid combination!!!!!!")
                score = 0
                model_config = "Invalid combination"
                self.get_memory(self.inter_turn, model_config, score)
                self.format_memory()
        if EVALUATE_METRIC[MODEL_TYPE[self.args.model]] not in ['MAE']:
            if not self.args.XR:
                best_score = sorted(self.memory_part,key=lambda x:x[-2],reverse=True)[0][-2] #最高分
                best_index = sorted(self.memory_part,key=lambda x:x[-2],reverse=True)[0][3]
            else:
                best_score = score
                best_index = model_config
            if best_score > self.threshold:
                score = best_score #将高于阈值的最低分作为新的阈值
                index = best_index #将对应的配置设置为最佳配置
            else:
                score = self.threshold  #若没有效果更好的尝试，则保持threshold不变
                index = self.threshold_index
        else:
            if not self.args.XR:
                best_score = sorted(self.memory_part,key=lambda x:x[-2])[0][-2]
                best_index = sorted(self.memory_part,key=lambda x:x[-2])[0][3]
            else:
                best_score = score
                best_index = model_config
            if best_score < self.threshold:
                score = best_score 
                index = best_index #将对应的配置设置为最佳配置
            else:
                score = self.threshold  #若没有效果更好的尝试，则保持threshold不变
                index = self.threshold_index
        return index, score
        
    def format_memory(self):
        """
        格式化存储self.memory
        """
        memory = "<MEMORY>\n"
        memory_part = '[' + (',').join(str(item) for item in self.memory_part) + ']'
        self.memory = memory + memory_part

    def get_memory(self, turn_num, indexes, score):
        """
        每个step向memory中添加记录
        """
        if not self.args.XR:
            if EVALUATE_METRIC[MODEL_TYPE[self.args.model]] not in ['MAE']:
                if score > self.threshold:
                    self.memory_part.append(["step num:",turn_num, "hyperparameters combination:",indexes, "score:",score,"experience:this hyperparameters combination seems work well"])
                    # self.scratchpad += "[" + (',').join(str(item) for item in indexes) + ']' + f"The score is {score},which is good enough."
                else:
                    self.memory_part.append(["step num:",turn_num, "hyperparameters combination:",indexes, "score:",score,"experience:this hyperparameters combination seems not good enough,please try other combinations."])
                    # self.scratchpad += "[" + (',').join(str(item) for item in indexes) + ']' + f"The score is {score},which is not good enough."
            else:
                if score < self.threshold:
                    self.memory_part.append(["step num:",turn_num, "hyperparameters combination:",indexes, "score:",score,"experience:this hyperparameters combination seems work well"])
                    # self.scratchpad += "[" + (',').join(str(item) for item in indexes) + ']' + f"The score is {score},which is good enough."
                else:
                    self.memory_part.append(["step num:",turn_num, "hyperparameters combination:",indexes, "score:",score,"experience:this hyperparameters combination seems not good enough,please try other combinations."])
                    # self.scratchpad += "[" + (',').join(str(item) for item in indexes) + ']' + f"The score is {score},which is not good enough."        
        else:
            self.memory_part=[]
        self.format_memory()
        if len(self.memory_part)>self.max_memory or (time.time() - self._last_reflect_time > self.reset_period):
            self.get_reflect()
        
    def get_reflect(self):
        """
        # TODO: 反思机制
        定期更新memory中内容，即开始反思
        """
        print(self.scratchpad)
        self._last_reflect_time = time.time() 

    def prompt_agent(self, part) -> str:  # 输出包含indexes, reason, action
        return self.llm.get_response(prompt_text=self._build_agent_prompt(part))  # self.llm(self._build_agent_prompt(））为基于历史输入采取的动作
    
    def _build_agent_prompt(self, part) -> str:  # 将每一步的结果作为prompt传进去
        question = self.question
        memory = self.memory
        prompt = question + memory + self.scratchpad + part
        return prompt
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        """
        超过最大步数限制，仍未达到优化效果，halted
        """
        return int(self.step_n) > int(self.max_steps) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = '<SCRATCHPAD>:\n'
        self.memory: str = "<MEMORY>:\n"


class ParamReflectAgent(ParamAgent):
    def __init__(self,
                 score: float,
                 index: list,
                 args:  Any,
                 n_cpu: int,
                 enhance: float,
                 max_steps: int,
                 ) -> None:
        self.args = args
        n_cpu = 3, 
        self.enhance = args.enhance
        self.max_steps = args.max_step
        self.model_config = get_config_param(self.args)
        if not self.args.XR:
            self.question = QUESTION.format(config_params=self.model_config, hyparameter_meanings = PARAMS_DESCRIPTION[self.args.model])
            self.thought = THOUGHT
            self.action = ACTION
        else:
            self.question = XR_QUESTION.format(config_params=self.model_config)
            self.thought = XR_THOUGHT
            self.action = XR_ACTION
            
        super().__init__(score, index, n_cpu, args, enhance, max_steps)

    def run(self, reset = True):
        # if not self.is_finished() or self.is_halted():
        #     self.reflect()  #如果之前所有轮的run()运行未得到合理结果，进行反思机制，即reflect
        error_cnt = 0
        correct_cnt = 0
        result_scores = 0
        for i in range(self.args.trial_num):
            try:
                result_indexes, result_score = super().run()  #memory存储历史所有尝试组合和准确率
                result_scores += result_score
                correct_cnt += 1
            except Exception as e:
                print(f"Trial {i} failed: {e}")
                error_cnt += 1
                result_indexes={}
        if correct_cnt==0:
            correct_result=self.threshold
        else:
            correct_result=result_scores/correct_cnt
        print("total counts:correct_rate:", correct_cnt/int(self.args.trial_num), "correct_result:",correct_result)
        return result_indexes, correct_result
