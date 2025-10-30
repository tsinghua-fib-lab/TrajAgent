import os
import random
import time
import json
import pandas as pd
import tqdm
import numpy as np
import multiprocessing as mp
from typing import List, Any
from param_optimize.utils.utils import train_model, get_model_result, get_config_param, save_config_param
from prompt_optimize.utils.utils import *
from prompt_optimize.utils.prompt import ONE_SHOT_PROMPT, OPTIMIZE_PROMPT
from data_augmentation.utils.base_llm import LLMWrapper
from UniEnv.etc.settings import LIMP_DATA_PATH, MODEL_TYPE, LLM_RESULT_PATH
from UniEnv.model_lib.llm_methods import LLMZSMethod, LLMMoveMethod, LIMPMethod

INTENT_A2I_PATH = os.path.join(LIMP_DATA_PATH,"A2I")
SFT_DATA_PATH = os.path.join(LIMP_DATA_PATH, "SFT/")  # 存储模型结果
INTENT_LABEL_PATH = os.path.join(LIMP_DATA_PATH,"intent_label.csv")

os.makedirs(SFT_DATA_PATH, exist_ok=True)

random.seed(114514)

class PromptAgent:
    def __init__(self,
                 score: float,
                 n_cpu: int,
                 args: Any,
                 enhance: float,
                 max_steps: int,
                 select_users: List,
                 ) -> None:
        self.args = args
        self.n_cpu = n_cpu        
        self.max_steps = max_steps
        self.enhance = enhance    #停止条件，(step_score - self.threshold)/self.threshold >= self.enhance
        self.reset_period = 1800   #反思间隔是30min 
        self.threshold = score  #初始值为未经增强的DL模型训练结果
        self.select_users = select_users  #初始值为未经增强的DL模型训练结果
        self._last_reflect_time = time.time()
        
        # Initialize LLM method based on model type
        self.llm_method = self._initialize_llm_method()
        
        self.__reset_agent()

    def _initialize_llm_method(self):
        """Initialize the appropriate LLM method based on model type"""
        config_path = os.path.join(os.getcwd(), "config", "llm_methods")
        os.makedirs(config_path, exist_ok=True)
        
        if self.args.model == "LLMZS":
            return LLMZSMethod(config_path)
        elif self.args.model == "LLMMove":
            return LLMMoveMethod(config_path)
        elif self.args.model == "LIMP":
            return LIMPMethod(config_path)
        else:
            # For other models, return None to use original logic
            return None

    def run(self, reset = True) :
        if reset:
            self.__reset_agent()   #重置agent,从step=1开始,重置memory
        self.inter_turn = 0
        self.llm = LLMWrapper(
                                    temperature=0,
                                    max_tokens=3000,
                                    model_name=self.args.base_model,
                                    platform="DeepInfra",
                                    model_kwargs={"stop": "\n"},
                                    )          
        while not self.is_halted() and not self.is_finished() and int(self.step_n) <= int(self.max_steps):
            step_score = self.step()
            self.scratchpad = '<SCRATCHPAD>:\n'  #清空短期记忆
            self.step_n += 1
            if (step_score - self.threshold)/self.threshold >= self.enhance:  #若某步优化效果超过self.enhance,则可以停止优化，提前结束
                self.finished = True
            self.threshold = max(step_score, self.threshold)  #将每一步的优化结果作为下一步的threshold
        return step_score
    
    def get_prediction(self, user_id, traj_id, prompt_text):
        """Get prediction using LLM method if available, otherwise use original logic"""
        output = dict()
        true_value = self.true_locations[user_id][traj_id]
        tar_poi = int(true_value['ground_stay'])
        pre_text = self.llm.get_response(prompt_text=prompt_text)
        
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
        return predictions

    def get_first_step_pred_llm(self, stay_points):
        """First step prediction using LLM methods"""
        candidate_poi_info = dict()
        self.all_predictions = dict()
        
        for user_id, user_data in self.test_dictionary.items():
            user_id = str(user_id)
            if user_id not in self.all_predictions:
                 self.all_predictions[user_id] = {}
            for traj_id, traj_seqs in user_data.items():
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
                
                predictions = self.get_prediction(user_id, traj_id, prompt_text)
                if not isinstance(predictions['prediction'], list):
                    print(predictions['prediction'])
                    continue
                self.all_predictions[user_id][traj_id] = predictions
                
        mrr,map_score, ndcg_score, accuracy_top_1, accuracy_top_3, accuracy_top_5 = evaluate(self.all_predictions)
        raw_results = {"mrr":mrr,"map_score":map_score, "ndcg_score":ndcg_score, "accuracy_top_1":accuracy_top_1,"accuracy_top_3":accuracy_top_3, "accuracy_top_5":accuracy_top_5}

        with open(os.path.join(SFT_DATA_PATH,f"{self.args.model}_user_{len(self.all_predictions)}_raw_pred.json"),'w') as f:
            json.dump(self.all_predictions, f, ensure_ascii=False)
        with open(self.result_file,'w') as f:
            json.dump(raw_results, f, ensure_ascii=False)
                
    def get_second_step_pred_llm(self, stay_points, shot, user_segments):
        """Second step prediction using LLM methods"""
        candidate_poi_info = dict()
        self.fin_predictions = dict()
        for user_id, user_data in self.test_dictionary.items():
            # 'dyna_id', 'type', 'time', 'entity_id', 'location','Longitude','Latitude','type_name'
            if user_id not in self.fin_predictions:
                user_id = str(user_id)
                self.fin_predictions[user_id] = {}
            for traj_id, traj_seqs in user_data.items():
                traj_id = str(traj_id)
                if traj_id not in self.fin_predictions[user_id]:
                    self.fin_predictions[user_id][traj_id] = None
                true_value = self.true_locations[user_id][traj_id]
                if self.args.model == "LLMMove":
                    all_pois = list(stay_points.keys())
                    tar_poi = int(true_value['ground_stay'])
                    # 5 is the limit for candidate POIs within a 5km range of the correct option, and 99 ensures that each prediction matches the original paper's setting, which is 100 options
                    for poi in all_pois:
                        if haversine_distance(stay_points[poi]['pos'][1],stay_points[poi]['pos'][0],traj_seqs['context_pos'][-1][1],traj_seqs['context_pos'][-1][0]) < 5:
                            candidate_poi_info.update({poi: stay_points[poi]})
                            if len(candidate_poi_info) > 99:
                                break
                    print(len(candidate_poi_info))
                # final prompt
                origin_pred = f"""{{'prediction':{self.all_predictions[user_id][traj_id]['prediction']},'reason':{self.all_predictions[user_id][traj_id]['reason']}}}"""
                prompt_text = prompt_generator(self.test_dictionary[user_id][traj_id], self.args.model, candidate_poi_info, origin_pred, shot) # v, prompt_type, rec, origin_pred, shot
                predictions = self.get_prediction(user_id, traj_id, prompt_text)
                if not isinstance(predictions['prediction'], list):
                    print("*"*20, "wrong format","*"*20)
                    print(predictions['prediction'])
                    continue
                # Construct the filename with model type and save to file
                self.fin_predictions[user_id][traj_id] = predictions
        mrr,map_score, ndcg_score, accuracy_top_1, accuracy_top_3, accuracy_top_5 = evaluate(self.fin_predictions)
        raw_results = {"mrr":mrr,"map_score":map_score, "ndcg_score":ndcg_score, "accuracy_top_1":accuracy_top_1,"accuracy_top_3":accuracy_top_3, "accuracy_top_5":accuracy_top_5}

        # with open(os.path.join(SFT_DATA_PATH,f"{self.args.model}_user_{len(self.all_predictions)}_raw_pred.json"),'w') as f:
        #     json.dump(self.all_predictions, f, ensure_ascii=False)
        with open(self.result_file,'w') as f:
            json.dump(raw_results, f, ensure_ascii=False)
            
        # candidate_poi_info = dict()
        # self.fin_predictions = dict()
        
        # for user_id, user_data in self.test_dictionary.items():
        #     if user_id not in self.fin_predictions:
        #         user_id = str(user_id)
        #         self.fin_predictions[user_id] = {}
        #     for traj_id, traj_seqs in user_data.items():
        #         traj_id = str(traj_id)
        #         if traj_id not in self.fin_predictions[user_id]:
        #             self.fin_predictions[user_id][traj_id] = None
        #         true_value = self.true_locations[user_id][traj_id]
                
        #         if self.llm_method is not None and isinstance(self.llm_method, LLMMoveMethod):
        #             # Use LLMMove method for candidate filtering
        #             all_pois = list(stay_points.keys())
        #             tar_poi = int(true_value['ground_stay'])
        #             # Use the last context position for candidate filtering
        #             if traj_seqs['context_pos']:
        #                 last_context_pos = traj_seqs['context_pos'][-1]
        #                 candidate_poi_info = self.llm_method.filter_candidate_pois(
        #                     all_pois, last_context_pos, max_distance=5.0, max_candidates=99
        #                 )
                
        #         # Generate prompt using LLM method if available
        #         if self.llm_method is not None:
        #             trajectory_data = {
        #                 'historical_stays': traj_seqs.get('historical_stays', []),
        #                 'context_stays': traj_seqs.get('context_stays', []),
        #                 'target_stay': traj_seqs.get('target_stay', [])
        #             }
                    
        #             if isinstance(self.llm_method, LLMMoveMethod):
        #                 prompt_text = self.llm_method.generate_prompt(trajectory_data, candidate_poi_info, shot=shot)
        #             elif isinstance(self.llm_method, LLMZSMethod):
        #                 prompt_text = self.llm_method.generate_prompt(trajectory_data, shot=shot)
        #             else:
        #                 # Fallback to original prompt generation
        #                 prompt_text = prompt_generator(traj_seqs, self.args.model, candidate_poi_info, origin_pred=None, shot=shot)
        #         else:
        #             # Use original prompt generation
        #             prompt_text = prompt_generator(traj_seqs, self.args.model, candidate_poi_info, origin_pred=None, shot=shot)
                
        #         predictions = self.get_prediction(user_id, traj_id, prompt_text)
        #         if not isinstance(predictions['prediction'], list):
        #             print(predictions['prediction'])
        #             continue
        #         self.fin_predictions[user_id][traj_id] = predictions
                
        # mrr,map_score, ndcg_score, accuracy_top_1, accuracy_top_3, accuracy_top_5 = evaluate(self.fin_predictions)
        # raw_results = {"mrr":mrr,"map_score":map_score, "ndcg_score":ndcg_score, "accuracy_top_1":accuracy_top_1,"accuracy_top_3":accuracy_top_3, "accuracy_top_5":accuracy_top_5}
        # with open(os.path.join(SFT_DATA_PATH,f"{self.args.model}_optimize_user_{len(self.fin_predictions)}_raw_pred.json"),'w') as f:
        #     json.dump(self.fin_predictions, f, ensure_ascii=False)
        # with open(os.path.join(self.result_path,f"{self.args.model}_optimize_user_{len(self.fin_predictions)}_raw_pred.json"),'w') as f:
        #     json.dump(raw_results, f, ensure_ascii=False)

    def get_first_step_pred_limp(self, llm_model, feature, user_segments, part):
        """First step prediction for LIMP using LLM method"""
        gpt_save_path = os.path.join(SFT_DATA_PATH, f'finetune_trajectory_annotated_{part}.csv')
        
        for user_id, user_data in user_segments.items():
            # Use LLM method for home/work identification if available
            if self.llm_method is not None and isinstance(self.llm_method, LIMPMethod):
                home, work = self.llm_method.identify_home_and_work(llm_model, user_data['feature'], feature)
            else:
                # Use original logic
                home, work, _ = identify_home_and_work_gpt(llm_model, user_data['feature'], feature, [])
            
            print('Home:', home)
            print('Work:', work)

            for daily_traj in user_data['daily_traj']:
                start=[k for k in daily_traj['start_time']]
                name=daily_traj['POI_name'].values
                trajinfo = ''
                for k in range(len(start)):
                    trajinfo += f'({name[k]},{start[k]})'
                
                # Use LLM method for intent prediction if available
                if self.llm_method is not None and isinstance(self.llm_method, LIMPMethod):
                    predicted_intent = self.llm_method.predict_intent(llm_model, trajinfo, home, work, feature, length=len(start))
                else:
                    # Use original logic
                    predicted_intent = get_predicted_state(llm_model, trajinfo, home, work, feature, [], None, length=len(start))
                
                if not predicted_intent:
                    continue
                daily_traj['predicted_intent'] = predicted_intent[0]
                
                if os.path.exists(gpt_save_path):
                    daily_traj.to_csv(gpt_save_path, index=False, encoding='utf-8', mode='a', header=False)
                else:
                    daily_traj.to_csv(gpt_save_path, index=False, encoding='utf-8', mode='a')
                    
    def get_second_step_pred_limp(self, shot, llm_model, feature, user_segments):
        """Second step prediction for LIMP using LLM method"""
        gpt_save_path = os.path.join(SFT_DATA_PATH, 'optimize_trajectory_annotated.csv')    
        
        for user_id, user_data in user_segments.items():
            target_traj = user_segments[user_id]['daily_traj']
            if self.step_n > 1:
                origin_pred=[k['optimize_intent'].values.tolist() for k in target_traj]
            else:
                origin_pred=[k['predicted_intent'].values.tolist() for k in target_traj]
            answer=[k['intent'].values.tolist() for k in target_traj]
            all_origin = [item for sublist in origin_pred for item in sublist]
            all_answer = [item for sublist in answer for item in sublist]
            acc = 0
            for i, pred in enumerate(all_origin):
                if pred == all_answer[i]:
                    acc += 1
            acc = acc/len(all_answer)  
            
            # Use LLM method for home/work identification if available
            if self.llm_method is not None and isinstance(self.llm_method, LIMPMethod):
                home, work = self.llm_method.identify_home_and_work(llm_model, user_data['feature'], feature)
            else:
                # Use original logic
                home, work, _ = identify_home_and_work_gpt(llm_model, user_data['feature'], feature, [])
            
            print('Home:', home)
            print('Work:', work)

            for idx, daily_traj in enumerate(user_data['daily_traj']):
                start=[k for k in daily_traj['start_time']]
                origin_predict = origin_pred[idx]
                name=daily_traj['POI_name'].values
                trajinfo = ''
                for k in range(len(start)):
                    trajinfo += f'({name[k]},{start[k]})'
                
                if acc > 0.9:
                    predicted_intent = origin_predict
                else:
                    # Use LLM method for intent prediction if available
                    if self.llm_method is not None and isinstance(self.llm_method, LIMPMethod):
                        predicted_intent = self.llm_method.predict_intent(llm_model, trajinfo, home, work, feature, origin_predict, shot, length=len(start))
                    else:
                        # Use original logic
                        predicted_intent = self.get_optimize(llm_model, trajinfo, home, work, feature, origin_predict, shot, length=len(start))
                
                if not predicted_intent:
                    continue
                daily_traj['optimize_intent'] = predicted_intent
                
                if os.path.exists(gpt_save_path):
                    daily_traj.to_csv(gpt_save_path, index=False, encoding='utf-8', mode='a', header=False)
                else:
                    daily_traj.to_csv(gpt_save_path, index=False, encoding='utf-8', mode='a')
                    
    def get_few_shot_limp(self, feature):
        """Get few-shot examples for LIMP using LLM method"""
        gpt_save_path = os.path.join(SFT_DATA_PATH, 'finetune_trajectory_annotated_train.csv')
        if self.prompt_type == "limp":
            data = pd.read_csv(gpt_save_path)
            accuracy = data.groupby('user_id').apply(lambda x: (x['predicted_intent'] == x['intent']).mean())
            result = accuracy.to_dict()
            all_users = list(result.keys())
            all_users.sort()
            target_user = random.sample(all_users,1)[0]
            target_trajs = data[data['user_id']==target_user]
            user_segments = split_user_data_by_date(target_trajs)
            target_traj = user_segments[target_user]['daily_traj']
            start=[k['start_time'].values.tolist() for k in target_traj] 
            shot_length = min(self.args.shot_length ,len(start))
            name=[k['POI_name'].values.tolist() for k in target_traj]
            start = [element for sublist in start for element in sublist][:int(shot_length)]
            name = [element for sublist in name for element in sublist][:int(shot_length)]
            origin_pred=[k['predicted_intent'].values.tolist() for k in target_traj]
            origin_predict = [element for sublist in origin_pred for element in sublist][:int(shot_length)]
            trajinfo = ''
            for k in range(int(shot_length)):
                trajinfo += f'({name[k]},{start[k]})'
            
            # Use LLM method for home/work identification if available
            if self.llm_method is not None and isinstance(self.llm_method, LIMPMethod):
                home, work, _ = self.llm_method.identify_home_and_work(self.llm, user_segments[target_user]['feature'], feature)
                shot = self.llm_method.predict_intent(self.llm, trajinfo, home, work, feature, origin_predict, shot=None, length=len(origin_predict))
            else:
                # Use original logic
                home, work, dial = identify_home_and_work_gpt(self.llm, user_segments[target_user]['feature'], feature, [])
                shot = self.get_optimize(self.llm, trajinfo, home, work, feature, origin_predict, shot=None, length=len(origin_predict))
            
            return shot
        
    def get_few_shot_llm(self, stay_points):
        """Get few-shot examples for LLM methods"""
        candidate_poi_info = dict()
        for user_id, user_info in self.all_predictions.items():
            for traj_id, predictions in user_info.items():
                origin_predict = predictions['prediction']
                true = predictions['true']
                if true not in origin_predict:
                    # prompt_generator(v, prompt_type, rec, origin_pred, shot)
                    traj_seqs = self.test_dictionary[user_id][traj_id]
                    if self.args.model == "LLMMove":
                        all_pois = list(stay_points.keys())
                        # 5 is the limit for candidate POIs within a 5km range of the correct option, and 99 ensures that each prediction matches the original paper's setting, which is 100 options
                        for poi in all_pois:
                            if haversine_distance(stay_points[poi]['pos'][1],stay_points[poi]['pos'][0],traj_seqs['context_pos'][-1][1],traj_seqs['context_pos'][-1][0]) < 5:
                                candidate_poi_info.update({poi: stay_points[poi]})
                                if len(candidate_poi_info) > 99:
                                    break
                        print(len(candidate_poi_info))
                    op_prompt = prompt_generator(traj_seqs, self.args.model, candidate_poi_info, origin_predict, None)
                    predictions = self.get_prediction(user_id, traj_id, op_prompt)
                    if not isinstance(predictions['prediction'], list):
                        print(predictions['prediction'])
                        continue
                    if true in predictions['prediction']:
                        shot = f"""Example:\n{{"input":{predictions['input']},"output":{{"prediction":{predictions['prediction']},"reason":{predictions['reason']}}}}}"""
                        with open(os.path.join(SFT_DATA_PATH,f"{self.args.model}_step_{self.step_n}_user_{len(self.all_predictions)}_shot.json"),'w') as f:
                            json.dump(shot, f, ensure_ascii=False)
                        return shot
        # candidate_poi_info = dict()
        # result_file = os.path.join(self.result_path, f"{self.args.model}_raw_user_{len(self.all_predictions)}_raw_pred.json")
        # with open(result_file) as f:
        #     data = json.load(f)
        # accuracy = data.groupby('user_id').apply(lambda x: (x['predicted_intent'] == x['intent']).mean())
        # result = accuracy.to_dict()
        # all_users = list(result.keys())
        # all_users.sort()
        # target_user = random.sample(all_users,1)[0]
        # target_trajs = data[data['user_id']==target_user]
        # user_segments = split_user_data_by_date(target_trajs)
        # target_traj = user_segments[target_user]['daily_traj']
        # start=[k['start_time'].values.tolist() for k in target_traj] 
        # shot_length = min(self.args.shot_length ,len(start))
        # name=[k['POI_name'].values.tolist() for k in target_traj]
        # start = [element for sublist in start for element in sublist][:int(shot_length)]
        # name = [element for sublist in name for element in sublist][:int(shot_length)]
        # origin_pred=[k['predicted_intent'].values.tolist() for k in target_traj]
        # origin_predict = [element for sublist in origin_pred for element in sublist][:int(shot_length)]
        # trajinfo = ''
        # for k in range(int(shot_length)):
        #     trajinfo += f'({name[k]},{start[k]})'
        
        # # Use LLM method for candidate filtering if available
        # if self.llm_method is not None and isinstance(self.llm_method, LLMMoveMethod):
        #     all_pois = list(stay_points.keys())
        #     # For few-shot, we need to get the context position from the target trajectory
        #     # This is a simplified approach - in practice, you might need to extract this from the actual trajectory data
        #     if hasattr(self, 'test_dictionary') and target_user in self.test_dictionary:
        #         # Try to get context position from test dictionary
        #         context_pos = [0, 0]  # Default position
        #         for traj_id, traj_data in self.test_dictionary[target_user].items():
        #             if traj_data.get('context_pos'):
        #                 context_pos = traj_data['context_pos'][-1]
        #                 break
        #         candidate_poi_info = self.llm_method.filter_candidate_pois(
        #             all_pois, context_pos, max_distance=5.0, max_candidates=99
        #         )
        
        # # Use LLM method for prompt generation if available
        # if self.llm_method is not None:
        #     trajectory_data = {
        #         'historical_stays': target_traj.get('historical_stays', []),
        #         'context_stays': target_traj.get('context_stays', []),
        #         'target_stay': target_traj.get('target_stay', [])
        #     }
            
        #     if isinstance(self.llm_method, LLMMoveMethod):
        #         shot = self.llm_method.generate_prompt(trajectory_data, candidate_poi_info, origin_predict)
        #     elif isinstance(self.llm_method, LLMZSMethod):
        #         shot = self.llm_method.generate_prompt(trajectory_data, origin_predict)
        #     else:
        #         # Fallback to original logic
        #         shot = prompt_generator(target_traj, self.args.model, candidate_poi_info, origin_predict, shot=None)
        # else:
        #     # Use original logic
        #     shot = prompt_generator(target_traj, self.args.model, candidate_poi_info, origin_predict, shot=None)
        
        # return shot

    def get_optimize(self,llm_model, trajectory, home, work,feature, origin_predict, shot=None, length=None):
        """Optimize predictions using LLM method if available"""
        if self.llm_method is not None and isinstance(self.llm_method, LIMPMethod):
            # Use LIMP method for optimization
            return self.llm_method.predict_intent(llm_model, trajectory, home, work, feature, origin_predict, shot, length)
        else:
            # Use original optimization logic
            max_retries = 20
            if not shot:
                fill = ONE_SHOT_PROMPT
            else:
                fill = "Here is a successful case:\n" + str(shot) + OPTIMIZE_PROMPT
            for attempt in range(max_retries):
                Q04 = [dict(role="user", content="""
                Your task is to give intent prediction using trajectory data. Let's think step by step.
                
                1. Analyze the user's behavior pattern based on the trajectory data.
                2. Consider and think about the name of the POI and the time distribution of visits to the POI with the intent. (This is the trajectory of one person, so thinking about the user's daily routine is important.)
                3. Based on the user's behavior pattern and please consider the features of intent 'At Home', 'Working', 'Running errands', predict the intent of each stay in the trajectory data.
                
                The trajectory data under analysis is as follows: {}.
                
                Each stay in trajectory data is represented as (poi, start time).
                
                Here's what each element means:
                - poi: the POI the user visited.
                - start time: the time the user arrived at the POI.
                
                Please judge the function of POI based on its name, time distribution, and features provided. You should take the meaning of each intent as reference, but the final judgment shouldn't be fully rely on that.
                
                Intent you can choose:['At Home', 'Working', 'Running errands', 'Eating Out', 'Leisure and entertainment', 'Shopping']
                
                Here's what each intent means:
                - At Home: When the user is at {}, it is mostly considered as being at home. And Other places are NOT considered as home! 
                - Working: When the user is at {}, it is mostly considered as working. And Other places are NOT considered as working!
                But, you should still consider the user\'s behavior pattern, POI_name, and the time the user visited the POI.
                
                Here are some unique features of intent 'At Home', 'Working', 'Running errands': 
                {}
                
                The origin predict intent is {}, which is not good enough.{}
            
                Note: If multiple conditions are met, priority should be given to 'At Home' and 'Running Errands'.
                
                There are {} stays in the trajectory data. So, the output should have {} predicted intents.
                
                Consider step by step, finally respond using the following JSON format (Make sure to have one predicted intent for each stay in the trajectory data, And you have to assign one of the intents to each stay in the trajectory data):
                {{
                "optimize_intent": ["adjusted predicted intents"],
                }}
                """.format(trajectory, home, work, feature, origin_predict, fill, length, length
                        ))  ## Attempt to change the prompt: "regardless of the time and POI category" is a bit too absolute, you might consider adding judgments for time and POI category.
                    ]
                answer04 = llm_model(Q04[0]['content'])
                 # answer04 = answer04.strip()
                start_pos = answer04.find('{')
                end_pos = answer04.rfind('}') + 1

                # extract json string
                if start_pos != -1 and end_pos != -1:
                    json_str = answer04[start_pos:end_pos]
                    # extract matrix and remove comments

                    cleaned_json_str = clean_json_str(json_str)

                    try:
                        # extract matrix
                        data = json.loads(cleaned_json_str)
                        optimize_intent = data.get("optimize_intent")
                        if optimize_intent is not None:
                            if len(optimize_intent) == length:
                                print("Data loaded successfully:", optimize_intent)
                                fin_shot = f"""{{user:The trajectory data under analysis is as follows: {trajectory}\nAt Home: When the user is at {home}, it is mostly considered as being at home. And Other places are NOT considered as home.\nWorking: When the user is at {work}, it is mostly considered as working. And Other places are NOT considered as working!\n
                                The origin predict intent is {origin_predict}, which is not good enough.\nHere are some unique features of intent 'At Home', 'Working', 'Running errands': {feature}\n{ONE_SHOT_PROMPT}\noptimize_intent:{optimize_intent}}}"""
                                if not shot:
                                    return fin_shot
                                else:
                                    return optimize_intent
                            else:
                                print("The length of the adjusted prediction does not match the original prediction.")
                                print(f"Retrying... ({attempt + 1}/{max_retries})")
                        else:
                            print("Failed to find 'optimize_intent' in the JSON data.")
                    except json.JSONDecodeError as e:
                        print(f"JSON Decode Error: {str(e)}")
                        print(f"Retrying... ({attempt + 1}/{max_retries})")
                else:
                    print("JSON data not found in the text")
                    print(f"Retrying... ({attempt + 1}/{max_retries})")

    def get_trajectories(self, user_segments, train_user_segments):
        """Get trajectories for prediction with consistent sampling"""
        self.test_dictionary = {}
        self.true_locations = {}
        # For each user, we retrieve the list of trajectory ids:
        # - the first 80% of the trajectory ids are used for training
        # - the last 20% of the trajectory ids are used for testing
        print('Processing training and test split using method trajectory_split...')
        # for each trajectory id in test trajectory ids, we retrieve the historical stays
        # (location ids and time from training) and the context stays (location ids and time
        # from testing with specific trajectory id)
        counter = 0
        
        # 保证user是有序的，便于重复试验
        user_list = [str(y) for y in sorted([int(x) for x in list(user_segments.keys())])]
        
        for user in user_list:
            trajectories = user_segments[user]
            traj_count = 0
            
            if len(trajectories) < self.traj_min_len:
                continue 
            elif len(trajectories) > self.traj_max_len:
                continue 
                
            if user not in self.test_dictionary:
                self.test_dictionary[str(user)] = {}
            if user not in self.true_locations:
                self.true_locations[str(user)] = {}
            
            # 保证traj_id是有序的，便于重复试验
            traj_ids = [str(y) for y in sorted([int(x) for x in list(trajectories.keys())])]
            
            # Process each trajectory (list of location points)
            for traj_id in traj_ids:
                trajectory = trajectories[traj_id]
                traj_count += 1
                
                # 限制每个用户的轨迹数量
                if traj_count > self.max_sample_trajectories:
                    break
                    
                if len(trajectory) < 2:  # Need at least 2 points for context and target
                    continue
                    
                counter += 1
                
                if traj_id not in self.test_dictionary[str(user)]:
                    self.test_dictionary[str(user)][traj_id] = {}
                if traj_id not in self.true_locations[str(user)]:
                    self.true_locations[str(user)][traj_id] = {}
                
                # Convert trajectory points to the expected format
                # Each point: [traj_id, type, hour_24h, weekday, entity_id, location, longitude, latitude, type_name]
                # Convert to: [hour, weekday, type_name, location] for historical/context stays
                # And: [longitude, latitude] for positions
                
                # Split trajectory into context (all but last) and target (last)
                context_points = trajectory[:-1]
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
                
                # Store in test dictionary
                self.test_dictionary[str(user)][traj_id] = {
                    'historical_stays': historical_stays[-self.history_stays:] if hasattr(self, 'history_stays') else historical_stays,
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
            if counter >= self.prompt_num:
                print("已经采集到足够的数据，要求轨迹:{} 实际轨迹:{}".format(self.prompt_num, counter))
                break
                
        if counter < self.prompt_num:
            print("数据不足，要求轨迹:{} 实际轨迹:{}".format(self.prompt_num, counter))

    def trajs_sampling(self, test_dataset):
        """Sample trajectories for testing"""
        user_segments = {}
        for user_id, user_trajectories in test_dataset.items():
            if str(user_id) not in self.select_users:
                continue
            if user_id not in user_segments:
                user_segments[user_id] = {}
            
            # Each user has a list of trajectories
            for traj_idx, trajectory in enumerate(user_trajectories):
                if len(trajectory) < self.traj_min_len or len(trajectory) > self.traj_max_len:  # Need at least 2 points
                    continue
                    
                # Extract trajectory ID from the first point
                # Data format: [traj_id, type, hour_24h, weekday, entity_id, location, longitude, latitude, type_name]
                traj_id = str(trajectory[0][0]) if trajectory else str(traj_idx)
                
                # Store the trajectory as-is for processing in get_trajectories
                user_segments[user_id][traj_id] = trajectory
                
        return user_segments

    def step(self):
        """Execute one optimization step"""
        if self.step_n == 1:
            self.result_file = os.path.join(self.result_path, f"{self.args.model}_raw_pred.json")
        else:
            self.result_file = os.path.join(self.result_path, f"{self.args.model}_optimize_{self.step_n}_pred.json")
        
        # Load test data
        test_dataset = json.load(open(os.path.join(self.args.aug_data_path, "test_1000_1000.json")))
        user_segments = self.trajs_sampling(test_dataset)
        
        # Load training data for historical context
        train_dataset = json.load(open(os.path.join(self.args.aug_data_path, "train_1000_1000.json")))
        train_user_segments = self.trajs_sampling(train_dataset)
        
        # Get trajectories
        self.get_trajectories(user_segments, train_user_segments)
        
        # Load stay points for LLMMove
        if self.args.model == "LLMMove":
            stay_points = json.load(open(os.path.join(self.args.aug_data_path, "stay_points.json")))
        else:
            stay_points = None
        
        # Execute first step prediction
        if self.step_n == 1:
            if self.args.model in ["LLMMove", "LLMZS"]:
                self.get_first_step_pred_llm(stay_points)
            elif self.args.model == "LIMP":
                feature = "Features of intent 'At Home', 'Working', 'Running errands'"
                self.get_first_step_pred_limp(self.llm, feature, user_segments, "train")
        else:
            # Execute second step prediction with few-shot examples
            if self.args.model in ["LLMMove", "LLMZS"]:
                shot = self.get_few_shot_llm(stay_points)
                self.get_second_step_pred_llm(stay_points, shot, user_segments)
            elif self.args.model == "LIMP":
                feature = "Features of intent 'At Home', 'Working', 'Running errands'"
                shot = self.get_few_shot_limp(feature)
                self.get_second_step_pred_limp(shot, self.llm, feature, user_segments)
        
        # Calculate step score    
        if os.path.exists(self.result_file):
            with open(self.result_file, 'r') as f:
                results = json.load(f)
                step_score = results.get("accuracy_top_5", 0.0)
        else:
            step_score = 0.0
        
        return step_score

    def get_reflect(self):
        """Get reflection for optimization"""
        return "Reflection on optimization progress"

    def is_finished(self) -> bool:
        """Check if optimization is finished"""
        return self.finished

    def is_halted(self) -> bool:
        """Check if optimization is halted"""
        return time.time() - self._last_reflect_time > self.reset_period

    def __reset_agent(self) -> None:
        """Reset agent state"""
        self.step_n = 1
        self.finished = False
        self.scratchpad = ""

class PromptReflectAgent(PromptAgent):
    def __init__(self,
                 select_users: List,
                 score: float,
                 args:  Any,
                 n_cpu: int,
                 enhance: float,
                 max_steps: int,
                 ) -> None:
        self.args = args
        n_cpu = 3, 
        self.traj_min_len = 3
        self.traj_max_len = 100
        self.history_stays=40
        self.context_stays=6
        self.prompt_num = 200  # 200
        self.max_sample_trajectories = 100 # 100
        self.sample_one_traj_of_user = True
        self.enhance = enhance
        self.max_steps = max_steps
        self.select_users = select_users
        self.score = score       
        super().__init__(score, n_cpu, args, enhance, max_steps, select_users)
        self.result_path = os.path.join(LLM_RESULT_PATH, args.task, args.base_model)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def run(self, reset = True) -> None:
        # if not self.is_finished() or self.is_halted():
        #     self.reflect()  #如果之前所有轮的run()运行未得到合理结果，进行反思机制，即reflect
        return super().run(reset)
