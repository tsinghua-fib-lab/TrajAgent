import re
import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from UniEnv.etc.settings import LIMP_DATA_PATH, LLM_MAP
from data_augmentation.utils.base_llm import AnyOpenAILLM, DEEPINFRA
from prompt_optimize.utils.utils import split_user_data_by_date, identify_home_and_work_gpt, clean_json_str, get_predicted_state, get_home_and_work_feature, calculate_home_and_work
import argparse
import random
random.seed(114514)

INTENT_LABEL_PATH = os.path.join(LIMP_DATA_PATH,"intent_label.csv")
INTENT_A2I_PATH = os.path.join(LIMP_DATA_PATH,"A2I")
INTENT_LOG_PATH = os.path.join(LIMP_DATA_PATH,"A2I/logs/")
PRE_DATA_PATH = os.path.join(LIMP_DATA_PATH, "Predict/")
SFT_DATA_PATH = os.path.join(LIMP_DATA_PATH, "SFT/")  # 存储模型结果

os.makedirs(SFT_DATA_PATH, exist_ok=True)

# parse_dict = self.extract_llm(parse_dict_prompt)
dialogs = []



def round_to_half_hour_15(t):
    t=datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    minutes = (t.minute // 15) * 15
    if t.minute % 15 >= 8:
        minutes += 15
    if minutes >= 60:
        t = t + timedelta(hours=1)
        minutes = 0
    return t.replace(minute=minutes, second=0)





def get_json_intent(answer, length, predict_state):
    start_pos = answer.find('{')
    end_pos = answer.rfind('}') + 1

    # extract json string
    if start_pos != -1 and end_pos != -1:
        json_str = answer[start_pos:end_pos]
        # extract matrix and remove comments
        json_str = clean_json_str(json_str)
        try:
            # extract matrix
            data = json.loads(json_str)
            predicted_intent = data.get("predicted_intent")
            if predicted_intent is not None:
                if predict_state:
                    if len(predicted_intent) == len(predict_state):
                        print("Data loaded successfully:", predicted_intent)
                        return predicted_intent
                elif length:
                    if len(predicted_intent) == length:
                        print("Data loaded successfully:", predicted_intent)
                        return predicted_intent
                elif not predict_state and not length:
                    print("Warning: The length of the adjusted prediction cannot be checked.")
                    print("Data loaded successfully:", predicted_intent)
                    return predicted_intent
                else:
                    print("The length of the adjusted prediction does not match the original prediction.")
            else:
                print("Failed to find 'predicted_intent' in the JSON data.")
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")
    else:
        print("JSON data not found in the text")
    return None





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--user_num', type=int, default=20)
    args = parser.parse_args()
    # Load the dataset
    if "gpt" in args.base_model:
        llm_model = AnyOpenAILLM(
                        temperature=0,
                        max_tokens=3000,
                        model_name=LLM_MAP[args.base_model],
                        model_kwargs={"stop": "\n"},
                        openai_api_key=os.environ['OPENAI_API_KEY'])
    else:
        llm_model = DEEPINFRA(
                        temperature=0,
                        model_name=LLM_MAP[args.base_model],
                        max_tokens=3000,
                        model_kwargs={"stop": "\n"},
                        openai_api_key=os.environ['DEEPINFRA_API_KEY'])
    
    total=0
    acc=0
    
    prev_acc = 0
    
    gpt_ft_path = os.path.join(INTENT_A2I_PATH, 'finetune_trajectory.csv')
    gpt_save_path = os.path.join(SFT_DATA_PATH, 'finetune_trajectory_annotated.csv')
    hw_path = os.path.join(SFT_DATA_PATH, 'finetune_homeandwork.csv')
    # Get temporal features for each intent with gpt
    feature, dialogs = get_home_and_work_feature(llm_model, calculate_home_and_work(pd.read_csv(INTENT_LABEL_PATH)), dialogs)
    # For each user, compute temporal feature for each POI,and get daily trajectories.
    user_segments = split_user_data_by_date(pd.read_csv(gpt_ft_path))
    
    ints = []

    all_users = list(user_segments.keys())
    all_users.sort()
    select_users = random.choices(all_users, k=args.user_num)
    user_segments = {key:user_segments[key] for key in select_users}

    for user_id, user_data in tqdm(user_segments.items(), desc="Processing Users"):
        # Use GPT to identify home and work locations
        home, work, dialogs = identify_home_and_work_gpt(llm_model, user_data['feature'],feature, dialogs)
        
        print('Home:', home)
        print('Work:', work)
        homeandwork = pd.DataFrame({'user_id': [user_id], 'home': [home], 'work': [work]})
        
        if os.path.exists(hw_path):
            homeandwork.to_csv(hw_path, index=False, encoding='utf-8', mode='a', header=False)
        else:
            homeandwork.to_csv(hw_path, index=False, encoding='utf-8', mode='a')

        for daily_traj in tqdm(user_data['daily_traj'], desc="Processing Daily Trajectories"):
            start=[k for k in daily_traj['start_time']]
            
            name=daily_traj['POI_name'].values
            trajinfo = ''
            for k in range(len(start)):
                trajinfo += f'({name[k]},{start[k]})'
            # Based on the predicted work and home place, further consider the intent for each POI in trajectory
            predicted_intent = get_predicted_state(llm_model, trajinfo, home, work, feature, dialogs,
                                                   length=len(start)) # predicted intent
            
            daily_traj['predicted_intent'] = predicted_intent
            
            if os.path.exists(gpt_save_path):
                daily_traj.to_csv(gpt_save_path, index=False, encoding='utf-8', mode='a', header=False)
            else:
                daily_traj.to_csv(gpt_save_path, index=False, encoding='utf-8', mode='a')