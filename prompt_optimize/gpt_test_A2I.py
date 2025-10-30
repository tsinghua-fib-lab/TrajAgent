import re
import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse
from UniEnv.etc.settings import LIMP_DATA_PATH, LLM_MAP
from data_augmentation.utils.base_llm import AnyOpenAILLM, DEEPINFRA
INTENT_LABEL_PATH = os.path.join(LIMP_DATA_PATH,"intent_label.csv")
INTENT_A2I_PATH = os.path.join(LIMP_DATA_PATH,"A2I")

os.makedirs(INTENT_A2I_PATH, exist_ok=True)


dialogs = []

def calculate_home_and_work(df):
    home_df = df[df['intent_en'] == 'At Home']
    work_df = df[df['intent_en'] == 'Working']
    runningerrands_df = df[df['intent_en'] == 'Running errands']
    
    # Calculate percentage distribution for each intent
    total_intents = len(df)
    home_percentage = round(len(home_df) / total_intents * 100,2)
    work_percentage = round(len(work_df) / total_intents * 100,2)
    runningerrands_percentage = round(len(runningerrands_df) / total_intents * 100,2)
    
    # Calculate time distribution and duration
    def calculate_time_distribution_and_duration(intent_df, intent):
        intent_df['start_time'] = pd.to_datetime(intent_df['start_time'])
        intent_df['time_bin'] = intent_df['start_time'].dt.hour
        time_distribution = intent_df.groupby(['time_bin', 'intent_en']).size().unstack().fillna(0)
        time_distribution = time_distribution.div(time_distribution.sum(axis=1), axis=0) * 100
        
        
        intent_df_intent = intent_df[intent_df['intent_en'] == intent]
        intent_df_intent['start_time'] = pd.to_datetime(intent_df_intent['start_time'])
        
        return round(time_distribution,2)
    time_distribution = calculate_time_distribution_and_duration(df, 'At Home')
    
    
    work_df['date'] = pd.to_datetime(work_df['start_time']).dt.date
    w = work_df.groupby(['user_id','date','POI_name']).size().reset_index(name='counts')
    work_avg_visit = sum(w['counts'])/len(w)
    
    home_df['date'] = pd.to_datetime(home_df['start_time']).dt.date
    h = home_df.groupby(['user_id','date','POI_name']).size().reset_index(name='counts')
    home_avg_visit = sum(h['counts'])/len(h)
    
    runningerrands_df['date'] = pd.to_datetime(runningerrands_df['start_time']).dt.date
    r = runningerrands_df.groupby(['user_id','date','POI_name']).size().reset_index(name='counts')
    runningerrands_avg_visit = sum(r['counts'])/len(r)
    

    result = {
        'At Home': {
            'percentage_distribution': home_percentage,
            'average_visit': home_avg_visit
        },
        'Working': {
            'percentage_distribution': work_percentage,
            'average_visit': work_avg_visit
        },
        'Running errands': {
            'percentage_distribution': runningerrands_percentage,
            'average_visit': runningerrands_avg_visit
        },
        'Time Distribution of Intents':  time_distribution.to_dict() 
    }    
    
    return result


def round_to_half_hour_15(t):
    t=datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    minutes = (t.minute // 15) * 15
    if t.minute % 15 >= 8:
        minutes += 15
    if minutes >= 60:
        t = t + timedelta(hours=1)
        minutes = 0
    return t.replace(minute=minutes, second=0)


def split_user_data_by_date(df):
    # Split data for each user into daily segments and calculate the frequency percentage for each POI.
    temp_df = df.copy()

    # Ensure the start time is in datetime format only for processing
    temp_df['start_time_dt'] = pd.to_datetime(temp_df['start_time'])
    temp_df['hour'] = temp_df['start_time_dt'].dt.hour

    # Extract date from the temporary start_time_dt
    temp_df['date'] = temp_df['start_time_dt'].dt.date

    user_segments = {}
    user_groups = temp_df.groupby('user_id')

    for user_id, group in user_groups:
        group = group.sort_values(by='start_time_dt')  # Ensure the group is sorted by time
        daily_segments = []
        date_groups = group.groupby('date')
        
        user_data = []
        poi_groups = group.groupby('POI_name')
        total_cnt = user_groups.get_group(user_id).shape[0]
        # Debugging: Print each POI group
        for poi, poi_group in poi_groups:

            poi_data = {}
            total_count = len(poi_group)  # Total visits to this POI
            cnt=poi_group.shape[0]
            time_group = poi_group.groupby('hour')
            time_list = []
            for time, time_group in time_group:
                time_list.append((f'{time}:00', str(round((time_group.shape[0] / cnt) * 100, 1)) + '%'))
            poi_data['Name'] = poi
            poi_data['Percent'] = cnt/total_cnt*100
            poi_data['Time Distribution'] = time_list
            user_data.append(poi_data)
        
        for i in range(len(user_data)):
            user_data[i]['Percent']=str(round(user_data[i]['Percent'],1))+'%'
        for date, date_group in date_groups:
            daily_segments.append(date_group.drop(columns=['start_time_dt', 'date']))  # drop temporary columns

        user_segments[user_id] = {'feature': user_data, 'daily_traj': daily_segments}

    return user_segments


def clean_json_str(json_str):
    json_str=re.sub(r'//.*|#[^\n]*', '', json_str)
    return re.sub('\n','',json_str)

def get_json(answer):
    start_pos = answer.find('{')
    end_pos = answer.rfind('}') + 1

    # extract json string
    if start_pos != -1 and end_pos != -1:
        json_str = answer[start_pos:end_pos]

        try:
            # extract matrix
            data = json.loads(json_str)
            return data
        except:
            return None
    else:
        return None


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


def get_home_and_work_feature(llm_model, int_dict):
    # Use GPT to extract the characteristics of home and workplace from statistical data
    max_retries = 20
    for attempt in range(max_retries):
        Q5 = [dict(role="user", content="""
        
        Your task is to extract the features of intent \'At Home\', \'Working\', \'Running errands\' from the statistical data. Please think step by step.
                   
        Here's the statistical data of the user's intent distribution:{}
        
        The meanings of statistical data are as follows:
        - Percentage Distribution: The percent of the intent in the whole dataset.
        - Average Visit: The average number of visits to the POI with the intent.
        - Time Distribution: The start_time distribution of visits to the POI with the intent, in the format of (start hour: percentage).
         
        There are 6 intents in total: ['At Home', 'Working', 'Running errands', 'Eating Out', 'Leisure and entertainment', 'Shopping'], each intent has a percentage distribution and a time distribution.
        
        Instruction:
        - You need to extract the unique and prominent features of intent 'At Home', 'Working', 'Running errands' which can distinguish them from other intents.
        - Each intent should have about 6-8 features.
        - Should be based on the percentage distribution, and time distribution of the intent.
        - Should be able to help identify the user's home, work place, and running errands place.
        - Some features need to be specificity to the intent, such as the time distribution of the intent.
        
        Answer using the following JSON format:
        {{
        "features": ["features of 'intents'"],
            }}
           
        """.format(int_dict)
        )]
        answer5 = llm_model(Q5[0]["content"])
        dialogs.append({"role": "user", "content": Q5[0]["content"]})
        dialogs.append({"role": "assistant", "content": answer5})
        home_and_work_feature = get_json(answer5)
        if home_and_work_feature is not None:
            return home_and_work_feature
        else:
            print(f"Retrying... ({attempt + 1}/{max_retries})")
    return None


def identify_home_and_work_gpt(llm_model, userdata,feature):
    # Use GPT to identify home and work locations
    max_retries = 20
    for attempt in range(max_retries):
        Q6 = [dict(role="user", content="""Your task is to identify the user's home and work place based on the trajectory data and the features of intent 'At Home' and 'Working'.
        The trajectory data under analysis is as follows: {}.
        Each entry represents a POI-intent pair that the user has visited.
        The meanings of each feature are as follows:
        - Name: POI name
        - Percent: The percentage of times the behavior pattern occurred
        - Time Distribution: The time distribution of visits to the POI with the intent, in the format of (hour, percentage).
        Here are the general and unique features of intent 'At Home' , 'Working' , 'Running errands':{}
        Respond using the following JSON format:
        {{
        "home": "home place",
        "work": "work place"
        "reason": "reason for prediction"
        }}
                   """.format(userdata, feature))]
        answer6 = llm_model(Q6[0]["content"])
        dialogs.append({"role": "user", "content": Q6[0]["content"]})
        dialogs.append({"role": "assistant", "content": answer6})
        home_and_work = get_json(answer6)
        if home_and_work is not None:
            return home_and_work['home'], home_and_work['work']
        else:
            print(f"Retrying... ({attempt + 1}/{max_retries})")
    return None, None


def get_predicted_state(llm_model, trajectory, home, work,feature,
                        predict_state=None, length=None):
    # Use GPT to annotate Intent
    max_retries = 20
    for attempt in range(max_retries):
        # Corresponding intent predictions are provided in {}.
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
    
        Note: If multiple conditions are met, priority should be given to 'At Home' and 'Running Errands'.
        
        There are {} stays in the trajectory data. So, the output should have {} predicted intents.
        
        Consider step by step, finally respond using the following JSON format (Make sure to have one predicted intent for each stay in the trajectory data, And you have to assign one of the intents to each stay in the trajectory data):
        {{
        "predicted_intent": ["adjusted predicted intents"],
        }}
        """.format(trajectory, home, work, feature, length, length
                   ))  ## Attempt to change the prompt: “regardless of the time and POI category” is a bit too absolute, you might consider adding judgments for time and POI category.
               ]
        answer04 = llm_model(Q04[0]["content"])
        dialogs.append({"role": "user", "content": Q04[0]["content"]})
        dialogs.append({"role": "assistant", "content": answer04})
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
                predicted_intent = data.get("predicted_intent")
                if predicted_intent is not None:
                    if predict_state:
                        if len(predicted_intent) == length:
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
                        print(f"Retrying... ({attempt + 1}/{max_retries})")
                else:
                    print("Failed to find 'predicted_intent' in the JSON data.")
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {str(e)}")
                print(f"Retrying... ({attempt + 1}/{max_retries})")
        else:
            print("JSON data not found in the text")
            print(f"Retrying... ({attempt + 1}/{max_retries})")


if __name__ == '__main__':
    # Load the dataset
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
    
    gpt_ft_path = os.path.join(INTENT_A2I_PATH, 'gpt_finetune_updated.csv')
    hw_path = os.path.join(INTENT_A2I_PATH, 'homeandwork.csv')

    feature = get_home_and_work_feature(llm_model, calculate_home_and_work(pd.read_csv(INTENT_LABEL_PATH)))
    user_segments = split_user_data_by_date(pd.read_csv(INTENT_LABEL_PATH))
    
    ints = []
    cnt = 0
    with open(os.path.join(INTENT_A2I_PATH, 'feature_w_workflow.txt'), 'w') as file:
       file.write(str(feature))

    for user_id, user_data in tqdm(user_segments.items(), desc="Processing Users"):
        # Use GPT to identify home and work locations
        home, work = identify_home_and_work_gpt(llm_model, user_data['feature'],feature)
        
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
            intentlist = daily_traj['intent_en'].values # true intent
            trajinfo = ''
            for k in range(len(start)):
                
                trajinfo += f'({name[k]},{start[k]})'
            predicted_intent = get_predicted_state(llm_model, trajinfo, home, work, feature,
                                                   length=len(intentlist)) # predicted intent
            
            with open(os.path.join(INTENT_A2I_PATH, 'predicted_intent.txt'), 'a') as file:
                file.write(str(predicted_intent))
                
            with open(os.path.join(INTENT_A2I_PATH, 'true_intent.txt'), 'a') as file:
                for i in intentlist:
                    file.write(str(i) + ',')
                
            for i in range(len(intentlist)):
                if intentlist[i] == predicted_intent[i]:
                    acc+=1
                total+=1
                current_acc = acc / total
                
                # Print accuracy change
                if current_acc > prev_acc:
                    print(f"Accuracy increased: {prev_acc:.4f} -> {current_acc:.4f}")
                elif current_acc < prev_acc:
                    print(f"Accuracy decreased: {prev_acc:.4f} -> {current_acc:.4f}")
                else:
                    print(f"Accuracy remained the same: {current_acc:.4f}")
                
                prev_acc = current_acc
            
            daily_traj['predicted_intent'] = predicted_intent
            
            if os.path.exists(gpt_ft_path):
                daily_traj.to_csv(gpt_ft_path, index=False, encoding='utf-8', mode='a', header=False)
            else:
                daily_traj.to_csv(gpt_ft_path, index=False, encoding='utf-8', mode='a')
            print(cnt)
            
            print('Actual intents:', intentlist)
            print('Accuracy:', current_acc)
        cnt+=1
