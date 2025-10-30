import json
import pandas as pd
import re
from math import radians, sin, cos, sqrt, atan2
import jsmin
import numpy as np

from prompt_optimize.utils.prompt import ONE_SHOT_PROMPT, OPTIMIZE_PROMPT

def printQA(Q, A):
    print('Question:', Q)
    print('Answer:', A+'\n')
    
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
def evaluate(predictions):
    correct_top_1 = 0
    correct_top_3 = 0
    correct_top_5 = 0
    mrr = 0
    ndcg_sum = 0
    ap_sum = 0
    num_queries = 0
    for user, trajs in predictions.items():
        for traj_id, prediction in trajs.items():
            try:
                prediction_values = prediction['prediction']
                true_value = prediction['true']
            # Check if true value matches predictions for different top-n accuracies
                if true_value in prediction_values:
                    if true_value == prediction_values[0]:  # Top-1 accuracy
                        correct_top_1 += 1
                    if true_value in prediction_values[:3]:  # Top-3 accuracy
                        correct_top_3 += 1
                    if true_value in prediction_values[:5]:  # Top-5 accuracy
                        correct_top_5 += 1
                    rank = prediction_values.index(true_value) + 1
                    mrr += 1 / rank
                    ap_sum += 1 / rank  # AP simplifies to 1/rank when only one relevant document exists

                    # Calculate DCG
                    dcg = (2 ** 1 - 1) / np.log2(rank + 1)  # rel_i for the correct answer is 1, others are 0
                    idcg = (2 ** 1 - 1) / np.log2(1 + 1)    # Ideal DCG if the first item is correct
                    ndcg = dcg / idcg
                    ndcg_sum += ndcg
                else:
                    ndcg_sum += 0  # NDCG is 0 if correct answer is not in predictions
                    mrr += 0
                    ap_sum += 0
                num_queries += 1
            except:
                correct_top_1 += 0
                correct_top_3 += 0
                correct_top_5 += 0
                ndcg_sum += 0
                mrr += 0
                ap_sum += 0
    if num_queries > 0:
        mrr = mrr / num_queries
        map_score = ap_sum / num_queries
        ndcg_score = ndcg_sum / num_queries
        # Calculate accuracies
        accuracy_top_1 = correct_top_1 / num_queries 
        accuracy_top_3 = correct_top_3 / num_queries 
        accuracy_top_5 = correct_top_5 / num_queries 
    else:
        mrr, map_score, ndcg_score = 0, 0, 0
        # When no queries, all accuracies are 0
        accuracy_top_1, accuracy_top_3, accuracy_top_5 = 0, 0, 0
    return mrr,map_score, ndcg_score, accuracy_top_1, accuracy_top_3, accuracy_top_5

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

def prompt_generator_llmmob(v, origin_pred, shot):
    if shot:
        op_part = "Here is a successful case:\n" + str(shot) + OPTIMIZE_PROMPT
    elif origin_pred:
        op_part = f"The origin prediction is {origin_pred}, which is not good enough." + ONE_SHOT_PROMPT
    else:
        op_part = ""
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. Note that this will be None in the <target_stay> introduced later.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      

    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you learned from <history>, e.g., repeated visits to certain places during certain times;
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the five most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <historical>[Format: (start_time, day_of_week, POIID)]: {[[item[0], item[1], item[3]] for item in v['historical_stays']]}
    <context>: {[[item[0], item[1], item[3]] for item in v['context_stays']]}
    <target_stay>: {[v['target_stay'][0], v['target_stay'][1]]}
    {op_part}
    """
    return prompt

def prompt_generator_llmmove(v, rec, origin_pred, shot):
    if shot:
        op_part = "Here is a successful case:\n" + str(shot) + OPTIMIZE_PROMPT
    elif origin_pred:
        op_part = f"The origin prediction is {origin_pred}, which is not good enough." + ONE_SHOT_PROMPT
    else:
        op_part = ""
    prompt =f"""\
    <long-term check-ins> [Format: (POIID, Category)]: {[(item[3],item[2]) for item in v['historical_stays']]}
    <recent check-ins> [Format: (POIID, Category)]: {[(item[3],item[2]) for item in v['context_stays']]}
    <candidate set> [Format: (POIID, Distance, Category)]: {[(item['poi'],  haversine_distance(item['pos'][1],item['pos'][0],v['context_pos'][-1][1],v['context_pos'][-1][0]), item['cat']) for _, item in rec.items()]}
    Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based on his/her trajectory information.
    The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
    Now I explain the elements in the format. "POIID" refers to the unique id of the POI, "Distance" indicates the distance (kilometers) between the user and the POI, and "Category" shows the semantic information of the POI.

    Requirements:
    1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
    2. Consider the recent check-ins to extract users' current perferences.
    3. Consider the "Distance" since people tend to visit nearby pois.
    4. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.

    Please organize your answer in a JSON object containing following keys:
    "prediction" (10 distinct POIIDs of the ten most probable places in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). Do not include line breaks in your output.
    {op_part}
    """
    return prompt

def match_prediction(text):
    match = re.search(r'[Pp]rediction(.*?)[Rr]eason', text, re.DOTALL)
    if match:
        prediction_text = match.group(1)
        # 在这段文本中提取所有ID
        place_ids = re.findall(r'\b[0-9a-f]{24}\b', prediction_text)
    else:
        place_ids = []
    return place_ids

def extract_json(full_text):
        # Attempt to load as JSON
        # TODO: we can use json_pair to repair invalid JSON https://github.com/mangiucugna/json_repair
        # we can use jsmin to remove comments in JSON https://github.com/tikitu/jsmin/
        if not isinstance(full_text, str):
            output_json = {
                "raw_response": ""
            }
            prediction = ""
            reason = ""
            return output_json, prediction, reason
        json_str = full_text[full_text.find('{'):full_text.rfind('}') + 1]
        if len(json_str)==0:
            json_str = full_text
        
        # remove potential comments in json_str
        try:
            json_str = jsmin.jsmin(json_str)
        except:
            pass

        try:            
            output_json = json.loads(json_str)            
            prediction = output_json.get('prediction')
            if len(prediction)==0:
                prediction = match_prediction(output_json)
            reason = output_json.get('reason')
        #增加答案解析提取,两层保险
        except json.JSONDecodeError:
            # If not JSON, store the raw full_text string in a new dictionary
            prediction = full_text[full_text.find('['):full_text.rfind(']') + 1]
            reason = ""
            if len(prediction) > 0:
                try:
                    prediction = json.loads(prediction)
                    prediction = [int(item) for item in prediction]
                except:
                    prediction = prediction              
            else:
                prediction = match_prediction(full_text)
            output_json = {
                "raw_response": full_text,
                "prediction": prediction,   
                "reason" : ""       
            }
        except Exception as e:
            reason = "Exception:{}".format(e)
            output_json = {
                "raw_response": full_text,
                "prediction": prediction,   
                "reason" : reason
            }

        return output_json, prediction, reason

def prompt_generator_llmzs(v, origin_pred, shot):
    if shot:
        op_part = f"The origin prediction is {origin_pred}.Here is a successful case for optimization:\n" + str(shot) + "\n" + OPTIMIZE_PROMPT
    elif origin_pred:
        op_part = f"The origin prediction is {origin_pred}, which is not good enough." + ONE_SHOT_PROMPT
    else:
        op_part = ""
    prompt = f"""
    		Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.

                Predict <next_place_id> by considering:
                1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
                2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.

                Present your answer in a JSON object with:
                "prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
                
		The data:
                    <historical_stays> [Format: (start_time, day_of_week, POIID)]: {[[item[0],item[1],item[3]] for item in v['historical_stays']]}
                    <context_stays> [Format: (start_time, day_of_week, POIID)]: {[[item[0],item[1],item[3]] for item in v['context_stays']]}
                    <target_stay> [Format: (start_time, day_of_week)]: {[v['target_stay'][0], v['target_stay'][1]]}
        {op_part}
                   """
    # [['hour', 'weekday', 'venue_category_name', 'venue_id']]               
    return prompt

# "LLMMove":"LLM","LLMZS":"LLM","LLMMob"
def prompt_generator(v, prompt_type, rec, origin_pred, shot):
    prompt = ''
    if "LLMZS" in prompt_type:
        prompt = prompt_generator_llmzs(v, origin_pred, shot)
    elif "LLMMob" in prompt_type:
        prompt = prompt_generator_llmmob(v, origin_pred, shot)
    elif "LLMMove" in prompt_type:
        prompt = prompt_generator_llmmove(v, rec, origin_pred, shot)
    return prompt

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371.0
    distance = radius * c
    return distance

def identify_home_and_work_gpt(llm_model, userdata,feature, dialogs=None):
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
        answer6 = llm_model(Q6[0]['content'])
        printQA(Q6, answer6)
        dialogs.append({"role": "user", "content": Q6[0]["content"]})
        dialogs.append({"role": "assistant", "content": answer6})
        home_and_work = get_json(answer6)
        if home_and_work is not None:
            return home_and_work['home'], home_and_work['work'], dialogs
        else:
            print(f"Retrying... ({attempt + 1}/{max_retries})")
            return None, None, dialogs

def get_predicted_state(llm_model, trajectory, home, work,feature,dialogs=None,
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
        answer04 = llm_model(Q04[0]['content'])
        # printQA(Q04, answer04)
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
                            return predicted_intent, dialogs
                    elif length:
                        if len(predicted_intent) == length:
                            print("Data loaded successfully:", predicted_intent)
                            return predicted_intent, dialogs
                    elif not predict_state and not length:
                        print("Warning: The length of the adjusted prediction cannot be checked.")
                        print("Data loaded successfully:", predicted_intent)
                        return predicted_intent, dialogs
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
 
            
def get_home_and_work_feature(llm_model, int_dict, dialogs=None):
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
        answer5 = llm_model(Q5[0]['content'])
        printQA(Q5, answer5)
        dialogs.append({"role": "user", "content": Q5[0]["content"]})
        dialogs.append({"role": "assistant", "content": answer5})
        home_and_work_feature = get_json(answer5)
        if home_and_work_feature is not None:
            return home_and_work_feature, dialogs
        else:
            print(f"Retrying... ({attempt + 1}/{max_retries})")
    return None, dialogs