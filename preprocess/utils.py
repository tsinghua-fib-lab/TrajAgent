from geopy.distance import geodesic
from token_count import TokenCount
import os
import pandas as pd
from tqdm import tqdm
import datetime
import json_repair
import json
import math

import numpy as np
from math import sin, cos, asin, sqrt, radians

from UniEnv.etc.settings import *
from prompt_optimize.utils.prompt import *

map_global = None


def token_count(text):
    tc = TokenCount(model_name="gpt-3.5-turbo")
    return tc.num_tokens_from_string(text)

def get_distance(lon_lat_coord1, lon_lat_coord2):
    lon1, lat1 = lon_lat_coord1
    lon2, lat2 = lon_lat_coord2
    return geodesic((lat1, lon1),(lat2, lon2)).km

def cal_angle(start_point, end_point):
        """
        方位角计算
        """
        return (round(90 - math.degrees(math.atan2(end_point.y - start_point.y, end_point.x - start_point.x)), 2)) % 360

def angle2dir(angle):
    """
    将方位角离散成8个基本方向
    """
    Direction = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']

    s = 22.5
    for i in range(8):
        if angle < s + 45 * i:
            return Direction[i]
    return Direction[0]

def angle2dir_4(angle):
    """
    将方位角离散成4个基本方向
    """
    Direction = ['north', 'east', 'south', 'west']
    
    if angle < 45 or angle >= 315:
        return Direction[0]  
    elif 45 <= angle < 135:
        return Direction[1]  
    elif 135 <= angle < 225:
        return Direction[2]  
    else:  
        return Direction[3]  

def get_timestamp(date, date_format):
    return datetime.datetime.strptime(date,date_format).timestamp()

def get_weekday(date, date_format):
    return datetime.datetime.strptime(date,date_format).weekday()

def get_day(date, date_format):
    """
    输入xx日xx时xx分xx秒，返回日。如输入"2017-07-08 04:40:00"，返回8
    """
    return datetime.datetime.strptime(date,date_format).day

def get_date(date, date_format):
    return datetime.datetime.strptime(date,date_format).date()

def get_hour(date, date_format):
    """
    输入xx日xx时xx分xx秒，返回24h制的0-23。如输入"2017-07-08 04:40:00"，返回4
    """
    return datetime.datetime.strptime(date,date_format).hour

def get_minute(date, date_format):
    return datetime.datetime.strptime(date, date_format).minute

def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance

def get_users(dataset, model, user_num, use_osm):
    """
    返回包含所有用户ID（150个用户） 的list
    """
    if use_osm:
        user_trajs = json.load(open(os.path.join(PROCESSED_PATH, dataset, model, f"{dataset}_user_traj_{user_num}_osm.json"),'r'))
    else:
        user_trajs = json.load(open(os.path.join(PROCESSED_PATH, dataset, model, f"{dataset}_user_traj_{user_num}.json"),'r'))
    all_users = list(user_trajs['context'].keys()) # 讲道理，context,history和test（ground）的用户应该是一样的
    all_users = [int(user) for user in all_users]
    return all_users

def get_results_generate(dataset, model_name, experiment_id, user_info=None, use_osm=False, test_user_num=50):
    """
    {'weekday': 2, 
    'st_time': '2017-08-30 08:49:00', 
    'end_time': '2017-08-30 09:56:00', 
    'poi_name': '百福隆超市(天坛店)', 
    'cat_name': nan, 
    'poi_id': 700347652, 
    'lon': 116.42539714285714, 
    'lat': 39.89760571428571, 
    'norm_in_day': 0.3333333333333333, 
    'cat_1': '购物'}
{
    "weekday": [
        [
            "外交部街33号院10号楼 from 06:30 to 07:30",...
        ],
        "The plan adheres to..."
    ],
    "weekend": [
        [
            "外交部街33号院10号楼 from 09:00 to 10:00",...
        ],
        "The plan meets the required format and ..."
    ]
}
    """
    # user_result = {}
    invalid_users = []
    result_path = os.path.join(GENERALIZED_PATH, model_name, dataset)
    if use_osm:
        true_file = os.path.join(PROCESSED_PATH, dataset, "agentsim", f"{dataset}_user_traj_{test_user_num}_osm.json")
    else:
        true_file = os.path.join(PROCESSED_PATH, dataset, "agentsim", f"{dataset}_user_traj_{test_user_num}.json")
    with open(true_file, 'r') as f:
        true_data = json.load(f)
    all_files = os.listdir(result_path)
    for user_file in all_files:
        user_result_path = os.path.join(result_path, user_file)
        for user_result_file in os.listdir(user_result_path):
            exp_id = user_result_file.strip(".json").split("_")[-1]
            user_id = str(user_file)
            if exp_id != str(experiment_id):
                continue
            candidate_traj_info = json.load(open(os.path.join(user_result_path , user_result_file), 'r'))
            result_list = []
            candidate_poi_info = {'generate': []}
            try:
                weekday_traj = candidate_traj_info['weekday'][0]
                weekend_traj = candidate_traj_info['weekend'][0]
                candidate_poi_info['generate'].append(weekday_traj)
                candidate_poi_info['generate'].append(weekend_traj)
                candidate_poi_info['true'] = true_data['ground'][user_id]
            except:
                invalid_users.append(user_id)
            # user_result.setdefault(user_id,candidate_poi_info)
            if user_info is not None:
                user_info[int(user_id)].update(candidate_poi_info)
    return user_info, invalid_users

def transform_dict_key(tar_dict, key):
    if isinstance(next(iter(tar_dict)), int):
        return int(key)
    else:
        return str(key)
    
def get_results_train(dataset, model_name, experiment_id, user_info=None):
    """
    更新user_info的generate, true部分
    {

    user_id:{
        "true":[{traj_point},{}],
        "generate"[[
        '花乡万芳园(二区) from 0:00 to 2:00'
        ...
        '花乡万芳园(二区) from 21:00 to 23:00'
        ],[...],...]
    },...
    }
    """
    # user_result = {}
    invalid_users = []
    result_path = os.path.join(RESULT_PATH, "agentsim", dataset, model_name)
    all_files = os.listdir(result_path)
    for user_file in all_files:
        if ".json" not in user_file:
            continue
        exp_id = user_file.strip(".json").split("_")[-1]
        user_id = int(user_file.strip(".json").split("_")[-2])
        if exp_id != str(experiment_id):
            continue
        if transform_dict_key(user_info, user_id) not in user_info:
            print(f"Abnormal user {user_id}!!!!")
            continue
        candidate_poi_info = json.load(open(os.path.join(result_path , user_file), 'r'))
        result_list = []
        try:
            for item in candidate_poi_info['generate'][0]:
                if isinstance(item, dict):  # 对于两种形式的答案提取
                    try:
                        reform = f"{item['POI_name']} from {item['start_time']} to {item['end_time']}"
                    except:
                        reform = f"{item['POI_name']} from {item['from']} to {item['to']}"
                else:
                    reform = item
                result_list.append(reform)
            candidate_poi_info['generate'] = [result_list,candidate_poi_info['generate'][1]]
            for idx, item in enumerate(candidate_poi_info['true']):
                candidate_poi_info['true'][idx]['lon'] = float(item['lon'])
                candidate_poi_info['true'][idx]['lat'] = float(item['lat'])    
        except:
            invalid_users.append(user_id)
        # user_result.setdefault(user_id,candidate_poi_info)
        if user_info is not None:
            user_info[user_id].update(candidate_poi_info)
    return user_info, invalid_users
        

def process_output(output, keys):
    fin_result = [output, None]
    results = []
    try:
        # 使用json_repair来更加稳定的解析地址
        res_dict = json_repair.repair_json(output)
        res_dict = json.loads(res_dict)
        fin_result = [res_dict, None]
        if isinstance(res_dict, dict):
            for key in keys:
                if key in res_dict:
                    results.append(res_dict[key])
                else:
                    results.append(None)  # 如果没有找到该key，返回None
        elif isinstance(res_dict, list):
            for key in keys:
                if key in res_dict[0]:
                    results.append(res_dict[0][key])
                else:
                    results.append(None)  # 如果没有找到该key，返回None
        else:
            for key in keys:
                results.append(None)  # 如果没有找到该key，返回None
        fin_result = results
        # 返回结果和 None 错误
        return (fin_result, None)  # 返回结果列表和 None 错误
    except json.JSONDecodeError as e:
        return (fin_result, f"JSONDecodeError for {output}: {e}")
    except Exception as e:
        return (fin_result, f"Error processing: {e}")

def distance(gps1,gps2):
    x1,y1 = gps1
    x2,y2 = gps2
    return np.sqrt((x1-x2)**2+(y1-y2)**2 )
      
def gen_matrix(data='geolife', user_num=150):
    train_data = pd.read_csv(RAW_PATH, data, f"intent_label_{user_num}_encode_address_full.csv")
    coord_map = {}
    for row in train_data.itertuples():
        poi_id = int(row.POI_encoded)
        lon = row.longitude
        lat = row.latitude
        coord_map.setdefault(poi_id, [lon, lat])
    lons = []
    lats = []
    max_locs = len(set(train_data['POI_encoded']))
    for i in range(max_locs):
        lons.append(coord_map[i][0])
        lats.append(coord_map[i][1])

    # reg1 = np.zeros([max_locs,max_locs])
    # for i in range(len(train_data)):
    #     line = train_data[i]
    #     for j in range(len(line)-1):
    #         reg1[line[j],line[j+1]] +=1
    reg2 = np.zeros([max_locs,max_locs])
    for i in range(max_locs):
        for j in range(max_locs):
            if i!=j:
                reg2[i,j] = distance((lons[i],lats[i]),(lons[j],lats[j]))
    
    # np.save(os.path.join(PROCESSED_PATH, data, 'agentsim', 'M1.npy'), reg1) # 转移矩阵
    # np.save(os.path.join(PROCESSED_PATH, data, 'agentsim', 'M2.npy'), reg2) #距离矩阵
    print('Matrix Generation Finished')
    return reg2

def transform_item_list(item, tar_list):
    if isinstance(tar_list[0], int):
        item = int(item)
    elif isinstance(tar_list[0], str):
        item = str(item)
    return item

def get_intent_from_cat(cat, poi, home, work, use_en):
    """
    不使用LLM，通过已有intent_label_150文件得出的规律，从一级分类推测动作或意图
    """
    if poi == home:
        if use_en:
            return 'At Home'
        else:
            return '在家'
    elif poi == work:
        if use_en:
            return 'Job/Work'
        else:
            return '工作'
    else:
        for intent, cat_list in INTENT_CAT.items():
            if cat in cat_list:
                if use_en:
                    return INTENT_MAP[intent]
                else:
                    return intent
    return None

def match_home_work(home_work_info, user_traj_info, poi_info):
    user_home_work = {}
    for user_traj in user_traj_info:
        user_id = user_traj['user_id']
        if transform_dict_key(home_work_info, user_id) in home_work_info:
            user_id = transform_dict_key(home_work_info, user_id)
            home = home_work_info[user_id]['home'] # 经纬度
            work = home_work_info[user_id]['work'] # 经纬度
            if home not in user_traj['user_loc'] or work not in user_traj['user_loc']:
                continue
            home_poi = user_traj['user_poi'][user_traj['user_loc'].index(home)]
            if "_" in home_poi:
                home_poi = home_poi.split("_")[1]
            work_poi = user_traj['user_poi'][user_traj['user_loc'].index(work)]
            if "_" in work_poi:
                work_poi = work_poi.split("_")[1]
            if home_poi not in poi_info or work_poi not in poi_info:
                continue
            user_home_work.setdefault(user_id, {'home': poi_info[home_poi]['name'], 'work': poi_info[work_poi]['name']})
    return user_home_work
            
            
def gen_home_work_test(dataset, llm, model_name, user_num, user_home_work={}):
    """
    获取数据集每个用户的home_work。训练数据user_num=1500, 测试数据user_num=150
    根据标注意图的时间分布分辨home/work地点.意图为'At Home', 'Sleeping'的为home, 意图为'Work', 'Job/Work'的为work
    """
    dialogs = []

    def calculate_home_and_work(df):
        home_df = df[df['intent_en'].isin(['At Home', 'Sleeping'])]
        work_df = df[df['intent_en'].isin(['Work', 'Job/Work'])]
        runningerrands_df = df[df['intent_en'].isin(['Short Trip', 'Sports', 'Medical Services', 'Handling Affairs'])]
        
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

    def get_home_and_work_feature(int_dict, llm):
        # Use GPT to extract the characteristics of home and workplace from statistical data
        max_retries = 20
        for attempt in range(max_retries):

            answer5 = llm.get_response(prompt_text=GET_HOMEWORK_FEATURE.format(int_dict))
            dialogs.append({"role": "user", "content": GET_HOMEWORK_FEATURE.format(int_dict)})
            dialogs.append({"role": "assistant", "content": answer5})
            home_and_work_feature = get_json(answer5)
            if home_and_work_feature is not None:
                return home_and_work_feature
            else:
                print(f"Retrying... ({attempt + 1}/{max_retries})")
        return None

    def identify_home_and_work_gpt(userdata, feature, llm):
        # Use GPT to identify home and work locations
        max_retries = 20
        for attempt in range(max_retries):
            answer6 = llm.get_response(prompt_text=IDENTIFY_HOME_WORK.format(userdata, feature))
            dialogs.append({"role": "user", "content": IDENTIFY_HOME_WORK.format(userdata, feature)})
            dialogs.append({"role": "assistant", "content": answer6})
            home_and_work = get_json(answer6)
            if home_and_work is not None:
                return home_and_work['home'], home_and_work['work']
            else:
                print(f"Retrying... ({attempt + 1}/{max_retries})")
        return None, None
    
    # 需要标注home work的数据   
    data4annotate = os.path.join(RAW_PATH, dataset, f"intent_label_{user_num}_osm.csv")
    # 标注的home work 存储位置
    hw_path = os.path.join(RAW_PATH, dataset, model_name)
    if not os.path.exists(hw_path):
        os.makedirs(hw_path, exist_ok=True) 
    # 训练数据和测试数据都需要标注
    # 对于LIMP_BJ/intent_label_150.csv，只要使用标注好的意图获得Homework_150.csv即可。对于其他更大规模的数据，需要先根据LIMP_BJ/intent_label_150.csv的意图时间分布判断出每个用户的home和work，再通过一级类别的映射得到其他意图
    hw_file = os.path.join(hw_path, f'homeandwork_{int(user_num)}.csv') 
    # 手工标注的意图数据，作为计算home_work分布的依据
    intent_label = os.path.join(RAW_PATH, "LIMP_BJ", 'intent_label_target.csv')

    feature = get_home_and_work_feature(calculate_home_and_work(pd.read_csv(intent_label)), llm)
    user_segments = split_user_data_by_date(pd.read_csv(data4annotate))

    for user_id, user_data in tqdm(user_segments.items(), desc="Processing Users"):
        user_id = transform_dict_key(user_home_work, user_id)
        if user_id in user_home_work:
            homeandwork = pd.DataFrame({'user_id': [user_id], 'home': [user_home_work[user_id]['home']], 'work': [user_home_work[user_id]['work']]})
        else:
            # Use GPT to identify home and work locations
            home, work = identify_home_and_work_gpt(user_data['feature'], feature, llm)
            print('Home:', home)
            print('Work:', work)
            homeandwork = pd.DataFrame({'user_id': [user_id], 'home': [home], 'work': [work]})
        # 将生成的home_work逐个用户写入csv
        if os.path.exists(hw_file):
            homeandwork.to_csv(hw_file, index=False, encoding='utf-8', mode='a', header=False)
        else:
            homeandwork.to_csv(hw_file, index=False, encoding='utf-8', mode='a')

def get_home_work_osm(dataset, user_num, citygpt_address):
    """
    根据intent, 识别intent_label_osm_map.csv(已经有意图分布)中每个user的home, work.
    默认home或work为None的情况在生成home/work的时候已经处理了
    """
    results = {}
    # 按user_id分组
    df = pd.read_csv(os.path.join(RAW_PATH, dataset, f"intent_label_{user_num}_encode_address_full_osm.csv"))
    grouped = df.groupby('user_id')
    
    for user_id, group in grouped:
        user_data = {'home': [
            "nan(None)",
            None
        ], 'work': [
            "nan(None)",
            None
        ]}
        
        # 检查每条记录
        for _, row in group.iterrows():
            intent = row['intent_en']
            name = row['name']
            poi_id = int(row['id_osm'])
            poi_encoded = row['POI_encoded']
            if citygpt_address:
                address = row['address_citygpt']
            else:
                address = f"(street:{row['street']}, subdistrict:{row['subdistrict']}, admin:{row['admin']})"
            # 检查是否是家庭地点
            # ({'admin':
            if intent in ['At Home', 'Sleeping']:
                user_data['home'] = [f"{name}(address:{address})", {'poi_id': poi_id, 'poi_encoded': poi_encoded}]
                
            # 检查是否是工作地点
            elif intent in ['Work', 'Job/Work']:
                user_data['work'] = [f"{name}(address:{address})", {'poi_id': poi_id, 'poi_encoded': poi_encoded}]
        
        # 如果 home 或 work 为 None，使用最频繁的地点填充
        if user_data['home'][1] is None or user_data['work'][1] is None:
            # 统计每个 POI 的访问频率
            poi_counts = group['POI_encoded'].value_counts()
            most_frequent_pois = poi_counts.index.tolist()
            
            # 填充 home 和 work
            if user_data['home'][1] is None and len(most_frequent_pois) > 0:
                most_frequent_home = most_frequent_pois[0]
                home_row = group[group['POI_encoded'] == most_frequent_home].iloc[0]
                home_name = home_row['name']
                if citygpt_address:
                    home_address = home_row['address_citygpt']
                else:
                    home_address = f"(street:{home_row['street']}, subdistrict:{home_row['subdistrict']}, admin:{home_row['admin']})"
                user_data['home'] = [
                    f"{home_name}(address:{home_address})",
                    {'poi_id': int(home_row['id_osm']), 'poi_encoded': most_frequent_home}
                ]
            
            if user_data['work'][1] is None and len(most_frequent_pois) > 1:
                most_frequent_work = most_frequent_pois[1]
                work_row = group[group['POI_encoded'] == most_frequent_work].iloc[0]
                work_name = work_row['name']
                if citygpt_address:
                    work_address =  work_row['address_citygpt']
                else:
                    work_address = f"(street:{work_row['street']}, subdistrict:{work_row['subdistrict']}, admin:{work_row['admin']})"
                user_data['work'] = [
                    f"{work_name}(address:{work_address})",
                    {'poi_id': int(work_row['id_osm']), 'poi_encoded': most_frequent_work}
                ]
        results.setdefault(user_id, user_data)
    return results

# def get_home_work(dataset, model_name, user_num):
#     """
#     homeandwork_{user_num}.csv -> {user_id: {'home': [home(address), poi_info], 'work': [work(address), poi_info]},...}
#     """
#     # 被标注home,work的所有数据
#     all_data = pd.read_csv(os.path.join(RAW_PATH, dataset, f"intent_label_{int(user_num)}_encode_address_full_osm.csv"))
#     # all_data = pd.read_csv(os.path.join(RAW_PATH, dataset, f"intent_label_{int(user_num)}_osm.csv"))
#     POI_address = {}
#     POI_info = {}
#     for row in all_data.itertuples():
#         POI_address.setdefault(row.POI_name, row.address_citygpt)
#         POI_info.setdefault(row.POI_name, {"poi_id": row.POI_ID, "poi_encoded": row.POI_encoded})
#         # POI_info.setdefault(row.POI_name, {"poi_id": row.POI_ID})
#     all_users = set(all_data['user_id'])
#     # 可以选择使用qwen2-7b/qwen2-72b标注的home_work数据
#     home_work_data = pd.read_csv(os.path.join(RAW_PATH, dataset, model_name, f"homeandwork_{int(user_num)}.csv"))
#     home_work_result = {user_id: {'home': None, 'work': None} for user_id in all_users}
#     # admin,subdistrict,poi,street
#     for row in home_work_data.itertuples():
#         user_id = int(row.user_id)
#         home = row.home
#         work = row.work
#         home_work_result[user_id] = {'home': [f"{home}({POI_address.get(home,None)})",POI_info.get(home,None)] , 'work': [f"{work}({POI_address.get(work, None)})",POI_info.get(work,None)]}
#         # home_work_result[user_id] = {'home': home, 'work': work}
#     return home_work_result

def get_home_work(dataset, model_name, user_num):
    """
    homeandwork_{user_num}.csv -> {user_id: {'home': home, 'work': work},...}
    """
    # 被标注home,work的所有数据
    all_data = pd.read_csv(os.path.join(RAW_PATH, dataset, f"intent_label_{int(user_num)}_encode_address_full_osm.csv"))

    all_users = set(all_data['user_id'])
    # 可以选择使用qwen2-7b/qwen2-72b标注的home_work数据
    home_work_data = pd.read_csv(os.path.join(RAW_PATH, dataset, model_name, f"homeandwork_{int(user_num)}.csv"))
    home_work_result = {user_id: {'home': None, 'work': None} for user_id in all_users}
    # admin,subdistrict,poi,street
    for row in home_work_data.itertuples():
        user_id = int(row.user_id)
        home = row.home
        work = row.work
        home_work_result[user_id] = {'home': home, 'work': work}
    return home_work_result

def get_home_work_eval(dataset, model_name, user_num, use_osm, citygpt_address):
    """
    homeandwork_{user_num}.csv -> {user_id: {'home': home, 'work': work},...}
    """
    # 被标注home,work的所有数据
    # all_data = pd.read_csv(os.path.join(RAW_PATH, dataset, f"intent_label_{int(user_num)}_encode_address_full.csv"))
    all_users = get_users(dataset, "agentsim" , user_num, use_osm)
    if use_osm:
        # f'LIMP_BJ_user_home_work_{user_num}_osm.json')
        with open(os.path.join(PROCESSED_PATH, dataset, 'agentsim', f"{dataset}_user_home_work_{int(user_num)}_osm_{citygpt_address}.json"), 'r') as f:
            home_work_data = json.load(f)
        home_work_result = {}
        for user_id in all_users:
            user_id = int(user_id)
            home = home_work_data[str(user_id)]['home'][0].split("(address:")[0]
            work = home_work_data[str(user_id)]['work'][0].split("(address:")[0]
            home_work_result.setdefault(user_id, {'home': home, 'work': work})
    else:
        home_work_data = pd.read_csv(os.path.join(RAW_PATH, dataset, model_name, f"homeandwork_{int(user_num)}.csv"))
        home_work_result = {user_id: {'home': None, 'work': None} for user_id in all_users}
        # admin,subdistrict,poi,street
        for row in home_work_data.itertuples():
            user_id = int(row.user_id)
            home = row.home
            work = row.work
            home_work_result[user_id] = {'home': home , 'work': work}
    return home_work_result

def sep_day_record(df):
    """
    将横跨两天的记录拆成两条
    """
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    split_records = []
    for _, row in df.iterrows():
        # 如果 start_time 和 end_time 不同日期
        if row['start_time'].date() != row['end_time'].date():
            # 第一条记录（从 start_time 到当天的 23:59:59）
            record_1 = row.copy()
            record_1['end_time'] = row['start_time'].replace(hour=23, minute=59, second=59)
            split_records.append(record_1)
            
            # 第二条记录（从第二天的 00:00:00 到 end_time）
            record_2 = row.copy()
            record_2['start_time'] = row['end_time'].replace(hour=0, minute=0, second=0)
            split_records.append(record_2)
        else:
            # 如果在同一天，保留原记录
            split_records.append(row)
    df_split = pd.DataFrame(split_records)
    df_split['start_time'] = df_split['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S') # 2019-10-16 20:04:21
    df_split['end_time'] = df_split['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df_split

def get_cat_map(poi_cat_id, is_en):
    """
    category_id(Tencent)-> 一级类名
    """
    poi_cat_id = str(poi_cat_id)
    poi_cat_head = poi_cat_id[:2]
    if is_en:
        return MAP_CATEGORY_NUM_EN.get(poi_cat_head, None)  # 提供默认值以避免 KeyError
    else:
        return MAP_CATEGORY_NUM.get(poi_cat_head, None)  # 提供默认值以避免 KeyError

def save_df_to_json_lines(df, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            json_str = json.dumps(row_dict)
            f.write(json_str + '\n')