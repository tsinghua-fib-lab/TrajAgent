from datetime import datetime, timedelta
import hashlib
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import copy
import numpy as np
import os
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
from UniEnv.etc.settings import OFFSET_DICT, MODEL_TYPE

def parse_time(time_in, timezone_offset_in_minute=0):
    """
    将 json 中 time_format 格式的 time 转化为 local datatime
    """
    date = datetime.strptime(time_in, '%a %b %d %H:%M:%S %z %Y')  # 这是 UTC 时间
    return date + timedelta(minutes=timezone_offset_in_minute)

def time_to_hours(use_day, iso_time, use_weekday=False):
    # 将ISO格式的时间字符串转换为datetime对象
    dt = datetime.strptime(iso_time, '%Y-%m-%dT%H:%M:%SZ')
    # 计算自Unix纪元以来的总秒数
    total_seconds = (dt - datetime(2008, 4, 1)).total_seconds()
    weekday = dt.weekday()
    hr_time = dt.hour
    monday_zero_result = weekday*24 + hr_time
    # 将秒数转换为小时数
    if use_weekday:
        result = monday_zero_result  # 从周一开始的小时数
    else:
        if use_day:
            result = (total_seconds / 3600)%24
        else:
            result = total_seconds / 3600
    return result

def get_traj_time(use_day, iso_time, model):
    if model in ["type4aug",'LLM']:
        res_time = int(time_to_hours(False, iso_time,False)) #从2008-04-01开始的小时数
    elif model in ['GETNext','LibCity']:
        res_time = iso_time
    elif model == 'ActSTD':
        res_time = int(time_to_hours(False, iso_time,True))  #0-24h
    else:
        res_time = int(time_to_hours(use_day, iso_time, False)) #从2008-04-01开始的天数
    return res_time

def hours_to_time(hours, **kwargs):
    # 将小时数转换为timedelta对象
    td = timedelta(hours=hours)
    # 计算对应的datetime对象
    # TODO: 这里的起始时间应该设定为数据集的最早起始时间，与数据集有关。需要写个函数提取一下
    dt = datetime(2008, 4, 1) + td
    # 格式化为ISO格式的时间字符串
    std_time = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    if 'weekday' in kwargs:
        return dt.weekday()
    elif 'hr_24h' in kwargs:
        return dt.hour
    elif 'min_24h' in kwargs:
        return dt.minute
    else:
        return std_time


def cal_timeoff(now_time, base_time):
    """
    计算两个时间之间的差值，返回值以小时为单位
    """
    delta = now_time - base_time
    return delta.days * 24 + delta.seconds / 3600
        

def convert_time(dataset, model, original_time_str):
    # 解析原始时间字符串的格式
    if dataset in['Shanghai_ISP','Shanghai_Weibo']: 
        parsed_time = datetime.strptime(original_time_str, "%a %b %d %H:%M:%S %Y")
    else:
        parsed_time = datetime.strptime(original_time_str, "%Y-%m-%dT%H:%M:%SZ")
    # 转换为目标格式的字符串
    if model == "GETNext":
        formatted_time_str = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
    elif model == "SNPM":
        formatted_time_str = parsed_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif model == "STHM":
        formatted_time_str = parsed_time.strftime("%Y-%m-%dT%H:%M:%S")
    return formatted_time_str

def delta_minutes(ori, cp):
    ori = datetime.strptime(ori, "%Y-%m-%dT%H:%M:%SZ")
    cp = datetime.strptime(cp, "%Y-%m-%dT%H:%M:%SZ")
    delta = (ori.timestamp() - cp.timestamp())/60
    if delta < 0:
        delta = 1
    return delta

def get_delta(arrival_times):
    copy_times = copy.deepcopy(arrival_times)
    copy_times.insert(0, copy_times[0]) 
    copy_times.pop(-1)
    return list(map(delta_minutes, arrival_times, copy_times))

def get_relativeTime(arrival_times): 
    first_time_list = [arrival_times[0] for _ in range(len(arrival_times))]
    return list(map(delta_minutes, arrival_times, first_time_list))

def distance(lon1, lat1, lon2, lat2):  
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
   
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    if lon1 == 0 or lat1 ==0 or lon2==0 or lat2==0:
        return 0
    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  
    return c * r  

def construct_spatial_matrix_accordingDistance(distance_theta, venue_cnt, venue_info, gaussian_beta=None):
    SS_distance = np.zeros((venue_cnt+1, venue_cnt+1))  
    SS_gaussian_distance = np.zeros((venue_cnt+1, venue_cnt+1))  
    SS_proximity = np.zeros((venue_cnt+1, venue_cnt+1))  
    for i in range(1, venue_cnt+1):
        for j in range(1, venue_cnt+1):
            distance_score = distance(venue_info[i]['lon'], venue_info[i]['lat'], venue_info[j]['lon'], venue_info[j]['lat'])
            SS_distance[i, j] = distance_score  
            if gaussian_beta is not None:
                distance_gaussian_score = np.exp(-gaussian_beta * distance_score) 
                SS_gaussian_distance[i, j] = distance_gaussian_score  
            if SS_distance[i, j] < distance_theta:  
                SS_proximity[i, j] = 1
    return SS_distance, SS_proximity, SS_gaussian_distance

def get_user_lidFreq(result_loc, loc_num):
    result_freq = np.zeros((len(result_loc.keys())+1,loc_num+1))  #(user_num, loc_num)
    for user, sessions in result_loc.items():
        freq_list = [0]*(loc_num+1)
        visit_num = sum(len(ses) for ses in sessions)
        for session in sessions:
            for poi in session:
                freq_list[poi] += 1
        result_freq[user] = [item/visit_num for item in freq_list]
    return result_freq

def get_loc_feature(map_loc_dict, loc_num):
    all_cats = set()
    result_lon_ori = [0]*(loc_num+1)
    result_lat_ori = [0]*(loc_num+1)
    result_feature_lon = [0]*(loc_num+1)
    result_feature_lat = [0]*(loc_num+1)
    result_feature_cat = [0]*(loc_num+1)
    for loc, info in map_loc_dict.items():
        result_lon_ori[loc] = info['lngidx']
        result_lat_ori[loc] = info['latidx']
        result_feature_lon[loc] = info['lon']
        result_feature_lat[loc] = info['lat']
        result_feature_cat[loc] = info['cat']
        all_cats.add(info['cat'])
    cat_num = len(all_cats)
    return result_lon_ori,result_lat_ori, result_feature_lon, result_feature_lat, result_feature_cat, cat_num
    

def string_to_md5_hex(s):
    # 创建MD5哈希对象
    hash_object = hashlib.md5()
    # 更新哈希对象，输入需要是bytes类型
    hash_object.update(s.encode('utf-8'))
    # 获取十六进制形式的摘要
    hex_dig = hash_object.hexdigest()
    return hex_dig

def convert_timestamp(dataset, time_str):
    if dataset in['Shanghai_ISP','Shanghai_Weibo']: 
        timestamp = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
    else:
        timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    midnight = timestamp.replace(hour=0, minute=0, second=0)
    total_minutes = (timestamp - midnight).total_seconds() / 60
    total_minutes_in_day = 24 * 60

    fraction = total_minutes / total_minutes_in_day

    return fraction

def id_encode(fit_df: pd.DataFrame, encode_df: pd.DataFrame, column: str, padding: int = -1) -> Tuple[dict, int]:
    id_le = LabelEncoder()
    id_le = id_le.fit(fit_df[column].values.tolist())
    if padding == 0:
        padding_id = padding
        encode_df[column] = [
            id_le.transform([i])[0] + 1 if i in id_le.classes_ else padding_id
            for i in encode_df[column].values.tolist()
        ]
    else:
        padding_id = len(id_le.classes_)
        encode_df[column] = [
            id_le.transform([i])[0] if i in id_le.classes_ else padding_id
            for i in encode_df[column].values.tolist()
        ]
    return id_le, padding_id

def ignore_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ignore the first check-in sample of every trajectory because of no historical check-in.

    """
    df['pseudo_session_trajectory_rank'] = df.groupby(
        'pseudo_session_trajectory_id')['UTCTimeOffset'].rank(method='first')
    df['query_pseudo_session_trajectory_id'] = df['pseudo_session_trajectory_id'].shift()
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'query_pseudo_session_trajectory_id'] = None
    df['last_checkin_epoch_time'] = df['UTCTimeOffsetEpoch'].shift()
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'last_checkin_epoch_time'] = None
    df.loc[df['UserRank'] == 1, 'SplitTag'] = 'ignore'
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'SplitTag'] = 'ignore'
    return df

def encode_poi_catid(
        fit_df: pd.DataFrame,
        encode_df: pd.DataFrame,
        source_column: str,
        target_column: str,
        padding: int = -1
) -> Tuple[LabelEncoder, int]:
    """
    将source_column列中的唯一值编码到target_column列，类似于STPM的id_encode函数。

    :param fit_df: 用于构建LabelEncoder的DataFrame
    :param encode_df: 需要编码的DataFrame
    :param source_column: 要编码的源列
    :param target_column: 编码后的目标列
    :param padding: 当值不存在于LabelEncoder中时的填充值
    :return: LabelEncoder实例和填充值padding_id
    """
    # 初始化LabelEncoder并进行fit
    id_le = LabelEncoder()
    id_le = id_le.fit(fit_df[source_column].values.tolist())
    
    # 如果padding为0，编码值从1开始
    if padding == 0:
        padding_id = padding
        encode_df[target_column] = [
            id_le.transform([i])[0] + 1 if i in id_le.classes_ else padding_id
            for i in encode_df[source_column].values.tolist()
        ]
    else:
        # 如果padding不是0，默认填充值为最大编码值+1
        padding_id = len(id_le.classes_)
        encode_df[target_column] = [
            id_le.transform([i])[0] if i in id_le.classes_ else padding_id
            for i in encode_df[source_column].values.tolist()
        ]
    
    return id_le, padding_id

def getnext_format(city, model, df):
    if city in OFFSET_DICT:
        offset = OFFSET_DICT[city]
    else:
        offset = 0
    df['TimezoneOffset'] = offset
    df['UTC_time'] = df['UTC_time'].apply(lambda x: convert_time(city, MODEL_TYPE[model], x))
    df['timezone'] = df.apply(
lambda x: datetime.strptime(x['UTC_time'], "%Y-%m-%d %H:%M:%S") + timedelta(hours=offset / 60),
axis=1)
    df['day_of_week'] = df['UTC_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday())
    df['POI_catid_code'] = df['POI_catname'].apply(lambda x: string_to_md5_hex(x))
    df['norm_in_day_time'] = df['UTC_time'].apply(lambda x: convert_timestamp(city, x))  
    df['UserRank'] = df.groupby('user_id')['timezone'].rank(method='first')
    df['check_ins_id'] = df['timezone'].rank(ascending=True, method='first') - 1 
    df['trajectory_id'] = df.apply(lambda x: f"{x['user_id']}_{x['trajectory_id_raw']}", axis=1) 
    df = df.sort_values(by=['user_id', 'timezone'], ascending=True)

    return df

# do label encoding using train dataset
def getnext_encode(df):
    df_train = df[df['SplitTag'] == 'train']
    # padding id use 0
    poi_id_le, padding_poi_ie = id_encode(df_train, df, 'POI_id', padding=-1)
    poi_category_le, padding_poi_category = id_encode(df_train, df, 'POI_catid', padding=-1)
    user_id_le, padding_user_id = id_encode(df_train, df, 'user_id', padding=-1)
    # Re-encode trajectory IDs and check-in IDs to ensure continuity
    # df['timezone'] = df.apply(lambda x: datetime.strptime(x['timezone'], "%Y-%m-%d %H:%M:%S"),axis=1)
    df['check_ins_id'] = df['timezone'].rank(ascending=True, method='first') - 1 
    traj_id_map = {id: idx for idx, id in enumerate(sorted(df['trajectory_id_raw'].unique()))}
    df['trajectory_id'] = df.apply(lambda x: f"{x['user_id']}_{traj_id_map[x['trajectory_id_raw']]}", axis=1) 
    return df

def remap_trajectory_id(df, existing_trajectory_ids):
    new_trajectory_id = max(existing_trajectory_ids) + 1  # 从现有最大 ID 开始
    for idx, row in df.iterrows():
        if row['trajectory_id_raw'] in existing_trajectory_ids:
            df.at[idx, 'trajectory_id_raw'] = new_trajectory_id
            existing_trajectory_ids.add(new_trajectory_id)
            new_trajectory_id += 1
    return df

def reform_test_val(df, start_id):
    # 创建一个字典来记录旧编号到新编号的映射
    df['timezone'] = pd.to_datetime(df['timezone'])
    df = df.sort_values(by=['user_id', 'timezone'], ascending=True)
    old_to_new_id_map = {}
    current_id = start_id

    for idx, row in df.iterrows():
        old_id = row['trajectory_id_raw']
        if old_id not in old_to_new_id_map:
            old_to_new_id_map[old_id] = current_id
            current_id += 1
        df.at[idx, 'trajectory_id_raw'] = old_to_new_id_map[old_id]
    return df

def reform_test_val_cacsr(args, map_continous, part):
    # dict_keys(['testX_target_lengths', 'testX_arrival_times', 'testX_duration2first', 'testX_session_arrival_times', 'testX_local_weekdays', 'testX_session_local_weekdays', 'testX_local_hours', 'testX_session_local_hours', 'testX_local_mins', 'testX_session_local_mins', 'testX_delta_times', 'testX_session_delta_times', 'testX_locations', 'testX_session_locations', 'testX_last_distances', 'testX_users', 'testX_lengths', 'testX_session_lengths', 'testX_session_num', 'testY_arrival_times', 'testY_delta_times', 'testY_locations', 'us', 'vs', 'feature_category', 'feature_lat', 'feature_lng', 'feature_lat_ori', 'feature_lng_ori', 'latN', 'lngN', 'category_cnt', 'user_cnt', 'venue_cnt', 'SS_distance', 'SS_guassian_distance'])
    mark = []
    mark_item = list
    loader = dict(np.load(os.path.join(args.aug_data_path, f'{args.aug_name}_{part}.npz'), allow_pickle=True))
    loader[f'{part}X_session_locations'] = np.array([[[map_continous.get(item,1) for item in ses] for ses in sublist]for sublist in loader[f'{part}X_session_locations']], dtype=object)
    loader[f'{part}Y_locations'] = np.array([[map_continous.get(item,1) for item in sublist] for sublist in loader[f'{part}Y_locations']], dtype=object)
    for i, x in enumerate(loader[f'{part}X_locations']):
        for j, y in enumerate(loader[f'{part}X_locations'][i]):
            if y not in map_continous:
                mark.append((i,j))
            if map_continous[y]==1:
                mark_item == loader[f'{part}X_last_distances'][i][j]
    for i, x in enumerate(loader[f'{part}X_last_distances']):
        for j, y in enumerate(loader[f'{part}X_last_distances'][i]):
            if (i,j) in mark:
                loader[f'{part}X_last_distances'][i][j] = mark_item          
    loader[f'{part}X_locations'] = np.array([[map_continous.get(item,1) for item in sublist] for sublist in loader[f'{part}X_locations']], dtype=object)
    if part == 'test':
        loader['us'] =  np.array([[map_continous.get(item,1) for item in sublist] for sublist in loader['us']], dtype=object)
        loader['vs'] =  np.array([[map_continous.get(item,1) for item in sublist] for sublist in loader['vs']], dtype=object)
    return loader
    

def cacsr_format_session(SS_gaussian_distance, all_session_based_loc_his, session_based_loc_hc, session_based_hour_hc, history_delta_hour, cacsr_result, uid, part):
    # session_based_loc_hc 其实是history+(context+target)
    session_loc = session_based_loc_hc[-1]  # 当前session
    n = len(session_loc)
    session_tim = [hours_to_time(item) for item in session_based_hour_hc[-1]]
    session_delta = get_delta(session_tim)
    session_based_history_tim = [list(map(hours_to_time, sublist)) for sublist in session_based_hour_hc][:-1]
    session_based_history_hour = [list(map(lambda hour: hours_to_time(hour, hr_24h=True), sublist)) for sublist in session_based_hour_hc][:-1]
    session_based_history_day = [list(map(lambda hour: hours_to_time(hour, weekday=True), sublist)) for sublist in session_based_hour_hc][:-1]
    session_based_history_minute = [list(map(lambda hour: hours_to_time(hour, min_24h=True), sublist)) for sublist in session_based_hour_hc][:-1]
    session_based_history_loc = session_based_loc_hc[:-1]
    session_based_history_delta_tim = [get_delta(sublist) for sublist in session_based_history_tim]
    session_num = len(session_based_loc_hc)
    session_based_lengths = [len(session) for session in session_based_loc_hc]
    
    history_tim = [item for sublist in session_based_history_tim for item in sublist] # flat
    history_delta_tim = history_delta_hour
    history_hour = [hours_to_time(item, hr_24h=True) for sublist in session_based_hour_hc[:-1] for item in sublist]
    history_day = [hours_to_time(item, weekday=True) for sublist in session_based_hour_hc[:-1] for item in sublist]
    history_minute = [hours_to_time(item, min_24h=True) for sublist in session_based_hour_hc[:-1] for item in sublist]
    history_loc = [item for sublist in session_based_history_loc for item in sublist]
    
    cur_tim = session_tim[:-1]
    cur_delta_tim = session_delta[:-1]
    cur_hour = [hours_to_time(item, hr_24h=True) for item in session_based_hour_hc[-1][:-1]]
    cur_day = [hours_to_time(item, weekday=True) for item in session_based_hour_hc[-1][:-1]]
    cur_minute = [hours_to_time(item, min_24h=True) for item in session_based_hour_hc[-1][:-1]]
    target_tim = session_tim[1:]
    target_delta_tim = session_delta[1:]
    cur_loc = session_loc[:-1]
    target_loc = session_loc[1:]
    
    all_tim = history_tim + (cur_tim)  #展平时间到当前的所有时间戳
    all_loc = history_loc + (cur_loc)
    all_day = history_day + (cur_day)
    all_hour = history_hour + (cur_hour)
    all_minute = history_minute + (cur_minute)
    all_delta_tim = history_delta_tim + (cur_delta_tim)
    
    all_session_based_tim = copy.deepcopy(session_based_history_tim) 
    all_session_based_tim.append(cur_tim) 
    all_session_based_loc = copy.deepcopy(session_based_history_loc)
    all_session_based_loc.append(cur_loc)
    all_session_based_delta_tim = copy.deepcopy(session_based_history_delta_tim)
    all_session_based_delta_tim.append(cur_delta_tim)
    all_session_based_day = copy.deepcopy(session_based_history_day)
    all_session_based_day.append(cur_day)
    all_session_based_hour = copy.deepcopy(session_based_history_hour)
    all_session_based_hour.append(cur_hour)
    all_session_based_minute = copy.deepcopy(session_based_history_minute)
    all_session_based_minute.append(cur_minute)
    sessions_duration2first = get_relativeTime(history_tim + (session_tim[:-1]))
    last_point_distances = [SS_gaussian_distance[lid] for lid in cur_loc]
    all_session_based_delta_tim = copy.deepcopy(session_based_history_delta_tim)
    all_session_based_delta_tim.append(cur_delta_tim)
    
    all_session_based_lengths_his = [len(session) for session in all_session_based_loc_his]
    length = sum(all_session_based_lengths_his) + n

    cacsr_result[f'{part}X_target_lengths'].append(n-1) 
    cacsr_result[f'{part}X_arrival_times'].append(all_tim)
    cacsr_result[f'{part}X_session_arrival_times'].append(all_session_based_tim)
    cacsr_result[f'{part}X_local_weekdays'].append(all_day)
    cacsr_result[f'{part}X_session_local_weekdays'].append(all_session_based_day) #星期几
    cacsr_result[f'{part}X_local_hours'].append(all_hour)
    cacsr_result[f'{part}X_session_local_hours'].append(all_session_based_hour)  #his(X)+cur(Y)
    cacsr_result[f'{part}X_local_mins'].append(all_minute)
    cacsr_result[f'{part}X_session_local_mins'].append(all_session_based_minute) #his(X)+cur(Y)
    cacsr_result[f'{part}X_delta_times'].append(all_delta_tim)
    cacsr_result[f'{part}X_session_delta_times'].append(all_session_based_delta_tim)  #his(X)+cur(Y)
    cacsr_result[f'{part}X_locations'].append(all_loc)
    cacsr_result[f'{part}X_session_locations'].append(all_session_based_loc) #his(X)+cur(Y)
    cacsr_result[f'{part}X_lengths'].append(length)  #当前session最后一个点在该用户所有点的位置
    cacsr_result[f'{part}Y_arrival_times'].append(target_tim)
    cacsr_result[f'{part}Y_delta_times'].append(target_delta_tim)
    cacsr_result[f'{part}Y_locations'].append(target_loc)
    cacsr_result[f'{part}X_session_num'].append(session_num)  #
    cacsr_result[f'{part}X_session_lengths'].append(session_based_lengths)  
    cacsr_result[f'{part}X_users'].append(uid)
    cacsr_result[f'{part}X_duration2first'].append(sessions_duration2first)
    cacsr_result[f'{part}X_last_distances'].append(last_point_distances)
    if part == 'test':
        cacsr_result['us'].append(cur_loc)
        cacsr_result['vs'].append(target_loc)
        


def get_mesh_lnglat(min_lng, min_lat, max_lng, max_lat, lng, lat):
    eps = 1e-7
    latN = 50
    lngN = 40
    latidx = int((lat-min_lat)*latN/(max_lat - min_lat + eps)) 
    lngidx = int((lng-min_lng)*lngN/(max_lng - min_lng + eps))
    latidx = latidx if latidx < latN else (latN-1)
    lngidx = lngidx if lngidx < lngN else (lngN-1)
    return latidx, lngidx 

def get_cat_idx(cat_list):
    # cat_list是一个包含所有cat名称（英文）的list
    category_idx = {}
    cnt2category = {}
    for cat in cat_list:
        if cat not in category_idx:
            category_idx[cat] = int(len(category_idx) + 1)
    for cat_name, idx in category_idx.items():
        if idx not in cnt2category:
            cnt2category[idx] = cat_name
    return category_idx, cnt2category 

def z_score_normalize(data, mean, std):
    """
    对轨迹数据进行 Z-Score 标准化。
    参数:
    - data: 待标准化的数据 (num_samples, 2, max_len)
    - mean: 每个维度的均值，形状为 (2,)
    - std: 每个维度的标准差，形状为 (2,)
    返回:
    - 标准化后的数据，形状同输入
    """
    return (data - mean[:, None]) / std[:, None]
# def get_encode_with_train(train_loc, val_loc, test_loc, map_loc_dict, map_continous):
#     user_map = {}
#     loc_map = {}
#     loc_result = {}
#     train_result = defaultdict(list)
#     val_result = defaultdict(list)
#     test_result = defaultdict(list)
#     if not map_continous:  # 初次生成
#         for user, sessions in train_loc.items():
#             if user not in user_map:
#                 user_map[user] = len(user_map)+1
#             for session in sessions:
#                 for loc in session:
#                     if loc not in loc_map:
#                         loc_map[loc] = len(loc_map)+1
#         for user, sessions in val_loc.items():
#             enc_user = user_map[user]
#             train_result[enc_user] = [[loc_map.get(loc,1) for loc in session] for session in train_loc[user]]
#             val_result[enc_user] = [[loc_map.get(loc,1) for loc in session] for session in sessions]
#             test_result[enc_user] = [[loc_map.get(loc,1) for loc in session] for session in test_loc[user]]
#         for loc, enc_loc in loc_map.items():
#             if enc_loc not in loc_result:
#                 loc_result[enc_loc] = [enc_loc, map_loc_dict[loc]['lat'],map_loc_dict[loc]['lon']]
#     else: # 基于之前的val,test重新编码,
#         for user, sessions in val_loc.items():
#             train_result[user] = [[map_continous.get(loc,1) for loc in session] for session in train_loc[user]]
#             val_result[user] = [[map_continous.get(loc,1) for loc in session] for session in sessions]
#             test_result[user] = [[map_continous.get(loc,1) for loc in session] for session in test_loc[user]]
#         for loc, enc_loc in map_continous.items():
#             if enc_loc not in loc_result:
#                 loc_result[enc_loc] = [enc_loc, map_loc_dict[loc]['lat'],map_loc_dict[loc]['lon']]
#     train_result = {k: train_result[k] for k in sorted(train_result.keys())}
#     val_result= {k: val_result[k] for k in sorted(val_result.keys())}
#     test_result = {k: test_result[k] for k in sorted(test_result.keys())}
#     loc_result = {k: loc_result[k] for k in sorted(loc_result.keys())}
#     return train_result, val_result, test_result, user_map, loc_result

            
        
    
    
         