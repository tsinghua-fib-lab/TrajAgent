import pandas as pd 
import os 
import json
import argparse
import numpy as np
from tqdm import tqdm
import uuid
from typing import Any
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split
from data_format.utils import format_utils
from UniEnv.etc.all_config import data_config, model_config
from UniEnv.etc.settings import *

# Generate trajectory sessions for training and input trajectory sessions for data augmentation

class PreAgent:
    def __init__(
        self,
        args:Any,           
        ):
        self.city=args.city
        self.dataset=args.dataset
        self.model=args.model
        self.args=args
        self.traj_mid_file_path = os.path.join(TRAJ_MID_PATH, self.dataset, self.city)
        if not os.path.exists(self.traj_mid_file_path):
            os.makedirs(self.traj_mid_file_path)
    
    def run(self):
        random_seed = 1     
        exist_file = False
        model_config = self.args.model_config[MODEL_TYPE[self.model]]
        # if self.args.dtype=='checkin':
        for file in os.listdir(self.traj_mid_file_path):
            if "traj_mid.csv" == file:
                exist_file = True
                break
        if not exist_file:
            user_traj = self.get_user_traj() #'dyna_id', 'type', 'time', 'entity_id', 'location', 'Longitude', 'Latitude'
            # user_traj in time order
        else:
            user_traj = pd.read_csv(os.path.join(self.traj_mid_file_path, "traj_mid.csv"))   
        sample_users = self.samples_generator(data=user_traj, threshold=model_config.threshold, user_freq = model_config.user_freq, seed=random_seed)
        # sample_users = sample_users[:10]
        res = self.get_cutter_filter_traj(self.args, sample_users, user_traj)
        if MODEL_TYPE[self.model] == 'GETNext':  #GETNext的编码方式很独特-使用训练集编码验证集与测试集，构图
            enc_res, enc_aug_time_train, enc_loc_train, results = self.get_unique_encode_graph(self.args, res)
            map_loc_dict = self.get_loc_map(self.args, enc_res)
            self.get_model_input(self.args, results, map_loc_dict)
            self.get_aug_input(self.args, enc_aug_time_train, enc_loc_train, map_loc_dict) 
        else:
            enc_res, enc_aug_time, enc_loc = self.get_unique_encode(self.args, res)
            map_loc_dict = self.get_loc_map(self.args, enc_res)  #在划分前统一连续编码
            self.get_model_input(self.args, enc_res, map_loc_dict) # 划分训练、验证、测试集
            self.get_aug_input(self.args, enc_aug_time, enc_loc, map_loc_dict) #enc_loc和enc_aug_loc是一样的
        # elif self.args.dtype=='gps':
        #     train_df, test_df = self.get_gps_traj()
        #     train_trajs = self.get_traj_pad(train_df)
        #     test_trajs = self.get_traj_pad(test_df)
        #     self.get_model_input_gps(args, train_trajs, 'train')
        #     self.get_model_input_gps(args, test_trajs, 'test')
            
    def get_gps_traj(self):
        data_config = self.args.data_config[self.dataset]
        model_config = self.args.model_config[MODEL_TYPE[self.model]]
        min_len = data_config.min_len
        max_len = data_config.max_len
        data_size = model_config.data_size
        # 加载数据
        df = pd.read_csv(os.path.join(PROCESS_DATA_INPUT_PATH, f"{self.args.dataset}",f"{self.args.dataset}.csv"), index_col=None,
                        usecols=['POLYLINE', 'NUM_POINT'], nrows=200000)
        df = df[(df['NUM_POINT'] >= min_len) & (df['NUM_POINT'] <= max_len)]
        df = df.iloc[:data_size]
        df['POLYLINE'] = df['POLYLINE'].map(eval)
        # 划分训练集和测试集
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print(f"train set size: {len(train_df)}, test set size: {len(test_df)}")
        return train_df, test_df
    
    def get_traj_pad(self, data):
        data_config = self.args.data_config[self.dataset]
        max_len = data_config.max_len
        min_len = data_config.min_len
        trajectories = []
        for polyline in data['POLYLINE']:
            traj = np.array(polyline)
            # 填充轨迹到 max_len
            padded_traj = np.zeros((max_len, 2))
            traj_len = len(traj)
            if traj_len <= max_len:
                padded_traj[:traj_len] = traj
            else:
                padded_traj = traj[:max_len]
            trajectories.append(padded_traj)
        return trajectories

    def get_model_input_gps(self, args, trajectories, part):
        data_config = self.args.data_config[self.dataset]
        max_len = data_config.max_len
        min_len = data_config.min_len
        output_path = os.path.join(PROCESS_DATA_OUTPUT_PATH, f"{args.dataset}/{MODEL_TYPE[args.model]}/Uni/{args.base_model}/{args.memory_length}")
        
        if not os.path.exists(output_path):
            os.makedirs(output_path) 
        if MODEL_TYPE[args.model] == 'TrajBERT':
            save_filename = os.path.join(output_path ,f'{MODEL_TYPE[args.model]}_{args.dataset}_{part}.npz')
            mask_ratio = 0.15
            attention_masks = []
            miss_masks = []
            miss_labels = []
            for trajectory in trajectories:
                trajectory = np.array(trajectory)
                traj_len = len(trajectory)
                padded_traj = np.zeros((max_len, 2))
                # 生成 attention_mask
                attn_mask = np.zeros(max_len, dtype=int)
                attn_mask[:min(traj_len, max_len)] = 1

                # 初始化 miss_mask 和 miss_label
                miss_mask = np.zeros(max_len, dtype=int)
                miss_label = np.zeros((max_len, 2))

                # 随机掩码 15% 的点(语言模型需要进行随机mask, 模型训练时需要根据上下文来预测这些被mask的词)
                if traj_len > 1:
                    num_to_mask = max(1, int(mask_ratio * traj_len))
                    mask_indices = np.random.choice(np.arange(traj_len), size=num_to_mask, replace=False)
                    for idx in mask_indices:
                        miss_mask[idx] = 1
                        miss_label[idx] = padded_traj[idx]
                        padded_traj[idx] = 0
                # 更新 attention_mask
                attn_mask = attn_mask & (1 - miss_mask)
                attention_masks.append(attn_mask)
                miss_masks.append(miss_mask)
                miss_labels.append(miss_label)
            trajectories = np.transpose(np.array(trajectories), (0, 2, 1))
            attention_masks = np.array(attention_masks)
            masks = np.array(miss_masks)
            labels = np.transpose(np.array(miss_labels), (0, 2, 1))
            np.savez_compressed(save_filename,
                            traj=trajectories,
                            attn_mask=attention_masks,
                            miss_mask=masks,
                            miss_label=labels)

    def get_user_traj(self):
    #city,user,time,venue_id,utc_time,lon,lat,venue_cat_name
        raw_data = pd.read_csv(os.path.join(FORMAT_DATA_OUTPUT_PATH,f"target_{self.dataset}_{self.city}.csv"), header=0,
                            names=['City','User ID','Time Offset', 'Venue ID',
                                'UTC time','Longitude','Latitude','Venue Category Name'], encoding='ISO-8859-1')
       # for geo message
        poi = raw_data[['Venue ID', 'Latitude', 'Longitude']]
        poi = poi.groupby('Venue ID').mean()
        poi.reset_index(inplace=True)

        category_info = raw_data.filter(items=['Venue ID', 'Venue Category Name'])
        category_info = category_info.drop_duplicates(['Venue ID'])
        loc_hash2ID = pd.merge(poi, category_info, on='Venue ID')

        # for dynatic message
        dyna = raw_data.filter(items=['User ID', 'Venue ID', 'Timezone offset in minutes', 'UTC time'])
        dyna = pd.merge(dyna, loc_hash2ID, on='Venue ID')
        dyna['Venue ID'],_ = pd.factorize(dyna['Venue ID'])
        dyna['User ID'],_ = pd.factorize(dyna['User ID'])
        dyna['Venue Category ID'],_ = pd.factorize(dyna['Venue Category Name'])
        dyna['time'] = dyna['UTC time'].apply(lambda x: datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y').strftime('%Y-%m-%dT%H:%M:%SZ'))
        dyna = dyna.rename(columns={'Venue ID': 'location', 'User ID': 'entity_id','Venue Category ID':'type','Venue Category Name':'type_name'})
        dyna = dyna.sort_values(by='time')
        dyna['dyna_id'] = dyna.index
        dyna = dyna.reindex(columns=['dyna_id', 'type', 'time', 'entity_id', 'location','Longitude','Latitude', 'type_name'])
        dyna.to_csv(os.path.join(self.traj_mid_file_path, "traj_mid.csv"), index=False, header=True, encoding='utf-8')
        return dyna

    def samples_generator(self, data, threshold=2000, user_freq=1, seed=1):
        # 'dyna_id', 'type', 'time', 'entity_id', 'location','Longitude','Latitude'
        # dyna_id,type,time,entity_id,location,Longitude,Latitude
        tmp = []
        sample_users = []
        np.random.seed(seed=seed)
        all_users = list(set(data['entity_id'].values))
        print("user nums of {data}:", len(all_users))
        for user in all_users:
            user = int(user)
            traj_points = data[data['entity_id']==user]
            trace_len = traj_points.shape[0]
            tmp.append([trace_len, user])
        # np.random.shuffle(tmp)
        tmp = sorted(tmp, key=lambda x: x[0], reverse=True)  # 轨迹数从多到少排序
        for trace_len, user in tmp:
            if trace_len >= user_freq:
                sample_users.append(user) 
        
        sample_users = sample_users[:threshold] #轨迹点数量最多的threshold个用户
        return sample_users

    def deal_cluster_sequence_for_each_user(self, sequence):
        max_merge_hours_limit = 3
        drops = 0
        keep = np.ones(len(sequence), dtype=bool)  # 用于标记哪些行需要保留
        # # 'dyna_id', 'type', 'time', 'entity_id', 'location','Longitude','Latitude'
        times = [datetime.strptime(row[2], '%Y-%m-%dT%H:%M:%SZ') for row in sequence]
        for index in range(len(sequence) - 1):  
            if not keep[index]: 
                continue
            current_poi_id = sequence[index, 4]  
            current_timestamp = times[index]
            dis = 1
            while index + dis < len(sequence) and keep[index + dis]:
                next_poi_id = sequence[index + dis, 4] 
                next_timestamp = times[index + dis]  
                # 如果 POI_id 相同且时间间隔小于阈值，标记为删除
                if current_poi_id == next_poi_id and format_utils.cal_timeoff(next_timestamp, current_timestamp) < max_merge_hours_limit:
                    keep[index + dis] = False  
                    drops += 1  
                    dis += 1  
                else:
                    break  # 如果不满足条件，停止继续检查后续记录
        return sequence[keep], drops
    
    # cut_data = self.cutter_filter()
    def get_cutter_filter_traj(self, args, sample_users, traj):
        """
        切割后的轨迹存储格式: (dict)
            {
                uid: [
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    ...
                ],
                ...
            }
        """
        # filter inactive poi
        # 'dyna_id', 'type', 'time', 'entity_id', 'location','Longitude','Latitude'
        model_config = self.args.model_config[MODEL_TYPE[args.model]]
        data_config = self.args.data_config[self.dataset]
        group_location = traj.groupby('location').count()
        filter_location = group_location[group_location['time'] >= model_config.loc_freq]
        location_index = filter_location.index.tolist()
        traj = traj[traj['location'].isin(location_index)]
        user_set = set(sample_users)
        res = {}
        # aug_time = {}
        min_session_len = data_config.min_session_len  
        if MODEL_TYPE[args.model] == "GETNext":
            min_session_len = 2
        max_session_len = data_config.max_session_len  
        min_sessions = data_config.min_sessions
        max_sessions = data_config.max_sessions
        window_size = model_config.window_size
        # use_day = model_config.use_day
        traj_num = 0
        # 按照时间窗口进行切割   
        # dyna_id,type,time,entity_id,location,Longitude,Latitude,type_name
        for uid in tqdm(user_set, desc="cut and filter trajectory"):
            usr_traj = traj[traj['entity_id'] == uid].to_numpy()
            sessions = []  # 存放该用户所有的 session
            # time_sessions = []
            session = []  # 单条轨迹
            # time_session = []
            usr_traj, drops = self.deal_cluster_sequence_for_each_user(usr_traj) # 如果checkin发生在同一地点，并且时间间隔小于指定的阈值（self.max_merge_seconds_limit），则将它们合并
            for index, row in enumerate(usr_traj):
                now_time = datetime.strptime(row[2], '%Y-%m-%dT%H:%M:%SZ')
                # traj_time = get_traj_time(use_day, row[2], MODEL_TYPE[args.model])
                # hour_point = get_traj_time(use_day, row[2], 'type4aug')
                if index == 0:
                    prev_time = now_time
                    # time_session.append(hour_point)
                    session.append([row[0],row[1],row[2],row[3],row[4],row[5],row[6], row[7]])   
                else:
                    time_off = format_utils.cal_timeoff(now_time, prev_time)
                    if time_off < window_size and time_off >= 0 and len(session) < max_session_len:
                        # time_session.append(hour_point)
                        session.append([row[0],row[1],row[2],row[3],row[4],row[5],row[6], row[7]])   
                    else:
                        if len(session) >= min_session_len:
                            sessions.append(session)
                            # time_sessions.append(time_session)
                        session = []   #放弃间隔过大的一部分离散轨迹点（与其他点间隔超出time-window,且数量少于min_session_len）
                        time_session = []
                        # time_session.append(hour_point)
                        session.append([row[0],row[1],row[2],row[3],row[4],row[5],row[6], row[7]])   
                prev_time = now_time
            if max_session_len >= len(session) >= min_session_len:
                sessions.append(session)
                # time_sessions.append(time_session)
            if max_sessions >= len(sessions) >= min_sessions:
                res[int(uid)] = sessions
                # aug_time[int(uid)] = time_sessions
            traj_num += len(sessions)
            print(f"traj num of {self.dataset}:", traj_num)
        return res

    def get_unique_encode(self, args, res):
        # 'dyna_id', 'type', 'time', 'entity_id', 'location'
        model_config = self.args.model_config[MODEL_TYPE[args.model]]
        use_day = model_config.use_day
        loc_map = {}
        loc_set = set()
        user_map = {}
        user_set = set()
        cat_map = {}
        cat_set = set()
        for _, sessions  in res.items():
            for session in sessions:
                for traj in session:
                    loc = int(traj[4])
                    user = int(traj[3])
                    loc_set.add(loc)
                    user_set.add(user)
                    if loc not in loc_map:
                        loc_map[loc] = len(loc_map)+1
                    if user not in user_map:
                        user_map[user] = len(user_map)+1
                    cat = int(traj[1])
                    cat_set.add(cat)
                    if cat not in cat_map:
                        cat_map[cat] = len(cat_map)+1
        encode_res = {} 
        encode_time = {}  
        encode_loc = {}  
        traj_id = 0  
        # 'dyna_id', 'type', 'time', 'entity_id', 'location' 
        for usr, sessions  in res.items():
            usr = int(usr)
            enc_usr = user_map[usr]
            if enc_usr not in encode_res:
                encode_res[enc_usr] = []
                encode_time[enc_usr] = []
                encode_loc[enc_usr] = []
            for idx1, session in enumerate(sessions):
                traj_id += 1
                res_session = []
                loc_session = []
                time_session = []
                for idx2, traj in enumerate(session):
                    loc = int(traj[4])
                    enc_loc = loc_map[loc]    
                    loc_session.append(int(enc_loc))
                    hour_point = format_utils.get_traj_time(use_day, traj[2], 'type4aug')
                    time_session.append(hour_point)
                    cat = int(traj[1])
                    enc_cat = cat_map[cat]
                    #'dyna_id', 'type', 'time', 'entity_id', 'location','Longitude','Latitude','type_name'
                    if MODEL_TYPE[args.model]=='LLM':
                        res_session.append([traj_id,enc_cat,traj[2], enc_usr, enc_loc,traj[5],traj[6],traj[7]])
                    else:
                        res_session.append([traj[0],enc_cat,traj[2], enc_usr, enc_loc,traj[5],traj[6],traj[7]])
                     
                encode_res[enc_usr].append(res_session)
                encode_time[enc_usr].append(time_session)
                encode_loc[enc_usr].append(loc_session)
        print("user nums:",len(encode_res),"POI nums:",len(loc_map))
        return encode_res, encode_time, encode_loc
    
    def get_unique_encode_graph(self, args, res):
         #'dyna_id', 'type', 'utc_time', 'entity_id', 'location','Longitude','Latitude', 'traj_id', sessions中的session是按时间排序的
        encode_time = {}  
        encode_res = {}  
        encode_loc = {}  
        res_session = []
        traj_id = 0  
        for usr, sessions  in res.items():  
            for idx1, session in enumerate(sessions):
                traj_id += 1 # 每个session的traj_id也是按时间递增的
                for traj in session:
                    if idx1 <= int(len(sessions)*0.7) - 1:
                        res_session.append([traj[0],traj[1],traj[2], traj[3], traj[4],traj[5],traj[6], traj[7], traj_id, 'train'])
                    elif int(len(sessions)*0.7) - 1 < idx1 <= int(len(sessions)*0.8) - 1:
                        res_session.append([traj[0],traj[1],traj[2], traj[3], traj[4],traj[5],traj[6], traj[7], traj_id, 'validation'])
                    else:
                        res_session.append([traj[0],traj[1],traj[2], traj[3], traj[4],traj[5],traj[6], traj[7], traj_id, 'test'])

        results = pd.DataFrame(res_session, columns=["check_ins_id", "POI_catid","UTC_time","user_id","POI_id","longitude","latitude","POI_catname", "trajectory_id_raw","SplitTag"])
        # 先划分之后，用训练集对其他进行编码
        results = format_utils.getnext_format(self.city, args.model, results)
        results = format_utils.getnext_encode(results)
        model_config = self.args.model_config[MODEL_TYPE[args.model]]
        use_day = model_config.use_day
        train_res = results[results['SplitTag']=='train']
        all_trajs = []
        train_res = train_res.sort_values(by=['user_id', 'timezone'], ascending=True)   #保证train_res中key(user)有序，对user做一下排序。且train_res中的值按时间排序，这样traj_id也可以递增
        for row in train_res.itertuples():
            usr = int(row.user_id)
            traj_point = int(row.trajectory_id_raw)
            raw_time = datetime.strptime(row.UTC_time, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ")
            hour_point = format_utils.get_traj_time(use_day, raw_time, 'type4aug')      
            loc = int(row.POI_id)
            if usr not in encode_res:
                encode_time[usr] = []
                encode_res[usr] = []
                encode_loc[usr] = []
            if traj_point not in all_trajs:
                encode_time[usr].append([hour_point])
                encode_res[usr].append([[int(row.check_ins_id), int(row.POI_catid), row.UTC_time, int(row.user_id), int(row.POI_id), row.longitude, row.latitude, row.POI_catname, int(row.trajectory_id_raw), row.SplitTag]])
                encode_loc[usr].append([loc])
                all_trajs.append(traj_point)
            else:
                encode_time[usr][-1].append(hour_point)
                encode_loc[usr][-1].append(loc)
                encode_res[usr][-1].append([int(row.check_ins_id), int(row.POI_catid), row.UTC_time, int(row.user_id), int(row.POI_id), row.longitude, row.latitude, row.POI_catname, int(row.trajectory_id_raw), row.SplitTag])
        return encode_res, encode_time, encode_loc, results # results用来训练，encode_res, encode_time, encode_loc用来数据增强
    
    
    def get_loc_map(self, args, enc_res):
        # "check_ins_id", "POI_catid","UTC_time","user_id","POI_id","longitude","latitude","POI_catname", "trajectory_id_raw","SplitTag
        map_loc_dict = {}
        if MODEL_TYPE[args.model] == 'CACSR':  
            all_lons = []
            all_lats = []
            for _, sessions in enc_res.items():
                all_lons.extend([traj[5] for session in sessions for traj in session])
                all_lats.extend([traj[6] for session in sessions for traj in session])
                min_lng = min(all_lons)
                min_lat = min(all_lats)
                max_lng = max(all_lons)
                max_lat = max(all_lats)
            for _, sessions in enc_res.items():
                for session in sessions:
                    for traj in session:
                        loc = traj[4]
                        lng = traj[5]
                        lat = traj[6]
                        cat = traj[1]
                        cat_name = traj[7]
                        latidx, lngidx = format_utils.get_mesh_lnglat(min_lng, min_lat, max_lng, max_lat, lng, lat)
                        if loc not in map_loc_dict:
                            map_loc_dict[loc] = {
                                "lon":lng,
                                "lat":lat,
                                "cat":cat,
                                "lngidx":lngidx,
                                "latidx":latidx,
                                "cat_name":cat_name,
                            }
        else:
            for _, sessions in enc_res.items():
                for session in sessions:
                    for traj in session:
                        lon = traj[5]
                        lat = traj[6]
                        cat = traj[1]
                        cat_name = traj[7]
                        loc = traj[4]
                        if loc not in map_loc_dict:
                            map_loc_dict[loc] = {
                                # "traj_id":traj_id,
                                "lon":lon,
                                "lat":lat,
                                "cat":cat,
                                "cat_name":cat_name,
                                "loc":loc
                            }
        return map_loc_dict
            
    # 用于模型训练数据输入
    def get_model_input(self, args, res, map_loc_dict):
        model_config = self.args.model_config[MODEL_TYPE[args.model]]
        use_day = model_config.use_day
        output_path = os.path.join(PROCESS_DATA_OUTPUT_PATH, f"{args.dataset}/{MODEL_TYPE[args.model]}/{args.city}/{args.base_model}/{args.memory_length}")
        # {dataset}/{MODEL_TYPE[model]}/{city}/{base_model}/{memory_length}
        if not os.path.exists(output_path):
            os.makedirs(output_path) 
        if MODEL_TYPE[args.model] == "LibCity": 
            res_all = {}
            for user, sessions in res.items():
                if user not in res_all:
                    res_all[int(user)] = []
                for session in sessions:
                    # [dyna_id, 'trajectory', time, entity_id, location]
                    all_session = [[x[0],'trajectory', format_utils.get_traj_time(use_day, x[2], MODEL_TYPE[args.model]),x[3],x[4]] for x in session]
                    res_all[int(user)].append(all_session)
            with open(os.path.join(output_path,"1000_1000.json"),'w') as f:
                json.dump(res_all, f) 
        elif MODEL_TYPE[args.model] == "LLM": 
            # 'traj_id', 'type', 24h, weekday, 'entity_id', 'location','Longitude','Latitude','type_name'
            test_res = {}
            train_res = {}
            for user, sessions in res.items():
                if user not in test_res:
                    test_res[int(user)] = []
                    train_res[int(user)] = []
                for session in sessions:
                    all_session = [[x[0],x[1], format_utils.hours_to_time(format_utils.get_traj_time(use_day, x[2], MODEL_TYPE[args.model]),hr_24h=True),format_utils.hours_to_time(format_utils.get_traj_time(use_day, x[2], MODEL_TYPE[args.model]),weekday=True),x[3],x[4],x[5],x[6],x[7]] for x in session]
                    test_res[int(user)].append(all_session[int(len(all_session)*0.7):])
                    train_res[int(user)].append(all_session[:int(len(all_session)*0.7)])
            with open(os.path.join(output_path,"test_1000_1000.json"),'w') as f:
                json.dump(test_res, f) 
            with open(os.path.join(output_path,"train_1000_1000.json"),'w') as f:
                json.dump(train_res, f) 
        elif MODEL_TYPE[args.model] == 'ActSTD':
            event_cnt = set()
            for k,v in map_loc_dict.items():
                event_cnt.add(v['cat'])
            res_all = []
            train_res = []
            val_res = []
            test_res = []
            for user, sessions in res.items():
                if user not in res_all:
                    user_session = []
                for session in sessions:
                    for item in session:
                        traj_point = format_utils.get_traj_time(use_day, item[2], MODEL_TYPE[args.model])
                        user_session.append([traj_point,item[1],[item[5],item[6]]])
                res_all.append(user_session)
                train_res.append(user_session[:int(len(user_session)*0.7)])
                val_res.append(user_session[int(len(user_session)*0.7):int(len(user_session)*0.8)])
                test_res.append(user_session[int(len(user_session)*0.8):])
            train_res = {'train':train_res,'num_event':len(event_cnt)}
            with open(os.path.join(output_path,"1000_1000_train.json"),'w') as f:
                json.dump(train_res, f) 
            with open(os.path.join(output_path,"1000_1000_val.json"),'w') as f:
                json.dump(val_res, f) 
            with open(os.path.join(output_path,"1000_1000_test.json"),'w') as f:
                json.dump(test_res, f) 
                
        elif MODEL_TYPE[args.model] in ['MainTUL','DPLink']:
            res_loc = {}
            res_time = {}
            res_cat = {}
            res_all = {}
            for user, sessions in res.items():
                if user not in res_loc:
                    res_loc[int(user)] = []
                    res_time[int(user)] = []
                    res_cat[int(user)] = []
                    res_all[int(user)] = []
                for session in sessions:
                    time_session = [int(format_utils.get_traj_time(use_day, x[2], MODEL_TYPE[args.model])) for x in session]
                    loc_session = [int(x[4]) for x in session]
                    cat_session = [int(x[1]) for x in session]
                    all_session = [x[:-1] for x in session]
                    res_loc[int(user)].append(loc_session)
                    res_cat[int(user)].append(cat_session)
                    res_time[int(user)].append(time_session)
                    res_all[int(user)].append(all_session)
            with open(os.path.join(output_path,"1000_1000_all.json"),'w') as f:
                json.dump(res_all, f)   
            with open(os.path.join(output_path,"1000_1000_loc.json"),'w') as f:
                json.dump(res_loc, f)
            with open(os.path.join(output_path,"1000_1000_time.json"),'w') as f:
                json.dump(res_time, f)
            with open(os.path.join(output_path,"1000_1000_cat.json"),'w') as f:
                json.dump(res_cat, f)
                
        elif MODEL_TYPE[args.model] == 'CACSR':
            train_res = defaultdict(list)
            val_res = defaultdict(list)
            test_res = defaultdict(list)
            cacsr_train_result = defaultdict(list)
            cacsr_val_result = defaultdict(list)
            cacsr_test_result = defaultdict(list)
            res_loc_train = {}
            cat_list = []
            venue_cnt = len(map_loc_dict.keys())  # 看起来venue_id需要连续编码
            distance_theta = 1
            gaussian_beta=10
            SS_distance, SS_proximity, SS_gaussian_distance = format_utils.construct_spatial_matrix_accordingDistance(distance_theta, venue_cnt, map_loc_dict, gaussian_beta)
            for user, sessions in res.items():
                train_res[user]=sessions[:int(len(sessions)*0.7)]
                val_res[user]=sessions[int(len(sessions)*0.7):int(len(sessions)*0.8)]
                test_res[user]=sessions[int(len(sessions)*0.8):]
                cat_list.extend([item[7] for session in sessions for item in session])
            category_idx, cnt2category = format_utils.get_cat_idx(list(set(cat_list)))
            for user, sessions in train_res.items():            
                all_time = [[int(format_utils.get_traj_time(use_day, traj[2], MODEL_TYPE[args.model])) for traj in ses] for ses in sessions]  # session-based 
                all_flat_time = [format_utils.hours_to_time(item) for ses in all_time for item in ses]
                all_delta_times = format_utils.get_delta(all_flat_time)
                all_location = [[int(traj[4]) for traj in ses] for ses in sessions]
                session_st = [time_session[0] for time_session in all_time] #该用户的每个session的起始时间
                if user not in res_loc_train:
                    res_loc_train[int(user)] = []
                for idx, session in enumerate(sessions):
                    loc_session = [int(x[4]) for x in session]
                    res_loc_train[int(user)].append(loc_session)
                    now_time = all_time[idx][1]
                    st = 0
                    for idx2, time_point in enumerate(session_st):  #idx是当前session的编号
                        if now_time - time_point < 28*24:
                            st = idx2
                            break
                    start_pos = 0
                    end_pos = 0
                    for cnt, session in enumerate(sessions):
                        if cnt < st:
                            start_pos += len(session)
                    for cnt, session in enumerate(sessions):
                        if cnt < idx:
                            end_pos += len(session)
                    session_based_hour_hct = all_time[st: idx + 1] # his不包含当前session,context为当前session的前n-1个元素
                    session_based_loc_hct = all_location[st: idx + 1]
                    all_session_based_loc_his = all_location[: idx]
                    history_delta_hour = all_delta_times[start_pos:end_pos]
                    format_utils.cacsr_format_session(SS_gaussian_distance, all_session_based_loc_his, session_based_loc_hct, session_based_hour_hct, history_delta_hour, cacsr_train_result, user, 'train')
            for user, sessions in val_res.items():
                all_time = [[int(format_utils.get_traj_time(use_day, traj[2], MODEL_TYPE[args.model])) for traj in ses] for ses in sessions]  # session-based 
                all_flat_time = [format_utils.hours_to_time(item) for ses in all_time for item in ses]
                all_delta_times = format_utils.get_delta(all_flat_time)
                all_location = [[int(traj[4]) for traj in ses] for ses in sessions]
                session_st = [time_session[0] for time_session in all_time] 
                if user not in res_loc_train:
                    res_loc_train[int(user)] = []
                for idx, session in enumerate(sessions):
                    loc_session = [int(x[4]) for x in session]
                    res_loc_train[int(user)].append(loc_session)
                    now_time = all_time[idx][1]
                    st = 0
                    for idx2, time_point in enumerate(session_st):  
                        if now_time - time_point < 28*24:
                            st = idx2
                            break
                    start_pos = 0
                    end_pos = 0
                    for cnt, session in enumerate(sessions):
                        if cnt < st:
                            start_pos += len(session)
                    for cnt, session in enumerate(sessions):
                        if cnt < idx:
                            end_pos += len(session)
                    session_based_hour_hct = all_time[st: idx + 1] 
                    session_based_loc_hct = all_location[st: idx + 1]
                    all_session_based_loc_his = all_location[: idx]
                    history_delta_hour = all_delta_times[start_pos:end_pos]
                    format_utils.cacsr_format_session(SS_gaussian_distance, all_session_based_loc_his, session_based_loc_hct, session_based_hour_hct, history_delta_hour, cacsr_val_result, user, 'val')
            for user, sessions in test_res.items():
                all_time = [[int(format_utils.get_traj_time(use_day, traj[2], MODEL_TYPE[args.model])) for traj in ses] for ses in sessions]  # session-based 
                all_flat_time = [format_utils.hours_to_time(item) for ses in all_time for item in ses]
                all_delta_times = format_utils.get_delta(all_flat_time)
                all_location = [[int(traj[4]) for traj in ses] for ses in sessions]
                session_st = [time_session[0] for time_session in all_time] 
                if user not in res_loc_train:
                    res_loc_train[int(user)] = []
                for idx, session in enumerate(sessions):
                    loc_session = [int(x[4]) for x in session]
                    res_loc_train[int(user)].append(loc_session)
                    now_time = all_time[idx][1]
                    st = 0
                    for idx2, time_point in enumerate(session_st):  
                        if now_time - time_point < 28*24:
                            st = idx2
                            break
                    start_pos = 0
                    end_pos = 0
                    for cnt, session in enumerate(sessions):
                        if cnt < st:
                            start_pos += len(session)
                    for cnt, session in enumerate(sessions):
                        if cnt < idx:
                            end_pos += len(session)
                    session_based_hour_hct = all_time[st: idx + 1] # his不包含当前session,context为当前session的前n-1个元素
                    session_based_loc_hct = all_location[st: idx + 1]
                    all_session_based_loc_his = all_location[: idx]
                    history_delta_hour = all_delta_times[start_pos:end_pos]
                    format_utils.cacsr_format_session(SS_gaussian_distance, all_session_based_loc_his, session_based_loc_hct, session_based_hour_hct, history_delta_hour, cacsr_test_result, user, 'test')
            loc_num = len(map_loc_dict.keys())
            user_num = len(res.keys())  #均为训练+验证+测试的全统计值
            user_lid_freq_train = format_utils.get_user_lidFreq(res_loc_train, loc_num)
            result_lon_ori,result_lat_ori, result_feature_lon, result_feature_lat, result_feature_cat, cat_num = format_utils.get_loc_feature(map_loc_dict, loc_num)
            train_save_filename = os.path.join(output_path ,f'1000_1000_train.npz')
            val_save_filename = os.path.join(output_path ,f'1000_1000_val.npz')
            test_save_filename = os.path.join(output_path ,f'1000_1000_test.npz')
            cat_save_filename = os.path.join(output_path ,'cnt2category2cnt.npz')
            
            np.savez_compressed(cat_save_filename, category_idx=category_idx, cnt2category=cnt2category, dtype=object)
            print('Save train dataset...')
            np.savez_compressed(train_save_filename,
                    trainX_target_lengths=np.array(cacsr_train_result['trainX_target_lengths']),
                    trainX_arrival_times=np.array(cacsr_train_result['trainX_arrival_times'],dtype=object),
                    trainX_duration2first=np.array(cacsr_train_result['trainX_duration2first'],dtype=object),
                    trainX_session_arrival_times=np.array(cacsr_train_result['trainX_session_arrival_times'],dtype=object),
                    trainX_local_weekdays=np.array(cacsr_train_result['trainX_local_weekdays'],dtype=object),
                    trainX_session_local_weekdays=np.array(cacsr_train_result['trainX_session_local_weekdays'],dtype=object),
                    trainX_local_hours=np.array(cacsr_train_result['trainX_local_hours'],dtype=object),
                    trainX_session_local_hours=np.array(cacsr_train_result['trainX_session_local_hours'],dtype=object),
                    trainX_local_mins=np.array(cacsr_train_result['trainX_local_mins'],dtype=object),
                    trainX_session_local_mins=np.array(cacsr_train_result['trainX_session_local_mins'],dtype=object),
                    trainX_delta_times=np.array(cacsr_train_result['trainX_delta_times'],dtype=object),
                    trainX_session_delta_times=np.array(cacsr_train_result['trainX_session_delta_times'],dtype=object),
                    trainX_locations=np.array(cacsr_train_result['trainX_locations'],dtype=object),
                    trainX_session_locations=np.array(cacsr_train_result['trainX_session_locations'],dtype=object),
                    trainX_last_distances=np.array(cacsr_train_result['trainX_last_distances'],dtype=object),
                    trainX_users=np.array(cacsr_train_result['trainX_users']), trainX_lengths=np.array(cacsr_train_result['trainX_lengths']),
                    trainX_session_lengths=np.array(cacsr_train_result['trainX_session_lengths'],dtype=object),
                    trainX_session_num=np.array(cacsr_train_result['trainX_session_num']),
                    trainY_arrival_times=np.array(cacsr_train_result['trainY_arrival_times'],dtype=object),
                    trainY_delta_times=np.array(cacsr_train_result['trainY_delta_times'],dtype=object),
                    trainY_locations=np.array(cacsr_train_result['trainY_locations'],dtype=object),
                    user_lidfreq=user_lid_freq_train, 
                    dtype=object)
            print('Save test dataset...')
            np.savez_compressed(test_save_filename,
                    testX_target_lengths=np.array(cacsr_test_result['testX_target_lengths']),
                    testX_arrival_times=np.array(cacsr_test_result['testX_arrival_times'],dtype=object),
                    testX_duration2first = np.array(cacsr_test_result['testX_duration2first'],dtype=object),
                    testX_session_arrival_times = np.array(cacsr_test_result['testX_session_arrival_times'],dtype=object),
                    testX_local_weekdays=np.array(cacsr_test_result['testX_local_weekdays'],dtype=object),
                    testX_session_local_weekdays=np.array(cacsr_test_result['testX_session_local_weekdays'],dtype=object),
                    testX_local_hours=np.array(cacsr_test_result['testX_local_hours'],dtype=object),
                    testX_session_local_hours=np.array(cacsr_test_result['testX_session_local_hours'],dtype=object),
                    testX_local_mins=np.array(cacsr_test_result['testX_local_mins'],dtype=object),
                    testX_session_local_mins=np.array(cacsr_test_result['testX_session_local_mins'],dtype=object),
                    testX_delta_times=np.array(cacsr_test_result['testX_delta_times'],dtype=object),
                    testX_session_delta_times=np.array(cacsr_test_result['testX_session_delta_times'],dtype=object),
                    testX_locations=np.array(cacsr_test_result['testX_locations'],dtype=object),
                    testX_session_locations=np.array(cacsr_test_result['testX_session_locations'],dtype=object),
                    testX_last_distances=np.array(cacsr_test_result['testX_last_distances'],dtype=object),
                    testX_users=np.array(cacsr_test_result['testX_users']), testX_lengths=np.array(cacsr_test_result['testX_lengths']),
                    testX_session_lengths=np.array(cacsr_test_result['testX_session_lengths'],dtype=object),
                    testX_session_num=np.array(cacsr_test_result['testX_session_num']),
                    testY_arrival_times=np.array(cacsr_test_result['testY_arrival_times'],dtype=object),
                    testY_delta_times=np.array(cacsr_test_result['testY_delta_times'],dtype=object),
                    testY_locations=np.array(cacsr_test_result['testY_locations'],dtype=object), 
                    us=np.array(cacsr_test_result['us'],dtype=object), vs=np.array(cacsr_test_result['vs'],dtype=object),
                    feature_category=np.array(result_feature_cat), feature_lat=np.array(result_feature_lat), feature_lng=np.array(result_feature_lon),
                    feature_lat_ori=np.array(result_lat_ori), feature_lng_ori=np.array(result_lon_ori),  
                    latN=50, lngN=40, category_cnt=cat_num + 1, #int
                    user_cnt=user_num + 1, venue_cnt=loc_num + 1, # 这里都+1,预留足够空间
                    SS_distance=np.array(SS_distance), SS_guassian_distance=np.array(SS_gaussian_distance))
            print('Save val dataset ...')
            np.savez_compressed(val_save_filename,
                                valX_target_lengths=np.array(cacsr_val_result['valX_target_lengths']),
                                valX_arrival_times=np.array(cacsr_val_result['valX_arrival_times'],dtype=object),
                                valX_duration2first=np.array(cacsr_val_result['valX_duration2first'],dtype=object),
                                valX_session_arrival_times=np.array(cacsr_val_result['valX_session_arrival_times'],dtype=object),
                                valX_local_weekdays=np.array(cacsr_val_result['valX_local_weekdays'],dtype=object),
                                valX_session_local_weekdays=np.array(cacsr_val_result['valX_session_local_weekdays'],dtype=object),
                                valX_local_hours=np.array(cacsr_val_result['valX_local_hours'],dtype=object),
                                valX_session_local_hours=np.array(cacsr_val_result['valX_session_local_hours'],dtype=object),
                                valX_local_mins=np.array(cacsr_val_result['valX_local_mins'],dtype=object),
                                valX_session_local_mins=np.array(cacsr_val_result['valX_session_local_mins'],dtype=object),
                                valX_delta_times=np.array(cacsr_val_result['valX_delta_times'],dtype=object),
                                valX_session_delta_times=np.array(cacsr_val_result['valX_session_delta_times'],dtype=object),
                                valX_locations=np.array(cacsr_val_result['valX_locations'],dtype=object),
                                valX_session_locations=np.array(cacsr_val_result['valX_session_locations'],dtype=object),
                                valX_last_distances=np.array(cacsr_val_result['valX_last_distances'],dtype=object),
                                valX_users=np.array(cacsr_val_result['valX_users']), valX_lengths=np.array(cacsr_val_result['valX_lengths']),
                                valX_session_lengths=np.array(cacsr_val_result['valX_session_lengths'],dtype=object),
                                valX_session_num=np.array(cacsr_val_result['valX_session_num']),
                                valY_arrival_times=np.array(cacsr_val_result['valY_arrival_times'],dtype=object),
                                valY_delta_times=np.array(cacsr_val_result['valY_delta_times'],dtype=object),
                                valY_locations=np.array(cacsr_val_result['valY_locations'],dtype=object), 
                                dtype=object)
                    
        elif MODEL_TYPE[args.model] == "GETNext":
    # #'dyna_id', 'type', 'utc_time', 'entity_id', 'location','Longitude','Latitude', 'traj_id','VAL'
 # "check_ins_id", "POI_catid","UTC_time","user_id","POI_id","longitude","latitude","POI_catname", "trajectory_id_raw","SplitTag"
            sample_file = os.path.join(output_path, '1000_1000_sample.csv')
            train_file = os.path.join(output_path, '1000_1000_train.csv')
            validate_file = os.path.join(output_path, '1000_1000_val.csv')
            test_file = os.path.join(output_path, '1000_1000_test.csv')
            res.to_csv(sample_file, index=False)
            res[res['SplitTag'] == 'train'].to_csv(train_file, index=False)
            res[res['SplitTag'] == 'validation'].to_csv(validate_file, index=False)
            res[res['SplitTag'] == 'test'].to_csv(test_file, index=False)
            
        elif MODEL_TYPE[args.model]=='S2TUL':
        #'dyna_id', 'type', 'time', 'entity_id', 'location','Longitude','Latitude','type_name'
        # 233522, 1, 40170, 1, 1, -73.987396, 40.765685, 'Karaoke Bar'
            train_res = {}
            val_res = {}
            test_res = {}
            train_time_res = {}
            train_time_res2 = {}
            val_time_res2 = {}
            test_time_res2 = {}
            val_time_res = {}
            test_time_res = {}
            train_list = []
            val_list = []
            test_list = []
            train_time_list = []
            val_time_list = []
            test_time_list = []
            venue_lonlat = []
            for user, sessions in res.items():
                train_res[user]=sessions[:int(len(sessions)*0.7)]
                val_res[user]=sessions[int(len(sessions)*0.7):int(len(sessions)*0.8)]
                test_res[user]=sessions[int(len(sessions)*0.8):]
            for user, sessions in train_res.items():
                val_sessions = val_res[user]
                test_sessions = test_res[user]
                train_res[user] = [[item[4] for item in ses] for ses in sessions]
                train_time_res[user] = [[format_utils.get_traj_time(use_day, item[2], MODEL_TYPE[args.model]) for item in ses] for ses in sessions]
                val_res[user] = [[item[4] for item in ses] for ses in val_sessions]
                val_time_res[user] = [[format_utils.get_traj_time(use_day, item[2], MODEL_TYPE[args.model]) for item in ses] for ses in val_sessions]
                test_res[user] = [[item[4] for item in ses] for ses in test_sessions]
                test_time_res[user] = [[format_utils.get_traj_time(use_day, item[2], MODEL_TYPE[args.model]) for item in ses] for ses in test_sessions]
            # train_res, val_res, test_res, user_map,loc_info = get_encode_with_train(train_res, val_res, test_res, map_loc_dict, None)
            for user, sessions in train_time_res.items():
                if user not in train_time_res2:
                    train_time_res2[user] = sessions
            for user, sessions in val_time_res.items():
                if user not in val_time_res2:
                    val_time_res2[user] = sessions
            for user, sessions in test_time_res.items():
                if user not in test_time_res2:
                    test_time_res2[user] = sessions
            train_time_res2 = {k: train_time_res2[k] for k in sorted(train_time_res2.keys())}
            val_time_res2 = {k: val_time_res2[k] for k in sorted(val_time_res2.keys())}
            test_time_res2 = {k: test_time_res2[k] for k in sorted(test_time_res2.keys())}
            for user in sorted(train_res.keys()):
                val_sessions = val_res[user]
                test_sessions = test_res[user]
                train_sessions = train_res[user]
                train_list.extend([[user]+[item for item in ses] for ses in train_sessions])
                val_list.extend([[user]+[item for item in ses] for ses in val_sessions])
                test_list.extend([[user]+[item for item in ses] for ses in test_sessions])
                train_time_list.extend(train_time_res2[user])
                val_time_list.extend(val_time_res2[user])
                test_time_list.extend(test_time_res2[user])
            for loc in map_loc_dict:
                # enc_loc, map_loc_dict[loc]['lat'],map_loc_dict[loc]['lon']
                venue_lonlat.extend([[loc, map_loc_dict[loc]['lat'],map_loc_dict[loc]['lon']]])
            with open(os.path.join(output_path,"1000_1000_train_trajs.txt"), 'w') as f:
                for row in train_list:
                    line = ' '.join(map(str, row)) + '\n'
                    f.write(line)              
            with open(os.path.join(output_path,"1000_1000_vidx_to_latlon.txt"), 'w') as f:
                for row in venue_lonlat:
                    line = ' '.join(map(str, row)) + '\n'
                    f.write(line)
            with open(os.path.join(output_path,"1000_1000_val_trajs.txt"), 'w') as f:
                for row in val_list:
                    line = ' '.join(map(str, row)) + '\n'
                    f.write(line)
            with open(os.path.join(output_path,"1000_1000_test_trajs.txt"), 'w') as f:
                for row in test_list:
                    line = ' '.join(map(str, row)) + '\n'
                    f.write(line)
            with open(os.path.join(output_path,"1000_1000_train_trajs_time.txt"), 'w') as f:
                for row in train_time_list:
                    line = ' '.join(map(str, row)) + '\n'
                    f.write(line)
            with open(os.path.join(output_path,"1000_1000_val_trajs_time.txt"), 'w') as f:
                for row in val_time_list:
                    line = ' '.join(map(str, row)) + '\n'
                    f.write(line)
            with open(os.path.join(output_path,"1000_1000_test_trajs_time.txt"), 'w') as f:
                for row in test_time_list:
                    line = ' '.join(map(str, row)) + '\n'
                    f.write(line)
        else:
            print("Model not support!!")

    # 用于数据增强的数据输入            
    def get_aug_input(self, args, res_time, res_loc, map_loc_dict):
        output_path = os.path.join(AUGMENT_DATA_INPUT_PATH, f"{args.dataset}/{MODEL_TYPE[args.model]}/{args.city}")
        save_path = os.path.join(OP_MID_DATA_PATH, args.dataset, args.model, args.city)
        if not os.path.exists(output_path):
            os.makedirs(output_path) 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if MODEL_TYPE[args.model] in ['MainTUL','DPLink', 'LibCity','GETNext']:
            res_time2 = res_time
            res_loc2 = res_loc
        else:
            res_time2 = defaultdict(list)
            res_loc2 = defaultdict(list)
            for user, sessions in res_time.items():
                loc_sessions = res_loc[user]
                res_time2[user]=sessions[:int(len(sessions)*0.7)]
                res_loc2[user]=loc_sessions[:int(len(loc_sessions)*0.7)]
        with open(os.path.join(output_path,"loc_list.json"),'w') as f:
            json.dump(res_loc2, f)
        with open(os.path.join(output_path,"time_list.json"),'w') as f:
            json.dump(res_time2, f)
        with open(os.path.join(output_path,"map_loc_dict.json"),'w') as f:
            json.dump(map_loc_dict, f)
        
        flag = ''.join(str(uuid.uuid4()).split('-'))
        with open(os.path.join(save_path, "uuid.json"),"w") as f:  #Update Memory_dict each time
            json.dump(flag, f)

def pre_main_checkin(city: str, dataset: str, model:str, data_config:Any, model_config:Any, base_model:str, memory_length:int):
    class Args:
        def __init__(self,
                    city: str,
                    dataset: str,
                    model: str,
                    data_config:Any, 
                    model_config:Any,
                    base_model: str,
                    memory_length: int,                
                    ):
            self.city=city 
            self.dataset=dataset
            self.model=model
            self.data_config=data_config
            self.model_config=model_config
            self.base_model=base_model
            self.memory_length=memory_length
    args = Args(
            city=city, 
            dataset=dataset,
            model=model,
            model_config=model_config,
            data_config=data_config,
            base_model=base_model,
            memory_length=memory_length,
            )
    pre_agent = PreAgent(args)
    pre_agent.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", choices=["foursquare", "gowalla", "brightkite", "standard", "geolife","chengdu","porto","agentmove"])
    parser.add_argument("--city", type=str)
    parser.add_argument("--base_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--memory_length", type=int, default="1")
    args = parser.parse_args()

    pre_main_checkin(base_model=args.base_model, memory_length=args.memory_length, city=args.city, dataset=args.dataset, model=args.model, model_config=model_config, data_config=data_config)
    print("Done!!")
