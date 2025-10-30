import os
import yaml
import time
import json
import uuid
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
import subprocess
import shutil 
# from data_augmentation.utils.data_transform import *
from yaml.loader import SafeLoader
from collections import Counter
import pandas as pd
from data_augmentation.utils import (
    operator_dict
)
from data_format.utils.utils import getnext_encode, getnext_format, reform_test_val, hours_to_time, cacsr_format_session, get_user_lidFreq, construct_spatial_matrix_accordingDistance, get_delta, get_traj_time
from UniEnv.etc.settings import MODEL_TYPE, AUGMENT_DATA_INPUT_PATH, AUG_LIST, EVALUATE_METRIC, BASE_MODEL_PATH, OPERATOR_DICT

tmp_id = "_tmp"

class Augment:
    def __init__(self, aug_data_path, aug_method, time_sample, item_sample, city, dataset, model, da_config, da_pa_config_file_path):
        self.aug_method = aug_method
        self.time_sample = time_sample
        self.item_sample = item_sample
        self.city = city
        self.aug_data_path = aug_data_path
        self.dataset = dataset
        self.model = model
        self.da_config = da_config
        self.da_pa_config_file_path = da_pa_config_file_path

class ParallelDA(object):
    def __init__(self, output_path, config_file):
        self.output_path = output_path
        self.tasks = set()
        self.semaphore = BoundedSemaphore(1)
        self.executor_pool = None
        self.indicator = None
        self.out_semaphore = BoundedSemaphore(1)
        self.max_worker = 4

        self.train_instances = None
        self.item_counter = None
        self.online_config = None

        self.submit_set = set()
        
        # with open(config_file) as f:
        #     params = yaml.load(f, Loader=SafeLoader)
        # self.params = params
    
    def generate_da(self, args, all_instances, all_timestamps, traj): 
        self.executor_pool = ThreadPoolExecutor(self.max_worker)
        result_time = {}
        result_loc = {}
        self.aug_base = args.aug_method
        config_file = os.path.join(args.da_pa_config_file_path, f"{OPERATOR_DICT[self.aug_base]}.yaml")
        with open(config_file) as f:
            params = yaml.load(f, Loader=SafeLoader)
            
        with open(args.da_config) as f:
            all_params = yaml.load(f, Loader=SafeLoader)

        self.item_sample = args.item_sample
        print(f"Data augmenting for aug_base: '{self.aug_base}'")
        start_time = time.time()
        
        # Enhance each session for each user separately.
        for uid, instances in all_instances.items():  
            result_time[uid] = []
            result_loc[uid] = []
            timestamps = all_timestamps[uid]
            # Note: call once!
            self.train_instances = instances
            self.timestamps = timestamps
            self.item_counter = self.item_pop(instances)
            self.pos_sample = "time"

            # Check aug_base
            # Disable aug_base when on_the_fly is on
            if self.aug_base is not None:  #离线数据增强
                # params = None
                # with open(config_file) as f:
                #     params = yaml.load(f, Loader=SafeLoader)
                item_counter = self.item_pop(instances)
                params["pop_counter"] = item_counter
                items = list(item_counter.keys())
                augment_operator = operator_dict[self.aug_base](items)
                params["pos"] = self.pos_sample
                config = params.copy()
                all_config = all_params.copy()
                config.update({k: v for k, v in all_config.items() if k not in config})
                # args, instances, timestamps, uid, traj, **kwargs
                augment_operator.init(args, all_instances, timestamps, uid, traj, **config)
                # Note: Only support unique DA methods
                config["operation_type"] = self.aug_base
                # if self.aug_base in AUG_METHODS:
                #     crop_nums = all_config["ti_threshold"]
                # elif self.aug_base=="Ti-crop":
                #     crop_nums = config["crop_nums"]
                #     crop_n_times = config["crop_n_times"]
                # end_pos = all_config["end_pos"]
                # start_pos = all_config["start_pos"]
                aug_seqs = []  # copy.deepcopy(instances)
                aug_ts = []  # copy.deepcopy(timestamps)
            # "insert", "crop", "replace", "reorder"
                for session_id,(seq, ts) in enumerate(zip(instances, timestamps)):
                    if self.aug_base in ["Ti-crop", "Ti-reorder"]:
                        chop_ratio = config["crop_ratio"] if 'crop_ratio' in config else config['reorder_ratio']
                        chop_n_times = config["ti_crop_n_times"] if "ti_crop_n_times" in config else config['reorder_n_times']
                        chop_nums = config['crop_nums'] if 'crop_nums' in config else config['reorder_nums']
                        chop_nums = max(chop_nums, int((all_config['end_pos'] + 1 - all_config['start_pos']) * chop_ratio), all_config['ti_threshold'])
                        if len(seq) <= min(chop_nums , chop_n_times + chop_nums - 1):  #对过短的session不做增强
                            aug_seqs.append(seq)
                            aug_ts.append(ts)
                            continue
                    elif self.aug_base == "Ti-replace":
                        if len(seq) < 3:
                            aug_seqs.append(seq)
                            aug_ts.append(ts)
                            continue
                    if self.aug_base == "subset-split":
                        if len(seq) < 2:
                            aug_seqs.append(seq)
                            aug_ts.append(ts)
                            continue
                    config['start_pos'] = all_config['start_pos']
                    config['end_pos'] = all_config['end_pos']
                    config['ti_threshold'] = all_config['ti_threshold']
                    user_seq, user_ts, time_sort = augment_operator.forward(
                        args, seq, ts, **config
                    )
                    for sub_seq in user_seq:
                        aug_seqs.append(sub_seq)
                    for sub_ts in user_ts:
                        aug_ts.append(sub_ts)
                instances = aug_seqs
                timestamps = aug_ts
                del aug_seqs, aug_ts
                result_loc[uid] = instances
                result_time[uid] = timestamps
            else:
                # Generate the baseline test
                self.submit_set.add("baseline")
                result_loc[uid].append(instances)
                result_time[uid].append(timestamps)
        end_time = time.time() 
        print(
            f"Data augmentation finished, elapsed: {round(end_time - start_time, 4)} s.",
            flush=True,
        )
        return result_loc,result_time
        
    def generate_all(self, args, indexes, traj):
        # params = self.params
        all_instances = self.read_inter(args, traj)
        all_timestamps = self.read_time(args, traj)
        indexes_name = [str(index) for index in indexes]
        task_name = "_".join(indexes_name)
        if task_name == "1000_1000":
            self.save_data(args, all_instances, all_timestamps, task_name)
        else:
            for aug_index in indexes:
                aug_item = AUG_LIST[aug_index-1]
                if "_" in aug_item:
                    aug_base = aug_item.split("_")[0]
                    item_sample = aug_item.split("_")[1]
                else:
                    aug_base = aug_item
                    item_sample = "memorybased"
                    # args, aug_base, item_sample, all_instances, all_timestamps, traj
                augment = Augment(args.aug_data_path, aug_base, args.time_sample, item_sample, args.city, args.dataset, args.model, args.da_config, args.da_pa_config_file_path)
                all_instances, all_timestamps = self.generate_da(augment, all_instances, all_timestamps, traj)
            self.save_data(augment, all_instances, all_timestamps, task_name)

    def item_pop(self, instances):
        all_items = []
        for seq in instances:
            all_items += seq
        item_pop_counter = Counter(all_items)
        return item_pop_counter

    @staticmethod        
    def read_inter(args, traj):
        city_name = args.city
        model = args.model
        dataset = args.dataset
        data_path = os.path.join(AUGMENT_DATA_INPUT_PATH, f"{dataset}/{MODEL_TYPE[model]}/{city_name}/loc_list.json")
        with open (data_path,"r") as f:
            instances = json.load(f)
        return instances

    @staticmethod
    def read_time(args, traj):
        data_path = os.path.join(AUGMENT_DATA_INPUT_PATH, f"{args.dataset}/{MODEL_TYPE[args.model]}/{args.city}/time_list.json")
        with open (data_path,"r") as f:
            timestamps = json.load(f)
        return timestamps

    def save_aug_traj(self, args, task_name, root, result_loc, result_time):
        map_data_path = os.path.join(AUGMENT_DATA_INPUT_PATH, f"{args.dataset}/{MODEL_TYPE[args.model]}/{args.city}/map_loc_dict.json")
        result_loc = {int(uid):result_loc[uid] for uid in result_loc.keys()}
        result_time = {int(uid):result_time[uid] for uid in result_time.keys()}
        with open (map_data_path,"r") as f:
            map_loc_dict = json.load(f)
        map_loc_dict = {int(uid):map_loc_dict[uid] for uid in map_loc_dict.keys()}
    # Define variables to store the results for each model
        if MODEL_TYPE[args.model]=="CACSR":       
            cacsr_result = defaultdict(list)
            venue_cnt = len(map_loc_dict.keys())  # Use full dataset for encode
            distance_theta = 1
            gaussian_beta=10
            SS_distance, SS_proximity, SS_gaussian_distance = construct_spatial_matrix_accordingDistance(distance_theta, venue_cnt, map_loc_dict, gaussian_beta)
        results_act = [] 
        results = [] 
        final_result = {}  #LibCity, MainTUL, DPLink
        loc_result = {}
        time_result = {}
        cat_result = {}
        self.semaphore.acquire()
        print(f"{len(result_loc)} sequences augmented for task '{task_name}'")
        self.semaphore.release()
        os.makedirs(args.aug_data_path, exist_ok=True)
        traj_id = 0
        dyna_id = 1
    # Process session-based data into the format required by each model.    
        for uid, sessions in result_loc.items():
            user_sessions = []
            uid = int(uid)
            if uid not in final_result:
                final_result[uid] = []
                loc_result[uid] = []
                cat_result[uid] = []
                time_result[uid] = []
            all_time = result_time[uid]  
            all_location = sessions
            session_st = [time_session[0] for time_session in all_time] # The start time of each session for this user
            if MODEL_TYPE[args.model]=="CACSR":
                all_flat_time = [hours_to_time(item) for ses in all_time for item in ses] # flat-format
                all_delta_times = get_delta(all_flat_time)
            for idx, session in enumerate(sessions):
                final_session = []
                loc_session = []
                time_session = []
                cat_session = []
                traj_id += 1
                # For these models, he same batch of sessions needs to be processed uniformly. 
                if MODEL_TYPE[args.model]=="CACSR": 
                    loc_result[uid].append(session)
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
                    cacsr_format_session(SS_gaussian_distance, all_session_based_loc_his, session_based_loc_hct, session_based_hour_hct, history_delta_hour, cacsr_result, int(uid), 'train')      
                else: 
                    # For these models, each session needs to be processed seperately         
                    for idx2, traj_point in enumerate(session):
                        dyna_id += 1
                        now_time = all_time[idx][idx2]  # hour format
                        if MODEL_TYPE[args.model]=="LibCity": 
                            time_point = hours_to_time(now_time)
                            final_session.append([dyna_id,"trajectory",time_point,int(uid),int(traj_point)])
                        elif MODEL_TYPE[args.model] in ["MainTUL","DPLink","S2TUL"]:
                            if int(traj_point) not in map_loc_dict:
                                print("Augmented point not in raw points!!!")
                                continue
                            else:
                                final_session.append([dyna_id,map_loc_dict[int(traj_point)]["cat"],now_time,int(uid),int(traj_point), map_loc_dict[int(traj_point)]["lon"], map_loc_dict[int(traj_point)]["lat"]])  # train.csv中一行含有的所有内容
                                loc_session.append(int(traj_point))
                                time_session.append(now_time)
                                cat_session.append(map_loc_dict[int(traj_point)]["cat"]) 
                        elif MODEL_TYPE[args.model] == 'ActSTD':
                            if int(traj_point) not in map_loc_dict:
                                print("Augmented point not in raw points!!!")
                                continue
                            user_sessions.append([get_traj_time(True, hours_to_time(now_time),MODEL_TYPE[args.model]), map_loc_dict[int(traj_point)]["cat"],[map_loc_dict[int(traj_point)]["lon"], map_loc_dict[int(traj_point)]["lat"]]])
                            
                        elif MODEL_TYPE[args.model]=="GETNext":   
                            # "check_ins_id", "POI_catid","UTC_time","user_id","POI_id","longitude","latitude","POI_catname", "trajectory_id_raw","SplitTag" 
                            if int(traj_point) not in map_loc_dict:
                                print("Augmented point not in raw points!!!")
                                continue 
                            else:
                                results.append([dyna_id,map_loc_dict[int(traj_point)]["cat"],hours_to_time(now_time),int(uid),int(traj_point), map_loc_dict[int(traj_point)]["lon"], map_loc_dict[int(traj_point)]["lat"],map_loc_dict[int(traj_point)]["cat_name"], traj_id,"train"])
                        else:
                            print("Not support model type!!")         
                    final_result[uid].append(final_session)  #get_rencoded
                    loc_result[uid].append(loc_session)
                    time_result[uid].append(time_session)
                    cat_result[uid].append(cat_session)
            results_act.append(user_sessions)
       # Save the processed result for each model   '%Y-%m-%dT%H:%M:%SZ'          
        if  MODEL_TYPE[args.model]=="GETNext":
            results = pd.DataFrame(results, columns=["check_ins_id", "POI_catid","UTC_time","user_id","POI_id","longitude","latitude","POI_catname", "trajectory_id_raw","SplitTag"])
            train_part = getnext_format(args.city, args.model, results)
            val_part = pd.read_csv(os.path.join(args.aug_data_path,"1000_1000_val.csv"))
            test_part = pd.read_csv(os.path.join(args.aug_data_path,"1000_1000_test.csv"))
            # 获取 train_part 中的所有 trajectory_id_raw（已使用的 ID）
            train_trajectory_ids = set(train_part['trajectory_id_raw'])
            # 为 test_part 和 val_part 中的重复 `trajectory_id_raw` 重新编号
            # reencoder: only trajectory_id_raw
            val_part = reform_test_val(val_part, max(train_trajectory_ids))
            val_trajectory_ids = set(val_part['trajectory_id_raw'])
            test_part = reform_test_val(test_part, max(val_trajectory_ids|train_trajectory_ids))
            all_part = pd.concat([train_part, val_part, test_part], ignore_index=True)
            # reorder: sessions of each user are in time order
            all_part = all_part.sort_values(by=['user_id', 'trajectory_id_raw', 'timezone'], ascending=True) 
            results = getnext_encode(all_part)  
            results.to_csv(os.path.join(args.aug_data_path, f"{task_name}_sample.csv"), index=False)
            results[results['SplitTag'] == 'train'].to_csv(os.path.join(args.aug_data_path, f"{task_name}_train.csv"), index=False)
            results[results['SplitTag'] == 'validation'].to_csv(os.path.join(args.aug_data_path, f"{task_name}_val.csv"), index=False)
            results[results['SplitTag'] == 'test'].to_csv(os.path.join(args.aug_data_path, f"{task_name}_test.csv"), index=False) 
        elif MODEL_TYPE[args.model]=="LibCity":
            with open(os.path.join(args.aug_data_path, f"{task_name}.json"), "w") as f:
                json.dump(final_result, f)
        elif MODEL_TYPE[args.model]=='CACSR':
            train_save_filename = os.path.join(args.aug_data_path ,f'{task_name}_train.npz')
            loc_num = len(map_loc_dict.keys())
            user_lid_freq = get_user_lidFreq(loc_result, loc_num)
            np.savez_compressed(train_save_filename,     
                    trainX_target_lengths=np.array(cacsr_result['trainX_target_lengths']),
                    trainX_arrival_times=np.array(cacsr_result['trainX_arrival_times'],dtype=object),
                    trainX_duration2first=np.array(cacsr_result['trainX_duration2first'],dtype=object),
                    trainX_session_arrival_times=np.array(cacsr_result['trainX_session_arrival_times'],dtype=object),
                    trainX_local_weekdays=np.array(cacsr_result['trainX_local_weekdays'],dtype=object),
                    trainX_session_local_weekdays=np.array(cacsr_result['trainX_session_local_weekdays'],dtype=object),
                    trainX_local_hours=np.array(cacsr_result['trainX_local_hours'],dtype=object),
                    trainX_session_local_hours=np.array(cacsr_result['trainX_session_local_hours'],dtype=object),
                    trainX_local_mins=np.array(cacsr_result['trainX_local_mins'],dtype=object),
                    trainX_session_local_mins=np.array(cacsr_result['trainX_session_local_mins'],dtype=object),
                    trainX_delta_times=np.array(cacsr_result['trainX_delta_times'],dtype=object),
                    trainX_session_delta_times=np.array(cacsr_result['trainX_session_delta_times'],dtype=object),
                    trainX_locations=np.array(cacsr_result['trainX_locations'],dtype=object),
                    trainX_session_locations=np.array(cacsr_result['trainX_session_locations'],dtype=object),
                    trainX_last_distances=np.array(cacsr_result['trainX_last_distances'],dtype=object),
                    trainX_users=np.array(cacsr_result['trainX_users']), trainX_lengths=np.array(cacsr_result['trainX_lengths']),
                    trainX_session_lengths=np.array(cacsr_result['trainX_session_lengths'],dtype=object),
                    trainX_session_num=np.array(cacsr_result['trainX_session_num']),
                    trainY_arrival_times=np.array(cacsr_result['trainY_arrival_times'],dtype=object),
                    trainY_delta_times=np.array(cacsr_result['trainY_delta_times'],dtype=object),
                    trainY_locations=np.array(cacsr_result['trainY_locations'],dtype=object),
                    user_lidfreq=user_lid_freq, 
                    dtype=object)
            
        elif MODEL_TYPE[args.model]=='S2TUL':
            train_res = {}
            train_time_res = {}
            train_list = []
            train_time_list = []
            for user, sessions in loc_result.items():
                train_res[user] = [[int(user)]+ses for ses in sessions]
                train_time_res[user] = result_time[user]
            for user in sorted(train_res.keys()):
                train_list.extend(train_res[user])
                train_time_list.extend(train_time_res[user])

            with open(os.path.join(args.aug_data_path,f"{task_name}_train_trajs.txt"), 'w') as f:
                for row in train_list:
                    line = ' '.join(map(str, row)) + '\n'
                    f.write(line)              

            with open(os.path.join(args.aug_data_path,f"{task_name}_train_trajs_time.txt"), 'w') as f:
                for row in train_time_list:
                    line = ' '.join(map(str, row)) + '\n'
                    f.write(line)
                    
        elif MODEL_TYPE[args.model]=='ActSTD':
            event_cnt = set()
            for k,v in map_loc_dict.items():
                event_cnt.add(v['cat'])
            results_act = {'train':results_act,'num_event':len(event_cnt)}
            with open(os.path.join(args.aug_data_path, f"{task_name}_train.json"), "w") as f:
                json.dump(results_act, f)
        else:
            with open(os.path.join(args.aug_data_path, f"{task_name}_all.json"), "w") as f:
                json.dump(final_result, f)
            with open(os.path.join(args.aug_data_path, f"{task_name}_loc.json"), "w") as f:
                json.dump(loc_result, f)
            with open(os.path.join(args.aug_data_path, f"{task_name}_time.json"), "w") as f:
                json.dump(time_result, f)
            with open(os.path.join(args.aug_data_path, f"{task_name}_cat.json"), "w") as f:
                json.dump(cat_result, f)
    def save_data(self, args, sequences, timestamps, task_name):
        self.save_aug_traj(
            args=args,
            task_name = task_name,
            root=self.output_path,
            result_loc=sequences,
            result_time=timestamps, 
            )

def save_pa_da_config(args, indexes, result_dict):
    uuid_file = os.path.join(args.da_pa_config_file_path,"uuid.json")
    flag = ''.join(str(uuid.uuid4()).split('-'))
    result_config_file = os.path.join(args.da_pa_config_file_path,f"result_config_{flag}.json")
    result_config = {}
    for index in indexes:
        aug_method = AUG_LIST[int(index)-1]
        if "_" in aug_method:
            aug_index = aug_method.split("_")[0]
        else:
            aug_index = aug_method
        operater = OPERATOR_DICT[aug_index]
        config_file = os.path.join(args.da_pa_config_file_path,f"{operater}.yaml")
        result_config[aug_index] = result_dict[str(index)]
        # uuid_file = os.path.join(args.da_pa_config_file_path,"uuid.json")
        # flag = ''.join(str(uuid.uuid4()).split('-'))
        # Use UUID to distinguish operator parameter configuration files generated in different rounds.
        with open(uuid_file,"w") as f:
            json.dump(flag, f)
        with open(config_file, 'w') as f:
            yaml.dump(result_dict[str(index)], f, default_flow_style=False)
    with open(result_config_file,"w") as f:
        json.dump(result_config, f)
            
 # generate_all(self, args, indexes, traj, params):    
def train_model(args, indexes):
    file_exists = False
    aug_data_path = args.aug_data_path
    file_path = args.result_path
    indexes_name = [str(index) for index in indexes]
    aug_methods_name = '_'.join(indexes_name)
    aug_methods_file = f"{aug_methods_name}.json"
    for aug_file in os.listdir(aug_data_path):
        if aug_methods_file==aug_file:
            file_exists = True
            break
    filename_LLM = f"{args.model}_{args.dataset}_{args.city}_{aug_methods_name}_epoch_{args.max_epoch}_step_{args.max_step}.json"
    base_model_file = os.path.join(BASE_MODEL_PATH,f"{args.model}.sh")
    if not args.pa_da:
        if not file_exists:
            da_module = ParallelDA(aug_data_path, args.da_config)
            da_module.generate_all(
                args=args,
                indexes=indexes,
                traj=True
            )
        if filename_LLM not in os.listdir(file_path):
            subprocess.call(['sh', base_model_file, str(args.gpu_id), str(aug_methods_name), str(aug_data_path), args.model, str(args.max_epoch), str(args.result_path), str(args.dataset), str(args.city), str(args.task), str(args.max_step)])
    else:
        da_module = ParallelDA(aug_data_path, args.da_config)
        da_module.generate_all(
            args=args,
            indexes=indexes,
            traj=True
        )
        uuid_file = os.path.join(args.da_pa_config_file_path, "uuid.json")
        if os.path.exists(uuid_file):
            with open(uuid_file, "r") as f:
                flag = json.load(f)
            subprocess.call(['sh', base_model_file, str(args.gpu_id), str(aug_methods_name), str(aug_data_path), args.model, str(args.max_epoch), str(args.result_path), str(args.dataset), str(args.city), str(args.task), str(args.max_step)])
            result_file = os.path.join(file_path, filename_LLM)
            os.rename(result_file,os.path.join(file_path,f"{args.model}_{args.dataset}_{args.city}_{aug_methods_name}_epoch_{args.max_epoch}_step_{args.max_step}_{flag}.json"))
        else:
            if filename_LLM not in os.listdir(file_path):
                subprocess.call(['sh', base_model_file, str(args.gpu_id), str(aug_methods_name), str(aug_data_path), args.model, str(args.max_epoch), str(args.result_path), str(args.dataset), str(args.city), str(args.task), str(args.max_step)])

def get_model_result(args, indexes):
    indexes_name = [str(index) for index in indexes]
    aug_methods_name = "_".join(indexes_name)
    filename_LLM = f"{args.model}_{args.dataset}_{args.city}_{aug_methods_name}_epoch_{args.max_epoch}_step_{args.max_step}.json"
    if not args.pa_da:
        with open(os.path.join(args.result_path, filename_LLM),"r") as f:
            result = json.load(f)
    else:
        uuid_file = os.path.join(args.da_pa_config_file_path, "uuid.json")
        if os.path.exists(uuid_file):
            with open(uuid_file, "r") as f:
                flag = json.load(f)
            filename_LLM = f"{args.model}_{args.dataset}_{args.city}_{aug_methods_name}_epoch_{args.max_epoch}_step_{args.max_step}_{flag}.json"
            with open(os.path.join(args.result_path, filename_LLM),"r") as f:
                result = json.load(f)
        else:
            print("This is the first trial!!")
            filename_LLM = f"{args.model}_{args.dataset}_{args.city}_{aug_methods_name}_epoch_{args.max_epoch}_step_{args.max_step}.json"
            with open(os.path.join(args.result_path, filename_LLM),"r") as f:
                result = json.load(f)
            
    return result[EVALUATE_METRIC[MODEL_TYPE[args.model]]]


def copy_allfiles(src,dest):
#src:原文件夹；dest:目标文件夹
  src_files = os.listdir(src)
  for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest)

        
    