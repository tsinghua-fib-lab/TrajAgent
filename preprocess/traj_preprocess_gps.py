import pandas as pd
from geopy.distance import geodesic
from collections import defaultdict
import argparse
import numpy as np
import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from UniEnv.etc.settings import *
from preprocess.utils import save_df_to_json_lines

def preprocessing4TR(model_name, data_name):
    min_len = 100
    max_len = 150
    data_size = 20000
    mask_ratio = 0.5

    # 加载数据
    df = pd.read_csv(os.path.join(PROCESS_DATA_INPUT_PATH, data_name, f'{data_name}.csv'), index_col=None,
                     usecols=['POLYLINE', 'NUM_POINT'])
    df = df[(df['NUM_POINT'] >= min_len) & (df['NUM_POINT'] <= max_len)]
    df = df.sort_values(by='NUM_POINT', ascending=False)
    df = df.iloc[:data_size]
    df['POLYLINE'] = df['POLYLINE'].map(eval)

    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"train set size: {len(train_df)}, test set size: {len(test_df)}")

    # 数据处理函数
    def process_data(data):
        trajectories = []
        attention_masks = []
        miss_masks = []
        miss_labels = []

        for polyline in data['POLYLINE']:
            traj = np.array(polyline)
            traj = np.round(traj, decimals=6)

            # 填充轨迹到 max_len
            padded_traj = np.zeros((max_len, 2))
            traj_len = len(traj)
            if traj_len <= max_len:
                padded_traj[:traj_len] = traj
            else:
                padded_traj = traj[:max_len]

            # 生成 attention_mask
            attn_mask = np.zeros(max_len, dtype=int)
            attn_mask[:min(traj_len, max_len)] = 1

            # 初始化 miss_mask 和 miss_label
            miss_mask = np.zeros(max_len, dtype=int)
            miss_label = np.zeros((max_len, 2))

            # 随机掩码 50% 的点
            if traj_len > 1:
                num_to_mask = max(1, int(mask_ratio * traj_len))
                mask_indices = np.random.choice(np.arange(traj_len), size=num_to_mask, replace=False)
                for idx in mask_indices:
                    miss_mask[idx] = 1
                    miss_label[idx] = padded_traj[idx]
                    padded_traj[idx] = 0

            # 更新 attention_mask
            attn_mask = attn_mask & (1 - miss_mask)

            trajectories.append(padded_traj)
            attention_masks.append(attn_mask)
            miss_masks.append(miss_mask)
            miss_labels.append(miss_label)

        return np.array(trajectories), np.array(attention_masks), np.array(miss_masks), np.array(miss_labels)

    # 处理训练集和测试集
    train_trajectories, train_attention_masks, train_miss_masks, train_miss_labels = process_data(train_df)
    test_trajectories, test_attention_masks, test_miss_masks, test_miss_labels = process_data(test_df)

    # 转换为 (num_samples, 2, max_len)
    train_trajectories = np.transpose(train_trajectories, (0, 2, 1))
    train_miss_labels = np.transpose(train_miss_labels, (0, 2, 1))
    test_trajectories = np.transpose(test_trajectories, (0, 2, 1))
    test_miss_labels = np.transpose(test_miss_labels, (0, 2, 1))

    # # 计算 Z-Score 参数
    # traj_mean = train_trajectories.mean(axis=(0, 2))  # 计算经纬度的均值 (2,)
    # traj_std = train_trajectories.std(axis=(0, 2))    # 计算经纬度的标准差 (2,)
    #
    # # 对轨迹数据进行 Z-Score 标准化
    # train_trajectories_normalized = z_score_normalize(train_trajectories, traj_mean, traj_std)
    # test_trajectories_normalized = z_score_normalize(test_trajectories, traj_mean, traj_std)

    # 打印结果信息
    print(f"训练集处理后: 轨迹 shape={train_trajectories.shape}, "
          f"attn_mask shape={train_attention_masks.shape}, "
          f"miss_mask shape={train_miss_masks.shape}, "
          f"miss_label shape={train_miss_labels.shape}")
    print(f"测试集处理后: 轨迹 shape={test_trajectories.shape}, "
          f"attn_mask shape={test_attention_masks.shape}, "
          f"miss_mask shape={test_miss_masks.shape}, "
          f"miss_label shape={test_miss_labels.shape}")

    # 保存处理后的数据
    np.savez_compressed(os.path.join(PROCESS_DATA_INPUT_PATH,data_name,f'TrajBERT/{model_name}_{data_name}_train.npz'),
                        traj=train_trajectories,
                        attn_mask=train_attention_masks,
                        miss_mask=train_miss_masks,
                        miss_label=train_miss_labels)
    np.savez_compressed(os.path.join(PROCESS_DATA_INPUT_PATH,data_name,f'TrajBERT/{model_name}_{data_name}_test.npz'),
                        traj=test_trajectories,
                        attn_mask=test_attention_masks,
                        miss_mask=test_miss_masks,
                        miss_label=test_miss_labels)
    print("数据已保存")

def preprocessing4TTE(model_name, data_name):
    DATA_SIZE=20000
    min_len=10
    if data_name == 'porto':
        # 读取数据
        df = pd.read_csv(os.path.join(PROCESS_DATA_INPUT_PATH, data_name, f'{data_name}.csv'), index_col=None,
                         usecols=['TAXI_ID', 'TIMESTAMP', 'NUM_POINT', 'POLYLINE'], nrows=100000)

        # 数据清洗
        df = df[(df['TIMESTAMP'] != "") & (df['POLYLINE'] != "[]") & (df['POLYLINE'] != "")]
        df.dropna(axis=0, how='any', inplace=True)

        # 处理 POLYLINE 列
        df['POLYLINE'] = df['POLYLINE'].swifter.apply(eval)
        df = df[df['NUM_POINT'] >= min_len]
        df = df.iloc[:DATA_SIZE]
        # 按时间戳排序并重置索引
        df.sort_values(by='TIMESTAMP', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 提取时间信息
        df['DATETIME'] = pd.to_datetime(df['TIMESTAMP'], unit='s', utc=True)
        df['dateID'] = df['DATETIME'].dt.day - 1  # 0-indexed day
        df['weekID'] = df['DATETIME'].dt.weekday  # 0 for Monday, 6 for Sunday
        df['timeID'] = df['DATETIME'].dt.hour * 60 + df['DATETIME'].dt.minute

        # 处理 lngs 和 lats 列
        df['lngs'] = df['POLYLINE'].swifter.apply(lambda x: [point[0] for point in x])
        df['lats'] = df['POLYLINE'].swifter.apply(lambda x: [point[1] for point in x])

        # 计算 dist_gap 和 dist
        def calculate_distance(row):
            coords = np.column_stack((row['lats'], row['lngs']))
            dist_gap = [geodesic(coords[0], coords[i]).km for i in range(len(coords))]
            return pd.Series([dist_gap, dist_gap[-1]])

        df[['dist_gap', 'dist']] = df.swifter.apply(calculate_distance, axis=1)

        # 计算 time_gap 和总时间 time
        df['time_gap'] = df['NUM_POINT'].swifter.apply(lambda x: np.linspace(0, (x - 1) * 15 / 60, num=x).tolist())
        df['time'] = df['time_gap'].swifter.apply(lambda x: x[-1])

        # 重命名和删除不需要的列
        df['TAXI_ID'], _ = pd.factorize(df['TAXI_ID'])
        df.rename(columns={'TAXI_ID': 'driverID'}, inplace=True)
        df.drop(columns=['TIMESTAMP', 'POLYLINE', 'DATETIME', 'NUM_POINT'], inplace=True)

    elif data_name == 'chengdu':
        df = pd.read_csv(os.path.join(PROCESS_DATA_INPUT_PATH, data_name, f'{data_name}.csv'), index_col=None, nrows=100000)
        df.dropna(axis=0, how='any', inplace=True)
        df['POLYLINE'] = df['POLYLINE'].swifter.apply(eval)
        df['TIMESTAMPS'] = df['TIMESTAMPS'].swifter.apply(eval)
        df = df[df['NUM_POINT'] >= min_len]
        df = df.iloc[:DATA_SIZE]
        df.sort_values(by='TIMESTAMP', inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['DATETIME'] = pd.to_datetime(df['TIMESTAMP'], unit='s', utc=True)
        df['dateID'] = df['DATETIME'].dt.day - 1  # 0-indexed day
        df['weekID'] = df['DATETIME'].dt.weekday  # 0 for Monday, 6 for Sunday
        df['timeID'] = df['DATETIME'].dt.hour * 60 + df['DATETIME'].dt.minute

        df['lngs'] = df['POLYLINE'].swifter.apply(lambda x: [point[0] for point in x])
        df['lats'] = df['POLYLINE'].swifter.apply(lambda x: [point[1] for point in x])

        # 计算 dist_gap 和 dist
        def calculate_distance(row):
            coords = np.column_stack((row['lats'], row['lngs']))
            dist_gap = [geodesic(coords[0], coords[i]).km for i in range(len(coords))]
            return pd.Series([dist_gap, dist_gap[-1]])

        df[['dist_gap', 'dist']] = df.swifter.apply(calculate_distance, axis=1)

        # 计算 time_gap 和总时间 time
        df['time_gap'] = df['TIMESTAMPS'].swifter.apply(lambda x: [(e - x[0]) / 60. for e in x])
        df['time'] = df['time_gap'].swifter.apply(lambda x: x[-1])

        # 重命名和删除不需要的列
        df['TAXI_ID'], _ = pd.factorize(df['TAXI_ID'])
        df.rename(columns={'TAXI_ID': 'driverID'}, inplace=True)
        df.drop(columns=['TIMESTAMP', 'POLYLINE', 'DATETIME', 'NUM_POINT', 'TIMESTAMPS'], inplace=True)

    num_samples = df.shape[0]
    sp0 = int(num_samples * 0.7)
    sp1 = int(num_samples * 0.2)
    save_df_to_json_lines(df.iloc[:sp0], os.path.join(PROCESS_DATA_INPUT_PATH,data_name,model_name,f'{data_name}_{model_name}_train.txt'))
    save_df_to_json_lines(df.iloc[sp0:sp0 + sp1], os.path.join(PROCESS_DATA_INPUT_PATH,data_name,model_name,f'{data_name}_{model_name}_val.txt'))
    save_df_to_json_lines(df.iloc[sp0 + sp1:], os.path.join(PROCESS_DATA_INPUT_PATH,data_name,model_name,f'{data_name}_{model_name}_test.txt'))

    # 计算统计信息
    config = {
        "driverID_max": int(df['driverID'].max()),
        "weekID_max": 6,
        "timeID_max": 1439,
        "dist_gap_mean": np.mean([np.mean(d) for d in df['dist_gap']]),
        "dist_gap_std": np.std([np.std(d) for d in df['dist_gap']]),
        "time_gap_mean": np.mean([np.mean(t) for t in df['time_gap']]),
        "time_gap_std": np.std([np.std(t) for t in df['time_gap']]),
        "lngs_mean": np.mean(np.concatenate(df['lngs'].values)),
        "lngs_std": np.std(np.concatenate(df['lngs'].values)),
        "lats_mean": np.mean(np.concatenate(df['lats'].values)),
        "lats_std": np.std(np.concatenate(df['lats'].values)),
        "dist_mean": df['dist'].mean(),
        "dist_std": df['dist'].std(),
        "time_mean": df['time'].mean(),
        "time_std": df['time'].std(),
        "train_set": [f"{data_name}_{model_name}_train.txt"],
        "eval_set": [f"{data_name}_{model_name}_val.txt"],
        "test_set": [f"{data_name}_{model_name}_test.txt"]
    }
    # 保存到 config.json 文件
    # with open(f'tasks/travel time estimation/{model_name}/{data_name}_config.json', 'w', encoding='utf-8') as f:
    #     json.dump(config, f, ensure_ascii=False, indent=4)
    print(config)
    print(f"====> Preprocessing Finished (Model: {model_name}, Dataset: {data_name}).")


def preprocessing4AD(model_name, data_name):
    output_path = os.path.join(PROCESS_DATA_OUTPUT_PATH, MODEL_TYPE[model_name])
    train_file_path = os.path.join(output_path, f'{data_name}_{model_name}_train.csv')
    val_file_path = os.path.join(output_path, f'{data_name}_{model_name}_val.csv')
    if data_name == 'porto':
        # 配置参数
        grid_height, grid_width = 0.1, 0.1
        boundary = {'min_lat': 41.140092, 'max_lat': 41.185969, 'min_lng': -8.690261, 'max_lng': -8.549155}
        shortest, longest = 20, 1200
        min_sd_traj_num = 25
        test_traj_num = 5

        # 辅助函数
        def height2lat(height):
            return height / 110.574

        def width2lng(width):
            return width / 111.320 / 0.99974

        def in_boundary(lat, lng, b):
            return b['min_lng'] < lng < b['max_lng'] and b['min_lat'] < lat < b['max_lat']

        # 预计算网格大小
        lat_size = height2lat(grid_height)
        lng_size = width2lng(grid_width)
        lat_grid_num = int((boundary['max_lat'] - boundary['min_lat']) / lat_size) + 1
        lng_grid_num = int((boundary['max_lng'] - boundary['min_lng']) / lng_size) + 1
        print(lat_grid_num, lng_grid_num)

        # 读取数据集
        df = pd.read_csv(os.path.join(PROCESS_DATA_INPUT_PATH, data_name, f'{data_name}.csv'), header=0, index_col="TRIP_ID")
        print(df.shape)
        sd_cnt = defaultdict(list)

        # 处理轨迹数据
        total_traj_num = len(df)
        for i, (idx, traj) in enumerate(df.iterrows()):
            if i % 10000 == 0:
                print(f"Complete: {i}; Total: {total_traj_num}")

            grid_seq = []
            valid = True
            polyline = json.loads(traj["POLYLINE"])  # Use json.loads for better safety

            if shortest <= len(polyline) <= longest:
                for lng, lat in polyline:
                    if in_boundary(lat, lng, boundary):
                        grid_i = int((lat - boundary['min_lat']) / lat_size)
                        grid_j = int((lng - boundary['min_lng']) / lng_size)
                        grid_seq.append(grid_i * lng_grid_num + grid_j)
                    else:
                        valid = False
                        break

                if valid:
                    # Append trajectory to the corresponding start-destination group
                    s, d = grid_seq[0], grid_seq[-1]
                    sd_cnt[(s, d)].append(json.dumps(grid_seq))

        # Split into train and validation sets and save to files
        with open(train_file_path, 'w') as fout_train, open(val_file_path, 'w') as fout_val:
            for trajs in sd_cnt.values():
                if len(trajs) >= min_sd_traj_num:
                    train_trajs = trajs[:-test_traj_num]
                    val_trajs = trajs[-test_traj_num:]
                    fout_train.write("\n".join(train_trajs) + "\n")
                    fout_val.write("\n".join(val_trajs) + "\n")

        print(f"====> Preprocessing Finished (Model: {model_name}, Dataset: {data_name}). "
              f"Train data saved to {train_file_path}, validation data saved to {val_file_path}.")
    elif data_name == 'chengdu':
        # 30.65294 30.72775 104.04214 104.12958
        # [30.5930, 103.9746, 30.75, 104.167]
        # 配置参数
        grid_height, grid_width = 0.1, 0.1
        boundary = {'min_lat': 30.65294, 'max_lat': 30.72775, 'min_lng': 104.04214, 'max_lng': 104.12958}
        shortest, longest = 20, 1200
        min_sd_traj_num = 25
        test_traj_num = 5

        # 辅助函数
        def height2lat(height):
            return height / 110.574

        def width2lng(width):
            return width / 111.320 / 0.99974

        def in_boundary(lat, lng, b):
            return b['min_lng'] < lng < b['max_lng'] and b['min_lat'] < lat < b['max_lat']

        # 预计算网格大小
        lat_size = height2lat(grid_height)
        lng_size = width2lng(grid_width)
        lat_grid_num = int((boundary['max_lat'] - boundary['min_lat']) / lat_size) + 1
        lng_grid_num = int((boundary['max_lng'] - boundary['min_lng']) / lng_size) + 1
        print(lat_grid_num, lng_grid_num)

        # 读取数据集
        df = pd.read_csv(os.path.join(PROCESS_DATA_INPUT_PATH, data_name, f'{data_name}.csv'), header=0, index_col=None)
        print(df.shape)
        sd_cnt = defaultdict(list)

        # 处理轨迹数据
        total_traj_num = len(df)
        for i, (idx, traj) in enumerate(df.iterrows()):
            if i % 10000 == 0:
                print(f"Complete: {i}; Total: {total_traj_num}")

            grid_seq = []
            valid = True
            POLYLINE = traj["POLYLINE"].replace('(', '[').replace(')', ']')

            polyline = json.loads(POLYLINE)  # Use json.loads for better safety

            if shortest <= len(polyline) <= longest:
                for lng, lat in polyline:
                    if in_boundary(lat, lng, boundary):
                        grid_i = int((lat - boundary['min_lat']) / lat_size)
                        grid_j = int((lng - boundary['min_lng']) / lng_size)
                        grid_seq.append(grid_i * lng_grid_num + grid_j)
                    else:
                        valid = False
                        break

                if valid:
                    # Append trajectory to the corresponding start-destination group
                    s, d = grid_seq[0], grid_seq[-1]
                    sd_cnt[(s, d)].append(json.dumps(grid_seq))

        # Split into train and validation sets and save to files
        with open(train_file_path, 'w') as fout_train, open(val_file_path, 'w') as fout_val:
            for trajs in sd_cnt.values():
                if len(trajs) >= min_sd_traj_num:
                    train_trajs = trajs[:-test_traj_num]
                    val_trajs = trajs[-test_traj_num:]
                    fout_train.write("\n".join(train_trajs) + "\n")
                    fout_val.write("\n".join(val_trajs) + "\n")

        print(f"====> Preprocessing Finished (Model: {model_name}, Dataset: {data_name}). "
              f"Train data saved to {train_file_path}, validation data saved to {val_file_path}.")



def pre_main_gps(task, model, dataset, base_model, memory_length, city, model_config, data_config):
    #     # pre_main(base_model=args.base_model, memory_length=args.memory_length, city=args.city, dataset=args.dataset, model=args.model, model_config=model_config, data_config=data_config)
    eval(f"preprocessing4{task}")(model, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", choices=["chengdu", "porto"])
    parser.add_argument("--task", type=str)
    parser.add_argument("--base_model", type=str, default="llama3-70b")
    parser.add_argument("--memory_length", type=int, default="1")
    args = parser.parse_args()

    # model_list = ['DeepTTE', 'GMVSAE']
    # model_str = model_list[1]
    # data_str = 'porto'  # or 'chengdu'
    pre_main_gps(args.task, args.model, args.dataset)
