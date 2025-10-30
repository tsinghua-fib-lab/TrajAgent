import os

# 获取项目根目录（TrajAgent目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据相关路径
# 优先级：环境变量 AUGMOVE_DATA_ROOT > 默认的 ../data（相对于项目根目录）
# 如果没有设置环境变量，默认使用 ../data
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, '..', 'data')
DATA_ROOT = os.getenv('AUGMOVE_DATA_ROOT', DEFAULT_DATA_ROOT)

FORMAT_DATA_PATH = os.path.join(DATA_ROOT, 'dataset', 'input_format')
FORMAT_DATA_OUTPUT_PATH = os.path.join(DATA_ROOT, 'dataset', 'input_format', 'target')
FORMAT_CODE_PATH = os.path.join(PROJECT_ROOT, 'UniEnv')

TRAJ_MID_PATH = os.path.join(DATA_ROOT, 'dataset', 'traj_mid')

PROCESS_DATA_INPUT_PATH = os.path.join(DATA_ROOT, 'dataset', 'input_format')
PROCESS_DATA_OUTPUT_PATH = os.path.join(DATA_ROOT, 'dataset', 'input_timeloc')  # RAW_PRETRAIN_PATH, 原始/增强后的用于模型训练的输入数据

AUGMENT_DATA_INPUT_PATH = os.path.join(DATA_ROOT, 'dataset', 'input_aug')  # 用于增强的数据
# AUGMENT_DATA_OUTPUT_PATH = "UniEnv/dataset/aug_data" # AUGMENTED_PRETRIN_PATH.停用，合并为PROCESS_DATA_OUTPUT_PATH
RESULT_PATH = os.path.join(DATA_ROOT, 'dataset', 'model_output')
DA_CONFIG_FILE = os.path.join(PROJECT_ROOT, "UniEnv", "etc", "da-config.yaml")

OP_MID_DATA_PATH = os.path.join(DATA_ROOT, 'dataset', 'aux')
DA_PA_RESULT_PATH = os.path.join(DATA_ROOT, 'dataset', 'model_output_da_pa')
LLM_RESULT_PATH = os.path.join(DATA_ROOT, 'dataset', 'llm_output_prompt')
DIAL_RESULT_PATH = os.path.join(DATA_ROOT, 'dataset', 'result_dialog')
DA_PA_CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, "UniEnv", "etc", "da_config")

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "UniEnv", "base_model")

BASE_MODEL_PARAM_PATH = os.path.join(PROJECT_ROOT, "UniEnv", "param_optimize", "base_model")

LIMP_DATA_PATH = os.path.join(DATA_ROOT, 'dataset', 'input_format', 'limp')

AUG_LIST = ["Ti-crop","Ti-insert_unvisited", "Ti-insert_memorybased", "Ti-insert_random", "Ti-replace_unvisited","Ti-replace_memorybased","Ti-replace_random","Ti-mask","Ti-reorder","subset-split"]
AUG_METHODS = ["crop", "insert", "replace","reorder","subset-split", "mask"]
MODEL_TYPE ={"LLMMove":"LLM","LLMZS":"LLM","LIMP":"LLM","LLMMob":"LLM","GRU":"LibCity","DeepMove":"LibCity","FPMC":"LibCity","LSTPM":"LibCity","RNN":"LibCity","MainTUL":"MainTUL","DPLink":"DPLink", "DeepTTE": "DeepTTE", "GMVSAE": "GMVSAE","GETNext": "GETNext","CACSR": "CACSR", "S2TUL": "S2TUL","TrajBERT":"TrajBERT",'ActSTD':'ActSTD',"GraphMM":"GraphMM","DeepMM":"DeepMM","DSTPP":"DSTPP","MulTTTE":"MulTTTE", "TrajCL":"TrajCL","DutyTTE":"DutyTTE"}
DATA_TYPE={"gowalla":"checkin","tencent":"map","chengdu":"gps","porto":"gps","foursquare":"checkin","brightkite":"checkin", "agentmove":"checkin","Earthquake":"time_series"}
EVALUATE_METRIC = {'LibCity':'Recall@5','GETNext':'Acc@5','MainTUL':'Acc@5','DPLink':'Hit_32', 'TrajBERT':'MAE','DeepTTE':'MAE', 'GMVSAE':'AUC', 'CACSR':'Acc@5','S2TUL':'Acc@5','GraphMM':'test_avg_acc','DeepMM':'test_avg_acc',"DSTPP":"MAE","MulTTTE":"MAE","TrajCL":"Hit@5", "LLM": "Acc@5","DutyTTE":"MAE"}
BASE_MODEL={"Next_Location_Prediction":"RNN","Trajectory_User_Linkage":None}


PARAM_OP_RESULT_PATH = os.path.join(DATA_ROOT, 'dataset', 'model_output_pa')

PARAM_CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, "UniEnv", "etc", "model_config")



MEANING_OF_OPERATORS="""
1.Ti-crop: For each user, filter out sessions with fewer trajectory points than ti_threshold + ti_crop_n_times. In each session, randomly extract several segments (ti_crop_n_times) of equal length (ti_threshold). Sample data segments according to the rules.
2.Ti-insert_unvisited: For each session, insert selected elements at specified positions. For each insertion position, also insert a new timestamp in augmented_ts, calculated as the average of the timestamps at the current and next positions to simulate the time of the newly inserted element. Insert unvisited POIs by the user, which are POIs not appearing in any POIs involved across all sessions of the user.
3.Ti-insert_memorybased: For each session, insert selected elements at specified positions. For each insertion position, also insert a new timestamp in augmented_ts. Choose the POI with the highest similarity to the target location POI for insertion. The similarity is calculated based on the item-to-item collaborative filtering matrix, where items that appear together more frequently have higher similarity.
4.Ti-insert_random: For each session, insert selected elements at specified positions. For each insertion position, also insert a new timestamp in augmented_ts. Randomly select elements from all POIs for insertion.
5.Ti-replace_unvisited: For each session, use the mask method to sample replace_nums mask positions, and sample POIs to replace elements at mask positions using the specified POI sampling method. Insert unvisited POIs by the user.
6.Ti-replace_memorybased: For each session, use the mask method to sample replace_nums mask positions, and sample POIs to replace elements at mask positions using the specified POI sampling method. Choose the POI with the highest similarity to the target location POI for insertion.
7.Ti-replace_random: For each session, use the mask method to sample replace_nums mask positions, and sample POIs to replace elements at mask positions using the specified POI sampling method. Randomly select elements from all POIs for insertion.
8.Ti-mask: For each session, sample mask_nums mask positions within [start_pos, end_pos], replacing the elements at each mask position with a specified value (0).
9.Ti-reorder: For each user, filter out sessions with fewer trajectory points than ti_threshold + ti_crop_n_times. Using the crop method, cut out segments of length sub_seq_length from the sequence, shuffle the order of items within the segment, and replace them back into the original sequence.
10.Subset-split: For each session, within [start_time, end_time], drop some trajectory points with a probability of dropout_prob, and concatenate the remaining parts of the session with the dropped parts.
"""

OPERATOR_DICT = {
    "Ti-insert": "Inserter",
    "Ti-crop": "Croper",
    "Ti-mask": "Masker",
    "Ti-reorder": "Reorderer",
    "Ti-replace": "Replacer",
    "subset-split": "SubsetSplit",
}


PARAM_OF_OPERATORS="""
1.Ti-crop: 
{{
crop_nums:(int) the size of cropping. default is 2.
crop_ratio:(float) the ratio of cropping. default is 0.
crop_n_times:(int) the number of cropping for each sequence for default setting. default is 2.
crop_time_sort:(str, choice in ["maximum", "minimum"]) choose the candidate subsequence in a descending/ascending order according to its time interval variance. default is 'minimum'.
ti_crop_n_times:(int) the number of cropping for each sequence for time-based setting. default is 3.
}}
2.Ti-insert_unvisited: 
{{
insert_nums:(int) the number of inserted items. default is 1.
insert_ratio:(float) the ratio of inserted items. default is 0.
percent_no_augment:(float) the length of no augmentation at the end of the sequence. default is 0.
insert_time_sort:(str, choice in ["maximum", "minimum"]) choose the insert position that has largest/smallest time interval. default is "maximum".
ti_insert_n_times:(int) the number of insertion for each sequence. default is 1.
}}
3.Ti-insert_memorybased: 
{{
insert_nums:(int) the number of inserted items. default is 1.
insert_ratio:(float) the ratio of inserted items. default is 0.
percent_no_augment:(float) the length of no augmentation at the end of the sequence. default is 0.
insert_time_sort:(str, choice in ["maximum", "minimum"]) choose the insert position that has largest/smallest time interval. default is "maximum".
ti_insert_n_times:(int) the number of insertion for each sequence. default is 1.
}}
4.Ti-insert_random:
{{
insert_nums:(int) the number of inserted items. default is 1.
insert_ratio:(float) the ratio of inserted items. default is 0.
percent_no_augment:(float) the length of no augmentation at the end of the sequence. default is 0.
insert_time_sort:(str, choice in ["maximum", "minimum"]) choose the insert position that has largest/smallest time interval. default is "maximum".
ti_insert_n_times:(int) the number of insertion for each sequence. default is 1.
}}
5.Ti-replace_unvisited: 
{{
replace_nums:(int) the number of replaced items. default is 1.
replace_ratio:(float) the ratio of replaced items. default is 0.
replace_time_sort:(str, choice in ["maximum", "minimum"]) see 'mask_time_sort'. default is 'minimum'.
ti_replace_n_times:(int) the number of replacement for each sequence. default is 1.
}}
6.Ti-replace_memorybased:
{{
replace_nums:(int) the number of replaced items. default is 1.
replace_ratio:(float) the ratio of replaced items. default is 0.
replace_time_sort:(str, choice in ["maximum", "minimum"]) see 'mask_time_sort'. default is 'minimum'.
ti_replace_n_times:(int) the number of replacement for each sequence. default is 1.
}}
7.Ti-replace_random:
{{
replace_nums:(int) the number of replaced items. default is 1.
replace_ratio:(float) the ratio of replaced items. default is 0.
replace_time_sort:(str, choice in ["maximum", "minimum"]) see 'mask_time_sort'. default is 'minimum'.
ti_replace_n_times:(int) the number of replacement for each sequence. default is 1.
}}
8.Ti-mask:
{{
mask_nums: (int) the number of masked items.default is 1.
mask_ratio: (float) the ratio of masked items.default is 0.
mask_value: (int) the value of masked items.default is 1. Do not use 0.
mask_time_sort: (str, choice in ["maximum", "minimum"]) choose the mask position that has largest/smallest time interval.default is 'minimum'.
ti_mask_n_times:(int) the number of mask for each sequence.default is 1.
}}
9.Ti-reorder: 
{{
reorder_nums:(int) the number of reordered items. default is 2.
reorder_ratio:(float) the ratio of reordered items. default is 0.
reorder_n_times: (int) the number of reorder for each sequence. default is 2.
sub_seq_length:(int) the length of the cropped subsequence. default is 5.
reorder_time_sort: (str, choice in ["maximum", "minimum"]) see 'crop_time_sort'. default is 'minimum'.
}}
10.Subset-split:
{{
## Subset split
subset_split_n_times:(int) the number of subset split for each sequence.default is 2.
dropout_prob:(float) the probability that interaction is being dropped out.default is 0.25.
}}
"""

OFFSET_DICT = {'NewYork':-240, 'Moscow':180, 'SaoPaulo':-180, 'Shanghai':480, 'Shanghai_ISP':480, 'Shanghai_Weibo':480}


