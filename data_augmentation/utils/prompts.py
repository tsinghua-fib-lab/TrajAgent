from UniEnv.etc.settings import AUG_LIST, MEANING_OF_OPERATORS, PARAM_OF_OPERATORS
aug_dict = {i + 1: AUG_LIST[i] for i in range(len(AUG_LIST))}
QUESTION = f"""
<TASK>
Please select proper augmentation methods and use them in proper order to jointly augment the input temporal data and user data.Please adjust the selection and combination sequence of operators based on MEANING OF OPERATORS, CHARACTERISTICS OF INPUT DATA, and MEMORY to get a high score.You should solve the task with interleaving Thought, Action and Observation steps. 
<CHARACTERISTICS OF INPUT DATA>
The input temporal data contains a time dictionary(key is the user ID,the value is a list containing all time points when the user is active in chronological order) , the input user data contains a user dictionary(key is the user ID,the value is a list containing all items that the user interacts with in chronological order).
<MEANING OF OPERATORS>
{MEANING_OF_OPERATORS}
"""
THOUGHT = """
In Thought step,you should reason how to choose the combination of operators to get a higher score.Please consider following aspects:
1.The meaning of each operator,and the impact of adding or removing operators at a specific location on the score.
2.Operators that often appear together in operator index lists with higher scores in MEMORY.For example,in [1,2,3] and [2,3,5],[2,3] often appear together.
3.The frequent order of operator index lists with higher scores in MEMORY.For example,in [1,2,3] and [2,3,5],3 often appears after 2.
4.The CHARACTERISTICS OF INPUT DATA.
5.Common characteristics of operator index lists with higher scores in MEMORY.
Please use the sentence structure 'Firstly... Then... Lastly'.Let's think step by step.
Thought:\n
"""
ACTION = """
In Action step, you should consider the Thought step in SCRATCHPAD, and give 2 lists.Each list should contain the indices of augmentation methods in {aug_dict}.For example, if you want to first use Ti-crop, secondly use Ti-insert_random, thirdly use Ti-mask, then the list should be '[1,4,8]'.
Please directly output all 2 lists.Separate every two lists with '\n'.For example:[1,2,3]\n[4,5,6]\n[7,2].
Action:\n
"""

PA_QUESTION = f"""
<TASK>
Please:
1.select proper augmentation methods and use them in proper order to jointly augment the input temporal data and user data.Please adjust the selection and combination sequence of operators based on MEANING OF OPERATORS, CHARACTERISTICS OF INPUT DATA, and MEMORY to get a high score.
2.select proper combination of hyperparameters of each augmentation method in CONFIG HYPERPARAMETERS.Adjust the selection, the combination of hyperparameters based on the main function of hyperparameters, the characteristics of the input data, and memory to get a high score. 
You should solve the task with interleaving Thought, Action and Observation steps. 
<CONFIG HYPERPARAMETERS>
{PARAM_OF_OPERATORS}
<CHARACTERISTICS OF INPUT DATA>
The input temporal data contains a time dictionary(key is the user ID,the value is a list containing all time points when the user is active in chronological order) , the input user data contains a user dictionary(key is the user ID,the value is a list containing all items that the user interacts with in chronological order).
<MEANING OF OPERATORS>
{MEANING_OF_OPERATORS}
"""
# 2.Operators that often appear together in operator index lists with higher scores in MEMORY.For example,in [1,2,3] and [2,3,5],[2,3] often appear together.
# 3.The frequent order of operator index lists with higher scores in MEMORY.For example,in [1,2,3] and [2,3,5],3 often appears after 2.
PA_THOUGHT = """
In Thought step,you should reason how to choose the combination of operators and proper combination of hyperparameters of each augmentation method to get a higher score.Please consider following aspects:
1.The meaning of each operator,and the impact of adding or removing operators at a specific location on the score.
4.Common characteristics of operator index lists with higher scores in MEMORY.Avoid using the same index list as MEMORY with scores lower than {threshold}.
5.The meaning of hyperparameters of each operator you select, and how to adjust the hyperparameters based on the characteristics of input data and hyperparameters with high scores in MEMORY.
6.Use a grid search: Perform a grid search over a range of hyperparameters with high scores to find the optimal combination with higher scores.
7. Stop or reverse the adjusting trend if the score is decreasing.
According to above aspects,please first learn experiences from MEMORY, then make plan for the action step.Please use the sentence structure 'Firstly... Then... Lastly'.Let's think step by step.
Thought:\n
"""
PA_ACTION = f"""
In Action step, you should consider the Thought step in SCRATCHPAD, and return a list and a dictionary.
The list should contain the indices of augmentation methods in {aug_dict}.For example, if you want to first use Ti-crop, secondly use Ti-insert_random, thirdly use Ti-mask, then the list should be '[1,4,8]'.
The dict:{{the indice of operator in {aug_dict}:{{hypermeter name:hypermeter value}}}}.The hypermeter name should be the same with the \
corresponding operator hypermeter names in CONFIG HYPERPARAMETERS,and hypermeter values should be the same type as the corresponding operator hypermeter values in CONFIG HYPERPARAMETERS.
Please directly output the list and dictionary.Separate them with '\n'.For example:[1,2]\n{{"1":{{crop_nums: 2,crop_ratio: 0,crop_n_times: 2,crop_time_sort: "minimum",ti_crop_n_times: 3}},\
"2":{{insert_nums: 1,insert_ratio: 0 ,percent_no_augment: 0,insert_n_times: 2,insert_time_sort: "maximum",ti_insert_n_times: 1}}}}.
Action:\n
"""

PA_QUESTION = f"""
<TASK>
Please:
1.select proper augmentation methods and use them in proper order to jointly augment the input temporal data and user data.Please adjust the selection and combination sequence of operators based on MEANING OF OPERATORS, CHARACTERISTICS OF INPUT DATA, and MEMORY to get a high score.
2.select proper combination of hyperparameters of each augmentation method in CONFIG HYPERPARAMETERS.Adjust the selection, the combination of hyperparameters based on the main function of hyperparameters, the characteristics of the input data, and memory to get a high score. 
You should solve the task with interleaving Thought, Action and Observation steps. 
<CONFIG HYPERPARAMETERS>
{PARAM_OF_OPERATORS}
<CHARACTERISTICS OF INPUT DATA>
The input temporal data contains a time dictionary(key is the user ID,the value is a list containing all time points when the user is active in chronological order) , the input user data contains a user dictionary(key is the user ID,the value is a list containing all items that the user interacts with in chronological order).
<MEANING OF OPERATORS>
{MEANING_OF_OPERATORS}
"""


XR_THOUGHT = """
In Thought step,you should reason how to choose the combination of operators and proper combination of hyperparameters of each augmentation method to get a higher score.Please consider following aspects:
1.The meaning of each operator,and the impact of adding or removing operators at a specific location on the score.
6.The meaning of hyperparameters of each operator you select, and how to adjust the hyperparameters based on the characteristics of input data and hyperparameters with high scores in MEMORY.
7.Use a grid search: Perform a grid search over a range of hyperparameters with high scores to find the optimal combination with higher scores.
8. Stop or reverse the adjusting trend if the score is decreasing.
According to above aspects,please make plan for the action step.Please use the sentence structure 'Firstly... Then... Lastly'.Let's think step by step.
Thought:\n
"""
XR_ACTION = f"""
In Action step, you should consider the Thought step in SCRATCHPAD, and return a list and a dictionary.
The list should contain the indices of augmentation methods in {aug_dict}.For example, if you want to first use Ti-crop, secondly use Ti-insert_random, thirdly use Ti-mask, then the list should be '[1,4,8]'.
The dict:{{the indice of operator in {aug_dict}:{{hypermeter name:hypermeter value}}}}.The hypermeter name should be the same with the \
corresponding operator hypermeter names in CONFIG HYPERPARAMETERS,and hypermeter values should be the same type as the corresponding operator hypermeter values in CONFIG HYPERPARAMETERS.
Please directly output the list and dictionary.Separate them with '\n'.For example:[1,2]\n{{"1":{{crop_nums: 2,crop_ratio: 0,crop_n_times: 2,crop_time_sort: "minimum",ti_crop_n_times: 3}},\
"2":{{insert_nums: 1,insert_ratio: 0 ,percent_no_augment: 0,insert_n_times: 2,insert_time_sort: "maximum",ti_insert_n_times: 1}}}}.
Action:\n
"""

XR_QUESTION = f"""
<TASK>
Please:
1.select proper augmentation methods and use them in proper order to jointly augment the input temporal data and user data.Please adjust the selection and combination sequence of operators based on MEANING OF OPERATORS, CHARACTERISTICS OF INPUT DATA to get a high score.
2.select proper combination of hyperparameters of each augmentation method in CONFIG HYPERPARAMETERS.Adjust the selection, the combination of hyperparameters based on the main function of hyperparameters, the characteristics of the input data to get a high score. 
You should solve the task with interleaving Thought, Action and Observation steps. 
<CONFIG HYPERPARAMETERS>
{PARAM_OF_OPERATORS}
<CHARACTERISTICS OF INPUT DATA>
The input temporal data contains a time dictionary(key is the user ID,the value is a list containing all time points when the user is active in chronological order) , the input user data contains a user dictionary(key is the user ID,the value is a list containing all items that the user interacts with in chronological order).
<MEANING OF OPERATORS>
{MEANING_OF_OPERATORS}
"""