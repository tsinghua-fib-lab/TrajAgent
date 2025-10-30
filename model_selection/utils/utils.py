import os
import json
import subprocess
from UniEnv.etc.settings import *


def get_model_description(model_dict):
    model_des = ""
    index = 0
    for model, des in model_dict.items():
        index += 1
        model_des += "Model{idx}:\n".format(idx=index) + model 
        model_des +=  "\n{description}:\n".format(description=des)
    # for data, des in data_dict.items():
    #     data_des += "Dataset:" + data + "\ndescription:" + des                                                                                                                                                                                                                                                                                                                                                                                               
    # return model_des, data_des
    return model_des

def get_model_dataset(model_dict, data_dict, llm):
    model_data = {}
    all_datasets = set()
    prompt = """Please select datasets that used in the EXPERIMENTAL RESULTS from the DATASET LIST. Please :
    1.directly output the datasets you choose.
    2.Dataset names you output should be consistent with those in the DATASET LIST.
    3.Seperate the dataset names with ','.
    <EXPERIMENTAL RESULTS>
    {results}
    <DATASET LIST>
    {datasets}
    """
    extract_prompt = """
    
    """
    for data, info in data_dict.items():
        all_datasets.add(data)
    for model, info in model_dict.items():
        if model not in model_data:
            model_data[model] = []
            exp_result = info["Experimental Results"]
            datasets = llm(prompt.format(results=exp_result,datasets=all_datasets))
            tar_datasets = datasets.split(",")
            tar_datasets = [tar_dataset.strip() for tar_dataset in tar_datasets]
            model_data[model].extend(tar_datasets)
    return model_data


def train_model(args, model):  #默认不使用增强策略，即aug_name="1000_1000"
    filename_LLM = f"{model}_{args.dataset}_{args.city}_1000_1000_epoch_{args.max_epoch}.json"
    file_path = args.result_path
    output_path = os.path.join(file_path, args.task)
    input_path = os.path.join(PROCESS_DATA_OUTPUT_PATH, args.dataset, MODEL_TYPE[model], args.city)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    result_path = os.path.join(RESULT_PATH,args.task)
    if filename_LLM not in os.listdir(output_path):
        subprocess.call(['sh', f'base_model/{model}.sh', str(args.gpu_id), "1000_1000", input_path, model, str(args.max_epoch), result_path, args.dataset, args.city, args.task])
def get_model_result(args, model):
    filename_LLM = f"{model}_{args.dataset}_{args.city}_1000_1000_epoch_{args.max_epoch}.json"
    file_path = args.result_path
    output_path = os.path.join(file_path, args.task)
    output_file = os.path.join(output_path,filename_LLM)
    with open(output_file,"r") as f:
        result = json.load(f)
    return result["Recall@5"]
        
    