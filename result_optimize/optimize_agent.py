import os
import time
import json
import numpy as np
import multiprocessing as mp
from typing import List, Any
from model_selection.utils.utils import train_model, get_model_result, get_model_description
from model_selection.utils.prompts import MODEL_DESCRIPTION
from UniEnv.etc.settings import *
from data_augmentation.utils.base_llm import LLMWrapper
import re

class OptimizeAgent:
    def __init__(self,
                 args: Any,
                 max_steps:int,
                 output_llm: LLMWrapper = LLMWrapper(
                                            temperature=0,
                                            model_name="llama3-70b",
                                            max_tokens=6000,
                                            model_kwargs={"stop": "\n"},
                                            platform="DeepInfra"),
                 ):
        self.args = args
        self.task = args.task     
        self.dataset = args.dataset
        self.city = args.city
        self.bs_model = args.bs_model

        if args.aug_method in ['PO','Joint']:
            self.input_path = os.path.join(PARAM_OP_RESULT_PATH,self.task)
        else:
            self.input_path = os.path.join(DA_PA_RESULT_PATH,self.task)

        self.llm = output_llm 
        self.max_steps = max_steps   
        self.reset_period = 1800   #反思间隔是30min 
        self._last_reflect_time = time.time()
        self.__reset_agent()
        self.read_input()
        self.parse_files()
        self.get_summary()  #获得record
    
    def run(self):
        best_choice = self.get_best()
        best_choice = self.parse_choice(best_choice)
        suggestion = self.get_suggestion()
        #根据建议，在不使用历史记录的基础上重新选择
        optimize_choice = self.get_optimize(suggestion)
        # TODO 解析优化结果，重新训练模型，并与原始最优结果作比较。若比较结果更优，则将优化结果作为最终结果。这个过程可以持续多轮，视为反思迭代过程
        try:
            optimize_choice = json.loads(optimize_choice)
        except json.JSONDecodeError:
            optimize_choice = self.parse_choice(optimize_choice)
        return best_choice, suggestion, optimize_choice

    def read_input(self):
        result_files = []
        for bs_model in os.listdir(self.input_path):
            if bs_model == self.bs_model:
                for file in os.listdir(os.path.join(self.input_path, bs_model)):
                    model = file.split("_")[0]
                    dataset = file.split("_")[1]
                    city = file.split("_")[2]
                    if dataset==self.dataset and city==self.city and dataset==self.dataset:
                        result_files.append(file)
        self.result_files = result_files
            
        
    def parse_files(self):
        attempts = []
        for file in self.result_files:
            aug_choices = file.split("_")[3:-6]
            model = file.split("_")[0]
            if "1000" in aug_choices:
                aug_method = ["use original dataset"]
            else:
                aug_method = [AUG_LIST[int(idx)-1] for idx in aug_choices]
            file_name =  os.path.join(self.input_path, self.args.bs_model, file)
            with open(file_name, 'r') as f:
                result = json.load(f)      
            attempt = {
                "model": model,
                "aug_method": aug_method,
                "result":result               
            }
            attempts.append(attempt)
        self.attempts = attempts
    
    def get_summary(self):
        step_match = {0:"1st",1:"2nd",2:"3rd"}
        # Think
        record = ""
        if self.args.aug_method == 'DA':
            for idx, attempt in enumerate(self.attempts):
                if idx in [0,1,2]:
                    step_mark = step_match[idx]
                else:
                    step_mark = str(idx)+"rd" 
                record += "The {} attempt:\nThe data augmentation methods used sequentially:{}\nThe model you choose:{}\nThe output result of the model:{}.".format(step_mark,attempt["aug_method"],attempt["model"],attempt["result"])
        elif self.args.aug_method == 'Joint':
            record += f"Now begin Hyperparameter Tuning with Data Augmentation."
            for idx, attempt in enumerate(self.attempts):
                if idx in [0,1,2]:
                    step_mark = step_match[idx]
                else:
                    step_mark = str(idx)+"rd" 
                record += "The {} attempt:\nThe model you choose:{}\nData augmentation method:{}\nThe hyperparameters combination:{}.".format(step_mark,attempt["model"], attempt['aug_method'], attempt["result"]['config'])  
        elif self.args.aug_method == 'PO':
            record += f"Now begin Hyperparameter Tuning.\n"
            for idx, attempt in enumerate(self.attempts):
                if idx in [0,1,2]:
                    step_mark = step_match[idx]
                else:
                    step_mark = str(idx)+"rd" 
                record += "The {} attempt:\nThe model you choose:{}\nThe hyperparameters combination:{}.".format(step_mark,attempt["model"],attempt["result"]['config'])     
        self.record = record
        
    def get_optimize(self, suggestion):
        record = "<SUGGESTION>\n" + suggestion
        da = "<MEANING OF AUGMENTATION METHODS>\n" + MEANING_OF_OPERATORS
        history = "<HISTORY TRIALS>\n" + str(self.attempts)
        model = "<MODELS>\n" + str(MODEL_DESCRIPTION)
        if self.args.aug_method == 'Joint':
            prompt_head = "You are an agent for optimizing the result of trajectory data mining task {}.The following are suggestions about model selection, data augmentation and hyperparameters tuning.Your task is choosing model, combination of augmentation methods and combination of model hypermeters considering the suggestion."
            prompt = prompt_head + record + da + model + history + "\nPlease output the model, the combination of data augmentation methods and model hyperparameters, and the reason in a json format.\nAnswer:\n"
        elif self.args.aug_method == 'PO':
            prompt_head = "You are an agent for optimizing the result of trajectory data mining task {}.The following are suggestions about model selection and hyperparameters tuning. Your task is choosing model and combination of model hypermeters considering the suggestion."
            prompt = prompt_head + record + model + history + "\nPlease output the model, the combination of model hyperparameters and the reason in a json format.\nAnswer:\n"
        else:
            prompt_head = "You are an agent for optimizing the result of trajectory data mining task {}.The following are suggestions about model selection and data augmentation. Your task is choosing model and combination of augmentation methods considering the suggestion."
            prompt = prompt_head + record + da + model + history + "\nPlease output the model, the combination of augmentation methods and the reason in a json format.\nAnswer:\n"

        optimize_result =  self.llm.get_response(prompt_text=prompt)
        return optimize_result
    
    def get_suggestion(self):
        if self.args.aug_method == 'Joint':
            prompt_head = """You are an agent for summarizing the result of trajectory data mining task {}, use dataset {}.The following record includes model selection, data augmentation and hyperparameter tuning of each attempt.Please summarize the record ,and give suggestions on:\n
            For getting good result on task {}, dataset {}:1.How to select model? 2.Which strategy of data augmentation seems work well? 3.How to do hyparameters tuning?"""
        elif self.args.aug_method == 'DA':
            prompt_head = """You are an agent for summarizing the result of trajectory data mining task {}, use dataset {}.The following record includes model selection, data augmentation.Please summarize the record ,and give suggestions on:\n
            For getting good result on task {}, dataset {}:1.How to select model? 2.Which strategy of data augmentation seems work well?"""    
        else:
            prompt_head = """You are an agent for summarizing the result of trajectory data mining task {}, use dataset {}.The following record includes model selection, data augmentation.Please summarize the record ,and give suggestions on:\n
            For getting good result on task {}, dataset {}:1.How to select model? 2.How to do hyparameters tuning?"""                  
        record = "<RECORD>\n" + self.record
        prompt = prompt_head + record + f"\nPlease answer in a json format,i.e:{{'summary':...,'suggestion':...}}.\nAnswer:\n"
        summary_result =  self.llm.get_response(prompt_text=prompt)
        return summary_result
    
    def get_best(self):
        prompt_head = "Following record includes model selection and the combination of augmentation methods of each attempt for trajectory data mining task {}.Your task is selecting the attempt with the best result among the attempts."
        record = "<RECORD>\n" + self.record
        prompt = prompt_head + record + "\nPlease output the model, the combination of augmentation methods and the reason in a json format.\nAnswer:\n"
        best_result =  self.llm.get_response(prompt_text=prompt)
        return best_result
        
    def analyse_result(self, results_to_analyse):
        prompt_head = "Please analyse following results and choose the best result."
        results = "<RESULTS>:\n" + "\n".join([str(result) for result in results_to_analyse])
        prompt = prompt_head + results + "\nPlease directly output the result you choose.\nAnswer:\n"
        best_result = self.llm(prompt)
        return best_result
    
    def parse_choice(self, raw_result):
        if self.args.aug_method == 'DA':
            parse_result_prompt = f"""
            Please directly output the JSON DICT containing model, the combination of data augmentation methods and the reason from <RAW_RESULT>. If no JSON dict with above contents is found, return an empty dict.Just output a JSON dict, do not output other contents.\n
            <RAW_RESULT>
            {raw_result}
            """
        elif self.args.aug_method == 'PO':
            parse_result_prompt = f"""
            Please directly output the JSON DICT containing model, the combination of model hyperparameters and the reason from <RAW_RESULT>. If no JSON dict with above contents is found, return an empty dict.Just output a JSON dict, do not output other contents.\n
            <RAW_RESULT>
            {raw_result}
            """
        else:
            parse_result_prompt = f"""
            Please directly output the JSON DICT containing model, the combination of model hyperparameters and data augmentation methods, and the reason from <RAW_RESULT>. If no JSON dict with above contents is found, return an empty dict.Just output a JSON dict, do not output other contents.\n
            <RAW_RESULT>
            {raw_result}
            """
        parse_dict = self.llm.get_response(prompt_text=parse_result_prompt)
        config_str = parse_dict.replace("'", '"')
        config_str = config_str.replace("```", "")
        try:
            result_dict = json.loads(json.dumps(eval(config_str)))
        except:
            result_dict={}
        return result_dict

    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        """
        超过最大步数限制，仍未达到优化效果，halted
        """
        return self.step > self.max_steps and not self.finished

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = '<SCRATCHPAD>:\n'
        self.memory: str = "<MEMORY>:\n"

