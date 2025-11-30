import os
from task_plan.prompt import TASK_DESCRIPTION, MODULE_DESCRIPTION
from task_plan.utils import extract_choice
from data_format.utils.format_simple import DATA_DESCRIPTION
from data_augmentation.utils.base_llm import LLMWrapper

class PlanAgent:
    def __init__(self,
                 task: str,
                 dataset:str,
                 output_llm: LLMWrapper = LLMWrapper(
                                            temperature=0,
                                            model_name="llama3-70b",
                                            max_tokens=6000,
                                            model_kwargs={"stop": "\n"},
                                            platform="DeepInfra"),
                 ):
        self.task = task     
        self.dataset = dataset
        self.llm = output_llm  
    
    def get_result(self):
        prompt_head = """
        You are an agent for designing and running experiments of trajectory data mining tasks.Please consider the information of task, dataset, and module, then decide whether to use data augmentation strategies or not.
        """
        prompt_mid = f"""
        <TASK INFORMATION>
        The trajectory data mining task is {self.task}.{TASK_DESCRIPTION[self.task]}
        <DATASET INFORMATION>
        The dataset is {self.dataset}.{DATA_DESCRIPTION[self.dataset]}
        <MODULE INFORMATION>
        The whole life cycle of {self.task} includes following modules:
        {MODULE_DESCRIPTION}
        """
        prompt = prompt_head + prompt_mid + "Please choose between following choices:\n A. It is necessary to use data augmentation module singly.\nB. It is necessary to use hyperparameter optimization module singly.\nC. It is necessary to use two modules jointly (data augmentation + hyperparameter optimization).\nPlease choose the most suitable one among A, B, C as the answer to this question.Please output the option directly.  "
        result = self.llm.get_response(prompt_text=prompt)
        choice = extract_choice(result, ['A','B','C'])
        return choice
    def run(self):
        result = self.get_result()
        return result


