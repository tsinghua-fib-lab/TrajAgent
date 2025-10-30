import os
import random
import time
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import List, Any
from data_augmentation.utils.base_llm import LLMWrapper
from data_format.utils.prompt_simple import PROMPT_HEAD,OPERATORS
from data_format.utils.format_simple import DATA_SAMPLE,DATA_DESCRIPTION
from UniEnv.etc.settings import *

class FormatAgent:
    def __init__(self,
                 source: str,
                 target: str,
                 args: Any,
                 react_llm: LLMWrapper = LLMWrapper(
                                            temperature=0,
                                            max_tokens=4000,
                                            model_name="llama3-70b",
                                            platform="DeepInfra",
                                            model_kwargs={"stop": "\n"}),                  
                 ):
        self.operator = OPERATORS
        self.source = source
        self.args = args
        self.target = target
        self.llm = react_llm  # 提取答案的LLM，用强一点的模型会比较好。之前是GPT-4o-mini，现在统一换成llama3-70b
        self.task = PROMPT_HEAD
        self.reason_llm = LLMWrapper(
                                temperature=0.5,
                                platform="DeepInfra",
                                max_tokens=4000,
                                model_name=self.args.base_model,
                                model_kwargs={"stop": "\n"})
            
    def parse_code(self, text):
        parse_prompt = """Please extract the python code in the text.Please output the code only."""
        code = self.llm.get_response(prompt_text=text + parse_prompt)
        return code
        
    def get_format_prompt(self): 
        operator = self.operator.format(output_path=self.args.fm_output_path, city=self.args.city, dataset=self.source)
        task = self.task.format(source_example=DATA_DESCRIPTION["gowalla"],source_sample=DATA_SAMPLE["gowalla"],source_example2=DATA_DESCRIPTION["foursquare"],source_sample2=DATA_SAMPLE["foursquare"],target_sample=DATA_SAMPLE["standard"], target_example=DATA_DESCRIPTION["standard"], path=self.args.fm_data_path, operator=operator, output_path=self.args.fm_output_path, city=self.args.city, dataset=self.source)
        source_data = "\n<SOURCE DATA DESCRIPTION>\n" + DATA_DESCRIPTION[self.source]
        target_data = "\n<TARGET DATA DESCRIPTION>\n" + DATA_DESCRIPTION[self.target]
        source_sample = "\n<SOURCE DATA SAMPLE>\n"  + DATA_SAMPLE[self.source]
        target_sample = "\n<TARGET DATA SAMPLE>\n" + DATA_SAMPLE[self.target]
        return task + source_data + target_data + self.operator + source_sample + target_sample + "\nResponse:\n"

    def run(self):
        prompt = self.get_format_prompt()
        text = self.reason_llm.get_response(prompt_text=prompt)         
        code = self.parse_code(text)
        return code
        
