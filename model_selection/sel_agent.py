import os
import time
import json
import numpy as np
import multiprocessing as mp
from typing import List, Any
from model_selection.utils.utils import get_model_description, get_model_dataset
from model_selection.utils.prompts import *
from data_augmentation.utils.base_llm import LLMWrapper
from UniEnv.etc.settings import *
from task_plan.prompt import *
from UniEnv.etc.settings import *

class SelAgent:
    def __init__(self,
                 n_cpu: int,
                 args: Any,
                 max_steps: int,
                 ) -> None:
        self.args = args
        self.n_cpu = n_cpu        
        self.max_steps = max_steps
        self.max_memory = args.max_memory  #记忆中存储记录超过10条，或者每轮运行超过10分钟仍未达到停止条件（即达到预期的提高效果）时，开始反思
        self.reset_period = 1800   #反思间隔是30min 
        self.react_llm = LLMWrapper(
                        temperature=0.5,
                        model_name=self.args.base_model,
                        max_tokens=3000,
                        model_kwargs={"stop": "\n"},
                        platform="DeepInfra")
        self.choose_llm = LLMWrapper(
                        temperature=0.5,
                        model_name="llama3-70b",
                        max_tokens=3000,
                        model_kwargs={"stop": "\n"},
                        platform="DeepInfra")
        self._last_reflect_time = time.time()
        self.__reset_agent()

    def run(self, reset = True) :
        if reset:
            self.__reset_agent()   #重置agent,从step=1开始,重置memory
        # self.model_data = get_model_dataset(MODEL_DESCRIPTION, DATA_DESCRIPTION, self.react_llm)  #让LLM先从所有模型的模型描述（由LLM生成）总结每个模型的适用数据集，以字典的形式存储。这个过程不区分任务。
        self.inter_turn = 0
        self.memory_part = []   
        self.flag = -1

        while not self.is_halted() and not self.is_finished() and self.step_n <= self.max_steps:
            step_model = self.step(self.args)  #每个step解析出的模型和数据集
            self.scratchpad = '<SCRATCHPAD>:\n'  #清空短期记忆
            self.step_n += 1
            if step_model:  #若选择到了有效的模型和数据集，停止思考，输出
                self.finished = True
        return step_model
    
    def parse_model(self, choice):
        parse_model_prompt = """
        Please select model name from MODEL LIST that appears in the TEXT.Please directly output the model name you choose.Do not output other things.\n
        """
        model = self.choose_llm.get_response(prompt_text=parse_model_prompt + '<TEXT>\n' + choice + '\n<MODEL LIST>\n' + str(MODEL_DESCRIPTION.keys()))
        return model

    def parse_thought(self, thought):
        
        # parse_thought_prompt = """
        # Please remove the part after "Action:" of the above sentence and output:\n
        # """
        # parse_thought = self.llm(thought + parse_thought_prompt)
        action_index = thought.find("Action:")# Action:
        if action_index!= -1:
            parse_thought = thought[:action_index]
        else:
            parse_thought = thought
        return parse_thought

    def step(self, args):
        # Think
        self.inter_turn += 1
        # self.scratchpad += f'\nThought{self.inter_turn}:'
        thought = self.prompt_agent(self.thought)  #从历史交互记录中获得反思，存储在scratchpad中
        parse_thought = self.parse_thought(thought)
        self.scratchpad += f'\nThought{self.inter_turn}:' + parse_thought

        # Act
        # self.scratchpad += f'\nAction{self.inter_turn}:'
        output = self.prompt_agent_choose(self.action)  
        model = self.parse_model(output) # 获得选择的模型,数据集
        self.scratchpad += f'\nAction{self.inter_turn}:' + model

        if model not in MODEL_DESCRIPTION.keys():
            result = "The choice of model is invalid, please choose the model again."
            self.flag=-1
        else:
            result = "The choice of dataset and model can be reasonable."
            self.flag=1
        # Observe
        self.scratchpad += f'\nObservation{self.inter_turn}: '  

        # (["step num:",self.inter_turn, "dataset:", self.args.dataset, "task:",self.args.task, "model name:",model , "score:",score])
        self.get_memory(model, result)
        self.format_memory()

        if self.flag > 0:
            result_model = model 
        else:
            result_model = None
        return result_model
        
    def format_memory(self):
        """
        格式化存储self.memory
        """
        memory = "<MEMORY>\n"
        memory_part = '[' + (',').join(str(item) for item in self.memory_part) + ']'
        self.memory = memory + memory_part
# self.args, self.inter_turn, 
    def get_memory(self, model, result):
        """
        每个step向memory中添加记录
        """
        self.memory_part.append(["step num:",self.inter_turn, "task:",self.args.task, "model name:",model , "result:",result])
        self.format_memory()
        if len(self.memory_part)>self.max_memory or (time.time() - self._last_reflect_time > self.reset_period):
            self.get_reflect()
        
    def get_reflect(self):
        """
        定期更新memory中内容，即开始反思
        """
        # print(self.scratchpad)
        # TODO 待增加反思机制
        self._last_reflect_time = time.time() 

    def prompt_agent(self, part) -> str:  # 输出包含indexes, reason, action
        return self.react_llm.get_response(prompt_text=self._build_agent_prompt(part))  # self.llm(self._build_agent_prompt(））为基于历史输入采取的动作
    
    def prompt_agent_choose(self, part) -> str:  # 输出包含indexes, reason, action
        return self.react_llm.get_response(prompt_text=self._build_agent_prompt(part))  # self.llm(self._build_agent_prompt(））为基于历史输入采取的动作

    
    def _build_agent_prompt(self, part) -> str:  # 将每一步的结果作为prompt传进去
        model_description = get_model_description(MODEL_DESCRIPTION)
        question = self.question.format(task=self.args.task,model_characteristics=model_description)
        memory = self.memory
        prompt = question + memory + self.scratchpad + part
        return prompt
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        """
        超过最大步数限制，仍未达到优化效果，halted
        """
        return self.step_n > self.max_steps and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = '<SCRATCHPAD>:\n'
        self.memory: str = "<MEMORY>:\n"

class SelReflectAgent(SelAgent):
    def __init__(self,
                 args:  Any,
                 n_cpu: int,
                 max_steps: int,
                 ) -> None:
        self.args = args
        n_cpu = 3, 
        self.max_steps = args.max_step
        self.question = QUESTION
        self.thought = THOUGHT
        self.action = ACTION

        super().__init__(n_cpu, args, max_steps)

    def run(self, reset = True) -> None:    
        error_cnt = 0
        correct_cnt = 0
        result_model = None
        # for i in range(self.args.trial_num):
        #     try:
        #         result_model= SelAgent.run(self)  #memory存储历史所有尝试组合和准确率
        #         if result_model:
        #             correct_cnt += 1
        #         else:
        #             error_cnt += 1
        #     except:
        #         error_cnt += 1
        # if correct_cnt==0:
        #     print("All trial fail!!")
        # print("total counts:correct_rate:", correct_cnt/int(self.args.trial_num))
        result_model= SelAgent.run(self)
        return result_model
