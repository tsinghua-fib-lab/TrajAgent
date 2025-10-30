import os
import random
import time
import json
import traceback
import numpy as np
import multiprocessing as mp
from typing import List, Any
from data_augmentation.utils.llm_da_utils import train_model, get_model_result, save_pa_da_config
from data_augmentation.utils.prompts import THOUGHT, QUESTION, ACTION, PA_ACTION, PA_QUESTION, PA_THOUGHT, XR_THOUGHT, XR_QUESTION, XR_ACTION
from data_augmentation.utils.base_llm import LLMWrapper
from UniEnv.etc.settings import *

class ReactAgent:
    def __init__(self,
                 score: float,
                 index: list,
                 n_cpu: int,
                 args: Any,
                 enhance: float,
                 max_steps: int,
                 ) -> None:
        self.args = args
        self.n_cpu = n_cpu        
        self.max_steps = max_steps
        self.enhance = enhance    #停止条件，(step_score - self.threshold)/self.threshold >= self.enhance
        self.max_memory = args.max_memory  #记忆中存储记录超过10条，或者每轮运行超过10分钟仍未达到停止条件（即达到预期的提高效果）时，开始反思
        self.reset_period = 1800   #反思间隔是30min 
        self.threshold = score  #初始值为未经增强的DL模型训练结果（来自da_agent_run.py中的get_model_result(args, [1000,1000])）
        self.threshold_index = index
        # self.llm = react_llm
        self._last_reflect_time = time.time()
        # memory优化参数
        self.memory_optimize_threshold = getattr(args, 'memory_optimize_threshold', 10)  # memory优化阈值
        self.memory_keep_ratio = getattr(args, 'memory_keep_ratio', 0.5)  # 保留比例
        
        # 打印初始threshold信息
        print(f"Initial threshold set to: {self.threshold} (from unenhanced model result)")
        print(f"Initial threshold_index: {self.threshold_index}")
        
        self.__reset_agent()

    def run(self, reset = True) :
        if reset:
            self.__reset_agent()   #重置agent,从step=1开始,重置memory
        self.inter_turn = 0
        self.memory_part = []   #效果好的记忆
        self.llm = LLMWrapper(
                                    temperature=0,
                                    max_tokens=3000,
                                    model_name=self.args.base_model,
                                    platform="DeepInfra",
                                    model_kwargs={"stop": "\n"},
                                    )
                # 固定使用 llama3-70b 进行答案提取
        self.extract_llm = LLMWrapper(
            temperature=0,
            max_tokens=3000,
            model_name='llama3-70b',
            platform="DeepInfra",
            model_kwargs={"stop": "\n"},
        )
        
            
        while not self.is_halted() and not self.is_finished() and int(self.step_n) <= int(self.max_steps):
            step_index, step_score = self.step()
            self.scratchpad = '<SCRATCHPAD>:\n'  #清空短期记忆
            self.step_n += 1
            if (step_score - self.threshold)/self.threshold >= self.enhance:  #若某步优化效果超过self.enhance,则可以停止优化，提前结束
                self.finished = True
            old_threshold = self.threshold
            self.threshold = max(step_score, self.threshold)  #将每一步的优化结果作为下一步的threshold
            self.threshold_index = step_index
            
            # 打印threshold更新信息
            if self.threshold != old_threshold:
                print(f"Step {self.step_n-1}: threshold updated from {old_threshold:.4f} to {self.threshold:.4f}")
            else:
                print(f"Step {self.step_n-1}: threshold unchanged at {self.threshold:.4f}")
                
        return step_index, step_score
    
    def parse_pa_choice(self, choice):

        parse_list_prompt = """
        Please directly output the list containing integers.If no list containing integers is found, return an empty list.Do not output other contents.\n
        """
        
        parse_dict_prompt = """
        Please directly output the dictionary, which keys are integers, and values are dictionaries.If no dictionary is found, return an empty dictionary.Do not output other contents.\n
        """
        parse_list = self.extract_llm.get_response(prompt_text=parse_list_prompt+choice)
        # parse_action = self.llm(parse_action_prompt+choice)
        parse_list = json.loads(parse_list)  
        # if len(parse_list)==0:
        #     print("The trial is invalid!!")
        #     parse_list=None
        parse_dict = self.extract_llm.get_response(prompt_text=parse_dict_prompt+choice)
        config_str = parse_dict.replace("'", '"')
        try:
            result_dict = json.loads(json.dumps(eval(config_str)))
        except:
            result_dict={}
        return parse_list, result_dict

        
    
    def parse_choices(self, choice):
        parse_lists_prompt = """
        Please directly output all lists containing integers in the following TEXT.Separate every two lists with '\n'.If no list containing integers is found, return an empty list.\n
        """
        parse_lists = self.extract_llm.get_response(prompt_text=parse_lists_prompt + '<TEXT BEGIN>\n' + choice + '\n<TEXT END>')
        parse_lists = parse_lists.split('\n')
        result_lists = []
        for parse_list in parse_lists:
            try:
                # 尝试解析列表
                result_list = json.loads(parse_list)
                result_lists.append(result_list)  # 如果解析成功，则添加到结果列表中
            except json.JSONDecodeError:
            # 如果解析失败，处理异常，例如可以打印错误信息，或者忽略错误的条目
                print(f"Error decoding JSON from: {parse_list}")
                continue
        return result_lists

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

    def step(self):
        # Think
        self.inter_turn += 1
        # self.scratchpad += f'\nThought{self.inter_turn}:'
        thought = self.prompt_agent(self.thought)  #从历史交互记录中获得反思，存储在scratchpad中
        parse_thought = self.parse_thought(thought)
        self.scratchpad += f'\nThought{self.inter_turn}:' + parse_thought

        for i in range(self.args.memory_length):
            # Act
            # self.scratchpad += f'\nAction{self.inter_turn}:'
            choices = self.prompt_agent(self.action)  #选择出需要增强的算子index
        
            if self.args.pa_da:
                indexes, parse_dict = self.parse_pa_choice(choices)
                self.scratchpad += f'\nAction{self.inter_turn}:' + str(indexes) + str(parse_dict)
            else:
                indexes = self.parse_choices(choices) #解析出index,用于进行算子增强;reason,选择这个增强顺序的原因;action,文字描述本次增强趋势
                self.scratchpad += f'\nAction{self.inter_turn}:' + str(indexes)
                
            # scratchpad为短期记忆，直接引导action。参考“三思而后行”，即行动前需要总结长期记忆，做出仔细规划，然后根据总结的经验立即执行。
            # 因此，总结经验（Thought）这一步比较重要，因为直接影响着行动/决策的合理性
            
            # Observe
            self.scratchpad += f'\nObservation{self.inter_turn}: '  #各种index以及效果组合
            # with mp.Pool(processes=self.n_cpu) as pool:
            #     pool.starmap(train_model, zip([self.args] * len(indexes), indexes))
            # # 每条memory组成：【尝试编号，算子组合，分数】。若一次生成多个算子组合，则同时生成多条具有相同尝试编号的memory记录
            # for index in indexes:
            #     score = get_model_result(self.args, index)
            #     self.get_memory(self.inter_turn, index, score)
            # indexes=[1,2,9,8]
            print(indexes)
            # parse_dict = {
            #     "1": {"crop_nums": 3, "crop_ratio": 0, "crop_n_times": 3, "crop_time_sort": "minimum", "ti_crop_n_times": 3}, 
            #     "2": {"insert_nums": 1, "insert_ratio": 0, "percent_no_augment": 0, "insert_time_sort": "maximum", "ti_insert_n_times": 1},
            #     "3": {"insert_nums": 2, "insert_ratio": 0, "percent_no_augment": 0, "insert_time_sort": "maximum", "ti_insert_n_times": 2},
            #     "4": {"insert_nums": 1, "insert_ratio": 0, "insert_time_sort": "maximum", "percent_no_augment": 0, "ti_insert_n_times": 1},
            #     "5": {"replace_nums": 2, "replace_ratio": 0, "replace_time_sort": "minimum", "ti_replace_n_times": 2}, 
            #     "9": {"reorder_nums": 3, "reorder_ratio": 0, "reorder_n_times": 2, "sub_seq_length": 5, "reorder_time_sort": "minimum"},
            #     "8": {"mask_nums": 1, "mask_ratio": 0, "mask_time_sort": "minimum","mask_value": 1, "ti_mask_n_times": 1},
            #     "10": {"subset_split_n_times": 2, "dropout_prob": 0.25 }
            #     }
            
            if self.args.pa_da:
                if len(parse_dict)>0:
                    save_pa_da_config(self.args, indexes, parse_dict)  #保存生成的配置文件
                    train_model(self.args, indexes)
                    score = get_model_result(self.args, indexes)
                    self.get_memory(self.inter_turn, indexes, score, parse_dict)
                else:
                    print("Invalid choice!!!")
                    self.get_memory(self.inter_turn, indexes, 0, parse_dict)
            else:
                for index in indexes:
                    train_model(self.args, index)
                    score = get_model_result(self.args, index)
                    self.get_memory(self.inter_turn, index, score, None)
            self.format_memory()

        if not self.args.XR:
            highest_score = sorted(self.memory_part,key=lambda x:x[-2],reverse=True)[0][-2]
            highest_index = sorted(self.memory_part,key=lambda x:x[-2],reverse=True)[0][3]
        else:
            highest_score = score
            highest_index = indexes
            
        if highest_score > self.threshold:
            score = highest_score #将高于阈值的最低分作为新的阈值
            index = highest_index #将高于阈值的最低分作为新的阈值
        else:
            score = self.threshold  #若没有效果更好的尝试，则保持threshold不变
            index = self.threshold_index

        return index, score
        
    def format_memory(self):
        """
        格式化存储self.memory
        """
        memory = "<MEMORY>\n"
        memory_part = '[' + (',').join(str(item) for item in self.memory_part) + ']'
        self.memory = memory + memory_part

    def get_memory(self, turn_num, indexes, score, parse_dict):
        """
        每个step向memory中添加记录
        """
        if not self.args.XR:
            if self.args.pa_da:
                if score > self.threshold:
                    self.memory_part.append(["step num:",turn_num, "operator index list:",indexes, "hypermeters of each operator:", parse_dict, "score:",score, "experience:Good enough.Please tuning the parameters of this operators combination."])
                    # self.memory_part += "[" + (',').join(str(item) for item in indexes) + ']' + f"The score is {score},which is good enough.Please tuning the parameters of this operators combination."
                elif score == 0:
                    self.memory_part.append(["step num:",turn_num, "operator index list:",indexes, "score:",score,"experience:The trial does not have valid results.Need to try other operators combinations.Do not use bad operators combination more than twice."])
                    # self.memory_part += "[" + (',').join(str(item) for item in indexes) + ']' + f"The score is {score},which means the trial does not have valid results.Need to try other operators combinations.Do not use bad operators combination more than twice."
                else:
                    self.memory_part.append(["step num:",turn_num, "operator index list:",indexes, "hypermeters of each operator:", parse_dict, "score:",score,"experience:Not good enough.Need to try other operators combinations.Do not use bad operators combination more than twice."])
                    # self.memory_part += "[" + (',').join(str(item) for item in indexes) + ']' + f"The score is {score},which is not good enough.Need to try other operators combinations. Do not use bad operators combination more than twice."
            else:
                if score > self.threshold:
                    self.memory_part.append(["step num:",turn_num, "operator index list:",indexes, "score:",score,"experience:Good enough.Please tuning the parameters of this operators combination"])
                    # self.memory_part += "[" + (',').join(str(item) for item in indexes) + ']' + f"The score is {score},which is good enough.Please tuning the parameters of this operators combination."
                elif score == 0:
                    self.memory_part.append(["step num:",turn_num, "operator index list:",indexes, "score:",score,"experience:The trial does not have valid results.Need to try other operators combinations. Do not use bad operators combination more than twice."])
                    # self.memory_part += "[" + (',').join(str(item) for item in indexes) + ']' + f"The score is {score},which means the trial does not have valid results.Need to try other operators combinations. Do not use bad operators combination more than twice."

                else:
                    self.memory_part.append(["step num:",turn_num, "operator index list:",indexes, "score:",score,"experience:Not good enough.Need to try other operators combinations. Do not use bad operators combination more than twice."])
                    # self.memory_part += "[" + (',').join(str(item) for item in indexes) + ']' + f"The score is {score},which is not good enough.Need to try other operators combinations. Do not use bad operators combination more than twice."
        else:
            self.memory_part = []     
            
        # 优化memory：当长度超过阈值时，丢弃效果不好的记录
        self._optimize_memory()
        
        self.format_memory()
        if len(self.memory_part)>self.max_memory or (time.time() - self._last_reflect_time > self.reset_period):
            self.get_reflect()
        
    def get_reflect(self):
        """
        定期更新memory中内容，即开始反思
        """
        print("=== Start Reflecting ===")
        print(self.scratchpad)
        
        # # 在反思前先优化memory
        # if len(self.memory_part) > self.memory_optimize_threshold:
        #     print("Optimizing memory before reflection...")
        #     self._optimize_memory()
        
        # # 获取memory统计信息
        # stats = self.get_memory_stats()
        # print(f"Memory stats: {stats}")
        
        # self.good_part = sorted(self.good_part, key=lambda x:x[2])
        # self.bad_part = sorted(self.bad_part, key=lambda x:x[0])
        # self.format_memory()  
        # choices = self.llm(self.memory + self.reflections)     
        # indexes = self.parse_choices(choices) #[[],[],...[]]
        # #基于memory进行反思，一次性提出5条优化方案，并行验证
        # with mp.Pool(processes=self.n_cpu) as pool:
        #     pool.starmap(train_model, zip([self.args] * len(indexes), indexes))
        # for index in indexes:
        #     if get_model_result(self.args, index) > self.threshold:
        #         self.good_part.append([len(self.good_part)+1, index, get_model_result(self.args, index)])
        #     else:
        #         self.bad_part.append([len(self.bad_part)+1, index, get_model_result(self.args, index)])
        # self.format_memory()
        self._last_reflect_time = time.time() 

    def prompt_agent(self, part) -> str:  # 输出包含indexes, reason, action
        return self.extract_llm.get_response(prompt_text=self._build_agent_prompt(part)) # self.llm(self._build_agent_prompt(））为基于历史输入采取的动作
    
    def _build_agent_prompt(self, part) -> str:  # 将每一步的结果作为prompt传进去
        question = self.question
        memory = self.memory
        prompt = question + memory + self.scratchpad + part
        return prompt
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        """
        超过最大步数限制，仍未达到优化效果，halted
        """
        return int(self.step_n) > int(self.max_steps) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = '<SCRATCHPAD>:\n'
        self.memory: str = "<MEMORY>:\n"

    def _optimize_memory(self):
        """
        优化memory：当长度超过阈值时，丢弃效果不好的记录
        递归筛选策略：
        1. 先挑选出所有score > threshold * 0.9的记录
        2. 若记录足够多，再保留这些记录中前memory_keep_ratio的记录
        3. 若剩余记录还是很多，就保留这些记录中最近的一部分记录
        4. 每次挑选时，若剩余memory数量大于N，则进一步筛选
        """
        if len(self.memory_part) <= self.memory_optimize_threshold:
            return
        
        print(f"Memory optimization triggered: current length {len(self.memory_part)} > threshold {self.memory_optimize_threshold}")
        
        # 递归筛选函数
        def memory_filter(records, target_size, level=0):
            """
            递归筛选记录
            records: 要筛选的记录列表
            target_size: 目标保留数量
            level: 递归层级
            """

            if len(records) <= target_size:
                print(f"Level {level}: Records count ({len(records)}) <= target ({target_size}), stopping recursion")
                return records
            
            # 按score排序
            records.sort(key=lambda x: x['score'], reverse=True)
            
            # 策略1：先挑选出所有score > threshold * 0.9的记录
            good_threshold = self.threshold * 0.9
            good_records = [item for item in records if item['score'] > good_threshold]
            print(f"Level {level} Strategy 1: Found {len(good_records)} records with score > {good_threshold:.4f}")
            
            if len(good_records) > target_size:
                # 策略2：若记录足够多，再保留这些记录中前memory_keep_ratio的记录
                top_count = max(target_size, int(len(good_records) * self.memory_keep_ratio))
                top_good_records = good_records[:top_count]
                print(f"Level {level} Strategy 2: Keeping top {top_count} records from {len(good_records)} good records")
            
                if len(top_good_records) > target_size:
                    # 按时间顺序排序（假设index越大越新）
                    top_good_records.sort(key=lambda x: x['index'], reverse=True)
                    recent_records = top_good_records[:target_size]
                    print(f"Level {level} Strategy 3: Keeping most recent records from remaining good records")
                    
                    # 合并所有保留的记录
                    kept_records = recent_records
                else:
                    kept_records = top_good_records
            else:
                print(f"Level {level} Warning: No records found with score > {good_threshold:.4f}, keeping all records")
                kept_records = good_records

            return kept_records
        
        # 提取所有记录的score信息
        memory_with_scores = []
        for i, memory_item in enumerate(self.memory_part):
            if len(memory_item) >= 6:  # 确保有足够的信息
                # 找到score的位置
                score_idx = None
                for j, item in enumerate(memory_item):
                    if isinstance(item, str) and item == "score:":
                        if j + 1 < len(memory_item):
                            try:
                                score = float(memory_item[j + 1])
                                memory_with_scores.append({
                                    'index': i,
                                    'memory_item': memory_item,
                                    'score': score
                                })
                                break
                            except (ValueError, TypeError):
                                continue
        
        if not memory_with_scores:
            print("Warning: No valid scores found in memory for optimization")
            return
        
        # 设置目标保留数量
        target_size = max(1, int(len(self.memory_part) * 0.3))  # 保留30%的记录
        print(f"Target size: {target_size} (30% of {len(self.memory_part)} records)")
        
        # 执行递归筛选
        filtered_records = memory_filter(memory_with_scores, target_size)
        
        # 提取保留的索引
        keep_indices = set(item['index'] for item in filtered_records)
        
        # 创建新的memory_part
        new_memory_part = []
        for i in range(len(self.memory_part)):
            if i in keep_indices:
                new_memory_part.append(self.memory_part[i])
        
        # 更新memory_part
        removed_count = len(self.memory_part) - len(new_memory_part)
        self.memory_part = new_memory_part
        
        print(f"Memory optimization completed: removed {removed_count} records, kept {len(new_memory_part)} records")
        print(f"Optimization parameters: threshold={self.memory_optimize_threshold}, keep_ratio={self.memory_keep_ratio}")
        
        # 打印保留的记录信息
        if new_memory_part:
            print("Kept records:")
            for i, item in enumerate(new_memory_part):
                if len(item) >= 6:
                    for j, sub_item in enumerate(item):
                        if isinstance(sub_item, str) and sub_item == "score:" and j + 1 < len(item):
                            try:
                                score = float(item[j + 1])
                                print(f"  Record {i}: score = {score}")
                                break
                            except (ValueError, TypeError):
                                continue

    def optimize_memory_manually(self):
        """
        手动触发memory优化
        """
        print("Manual memory optimization triggered")
        self._optimize_memory()
        self.format_memory()
        
    def get_memory_stats(self):
        """
        获取memory统计信息
        """
        if not self.memory_part:
            return {
                'total_records': 0,
                'avg_score': 0,
                'best_score': 0,
                'worst_score': 0
            }
        
        scores = []
        for memory_item in self.memory_part:
            if len(memory_item) >= 6:
                for j, item in enumerate(memory_item):
                    if isinstance(item, str) and item == "score:" and j + 1 < len(memory_item):
                        try:
                            score = float(memory_item[j + 1])
                            scores.append(score)
                            break
                        except (ValueError, TypeError):
                            continue
        
        if not scores:
            return {
                'total_records': len(self.memory_part),
                'avg_score': 0,
                'best_score': 0,
                'worst_score': 0
            }
        
        return {
            'total_records': len(self.memory_part),
            'avg_score': sum(scores) / len(scores),
            'best_score': max(scores),
            'worst_score': min(scores),
            'score_count': len(scores)
        }

class ReactReflectAgent(ReactAgent):
    def __init__(self,
                 score: float,
                 index: list,
                 args:  Any,
                 n_cpu: int,
                 enhance: float,
                 max_steps: int,
                 ) -> None:
        self.args = args
        self.n_cpu = n_cpu  # 修正语法错误
        self.enhance = args.enhance
        self.max_steps = args.max_step
        if self.args.pa_da:
            if self.args.XR:
                self.question = XR_QUESTION
                self.thought = XR_THOUGHT
                self.action = XR_ACTION
            else:
                self.question = PA_QUESTION
                self.thought = PA_THOUGHT.format(threshold=score)
                self.action = PA_ACTION
        else:
            self.question = QUESTION
            self.thought = THOUGHT
            self.action = ACTION
        super().__init__(score, index, n_cpu, args, enhance, max_steps)

    def run(self, reset = True):
        # if not self.is_finished() or self.is_halted():
        #     self.reflect()  #如果之前所有轮的run()运行未得到合理结果，进行反思机制，即reflect
        error_cnt=0
        correct_cnt = 0
        all_scores = 0
        for i in range(self.args.trial_num):
            print(f"run - {i+1}")
            try:
                result_indexes, result_score = super().run(reset)  #调用父类的run方法
                
                if len(result_indexes) == 0:
                    print(f"{i+1} error, result_indexes == 0")
                    error_cnt += 1
                else:
                    print(f"{i+1} ok")
                    all_scores += result_score
                    correct_cnt += 1
            except:
                print(f"{i+1} error")
                traceback.print_exc()
                result_indexes=[]
                result_score = 0.1707 
                error_cnt += 1
        if correct_cnt==0:
            ave_suc_score = 0
        else:
            ave_suc_score = all_scores/correct_cnt
            
        print("multi-turns:correct_rate:",correct_cnt/self.args.trial_num, ",ave_suc_score:",ave_suc_score, all_scores/self.args.trial_num)

        return result_indexes, result_score

    
    