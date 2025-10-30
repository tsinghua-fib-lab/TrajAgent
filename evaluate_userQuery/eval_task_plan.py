import os
from data_augmentation.utils.base_llm import AnyOpenAILLM, DEEPINFRA
from task_plan.prompt import TASK_DESCRIPTION, TASK_INDEX
import json
import argparse
from UniEnv.etc.settings import LLM_MAP, DIAL_RESULT_PATH
import random

PARSE_INSTRUCT = """
The description of each task is in TASK_DESCRIPTION.Please parse out the task names each sentence aims to address in RAW_INSTRUCTS. 
1. The task name should match the key in the TASK_DESCRIPTION. 
4. Please put each result you parse in a dictionary format.The dictionary format:{{'index':<index of the sentence in RAW_INSTRUCTS>,'task':<task name>}}.
5. Please output a list contains all dictionaries.Do not output other contents.
<TASK_DESCRIPTION>
{task_description}
<RAW_INSTRUCTS>
{raw_instructs}
"""
PARSE_INSTRUCT_SINGLE = """
The description of each task is in TASK_DESCRIPTION.Please parse out the task name the sentence aims to address in RAW_INSTRUCTS. 
1. The task name should match the key in the TASK_DESCRIPTION. 
5. Please only output the task name.Do not output other contents.
<TASK_DESCRIPTION>
{task_description}
<RAW_INSTRUCTS>
{raw_instructs}
"""

def parse_result(llm, choice):
    parse_prompt = f"""
    Please directly output the answer in CONTEXT.The answer must in {list(TASK_INDEX.keys())}.It there is no anwer,output an empty string.\n
    <CONTEXT>
    {choice}
    """
    parse_choice = llm(parse_prompt)
    return parse_choice

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()  
    parser.add_argument("--base_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--question_num", default=2, type=int)
    parser.add_argument("--single_eval", action='store_true')

    args = parser.parse_args()
    all_results = []
    parse_llm = AnyOpenAILLM(
                                temperature=0,
                                max_tokens=300,
                                model_name='gpt-4o-mini',
                                model_kwargs={"stop": "\n"},
                                openai_api_key=os.environ['OPENAI_API_KEY'])
    if "gpt" in args.base_model:
        generalize_llm = AnyOpenAILLM(
                                temperature=0,
                                max_tokens=1000,
                                model_name=LLM_MAP[args.base_model],
                                model_kwargs={"stop": "\n"},
                                openai_api_key=os.environ['OPENAI_API_KEY'])
    else:
        generalize_llm = DEEPINFRA(
                        temperature=0,
                        model_name=LLM_MAP[args.base_model],
                        max_tokens=1000,
                        model_kwargs={"stop": "\n"},
                        openai_api_key=os.environ['DEEPINFRA_API_KEY'])

    raw_instructs = []
    for file in os.listdir("./dataset/evaluate/"): 
        with open(os.path.join("./dataset/evaluate/",file), 'r', encoding='utf-8') as f:
            instruct_item = json.load(f)
            raw_instructs.extend(instruct_item)
    question_list = []
    answer = {}
    success_cnt = 0
  # 随机选择question_num个做测试
    selected_keys = random.sample(range(len(raw_instructs)), args.question_num)
    selected_instruct = [raw_instructs[key] for key in selected_keys]
    
    if args.single_eval:
        for idx, item in enumerate(selected_instruct):
            question = item['sentence']
            result = generalize_llm(PARSE_INSTRUCT_SINGLE.format(task_description=TASK_DESCRIPTION,raw_instructs=question))
            fin_result = TASK_INDEX[parse_result(parse_llm, result)]
            answer = item['target']
            if int(answer) == int(fin_result):
                success_cnt += 1
    else:
        for idx, item in enumerate(raw_instructs[:args.question_num]):
            result_item = {'index':idx,'sentence':item['sentence']}
            question_list.append(result_item)
            if idx not in answer:
                answer[int(idx)] = item['target']
        result = generalize_llm(PARSE_INSTRUCT.format(task_description=TASK_DESCRIPTION,raw_instructs=question_list))  
        result_list = eval(result)
        for item in result_list:
            index = item['index']
            result = int(TASK_INDEX[item['task']]) #将任务转成对应的编号
            target = int(answer[int(index)])  #ground truth答案未是泛化的原始任务编号
            if result==target:
                success_cnt += 1
    success_rate = success_cnt/args.question_num
    print(success_rate)
    with open(os.path.join(DIAL_RESULT_PATH,"instruct_parse.json"),'w') as f:
        json.dump(all_results, f, ensure_ascii=False)
 

