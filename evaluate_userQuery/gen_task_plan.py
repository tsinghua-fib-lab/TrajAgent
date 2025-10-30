import os
from data_augmentation.utils.base_llm import AnyOpenAILLM
from task_plan.prompt import TASK_DESCRIPTION
import json

RAW_INSTRUCTS = """
    1.I’m looking to make predictions about the next points of interest users might choose to visit. I’ve collected mobility data from users around the globe. What advice can you give me for structuring my experimental plan?
    2.I want to ascertain if two trajectories were recorded from the same individual or if they correspond to similar movement patterns. What should be my experimental approach?
    3.I’m trying to get a grasp on the time it takes to travel from one point to another. How do you suggest I set this experiment up?
    4.I want to fully reconstruct movement paths using trajectory data that may only be partially available. What’s the most effective way to perform this experiment?
    5.I seek to identify movement trajectories that deviate significantly from standard routes. How should I go about designing my experiment?
    6. I want to align trajectory data from various sources, like GPS, to their respective locations on the road network. What’s the best way to set up this experiment?
    """
GENERALIZE_INSTRUCT = f"""
Please generalize the semantic of each sentence in RAW_INSTRUCTS, with the requirements: 
1. Keep the original meaning unchanged
2. Enrich the expression and sentence structure. Use a more colloquial expression.Sometimes more ambiguous expressions are used. 
3. For each sentence in RAW_INSTRUCTS,generalize it to 100 sentences. 
4. Please put each sentence you generalize in a dictionary format.If it is generalized from the 1st sentence,then the target should be '1';or the target should be '2'.The dictionary format:{{'sentence':<generalized sentence>,'target':<target of the sentence>}}.
5. Please output a list contains all dictionaries.Do not output other contents.
<RAW_INSTRUCTS>
{RAW_INSTRUCTS}
"""


generalize_llm = AnyOpenAILLM(temperature=0.8,
                            max_tokens=3000,
                            model_name="gpt-4o-mini",
                            model_kwargs={"stop": "\n"},
                            openai_api_key=os.environ['OPENAI_API_KEY'])

def parse_result(result):
    parse_list_prompt = f"""
    Please directly output the list in RESULT.Each value of the list should be a dictionary.\n
    <RESULT>
    {result}
    """
    parse_list = generalize_llm(parse_list_prompt)
    # 将字符串转换为JSON格式
    try:
        result = eval(parse_list)
    except:
        result=[]
    return result
if __name__ == '__main__':
    result = generalize_llm(GENERALIZE_INSTRUCT)
    # fin_result = parse_result(result)
    # idx_result = []
    # for idx, result_item in enumerate(fin_result):
    #     idx_item = {}
    #     idx_item['index'] = idx
    #     idx_item['sentence'] = result_item['sentence']
    #     idx_item['target'] = result_item['target']
    #     idx_result.append(idx_item)
    # fm_result = {"plan_eval":idx_result}
    try:
        fin_result = eval(result)
    except:
        fin_result=result
    with open("./dataset/evaluate/task_plan_6.json", 'w') as f:
        json.dump(fin_result, f, ensure_ascii=False)


