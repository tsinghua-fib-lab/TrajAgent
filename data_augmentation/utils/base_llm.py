from typing import Union, Literal
import os
import httpx
from openai import OpenAI
from token_count import TokenCount
import time
import random
import jsonlines
import datetime
import argparse

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from UniEnv.etc.settings import *


PROXY = "http://127.0.0.1:10190"
MAX_RETRIES = 5
WAIT_TIME_MIN = 3
WAIT_TIME_MAX = 60
ATTEMPT_COUNTER = 3
FREE_APIS=['llama3-8b', 'llama3.1-8b', 'gemma2-9b', 'mistral7bv2', 'qwen2-1.5b', 'qwen2-7b', 'glm4-9b', 'glm3-6b']

nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') 

def token_count(text):
    tc = TokenCount(model_name="gpt-3.5-turbo")
    return tc.num_tokens_from_string(text)

#     def __init__(self, *args, **kwargs):
#         # Initialize with default values and override with any given kwargs
#         self.model_name = kwargs.get('model_name', 'gpt-4o-mini') 
#         self.max_tokens = kwargs.get('max_tokens', 300) 
#         self.temperature = kwargs.get('temperature', 0)
#         self.chat_history = []  # Empty list to store the chat history
#         self.history_file_name = os.path.join(DIAL_RESULT_PATH, self.model_name.split("/")[-1]+str(nowTime)+"_history.json")
     
#     def __call__(self, prompt: str):
#         # Initialize OpenAI client
#         client = OpenAI(
#                 base_url=OPENAI_API_BASE,
#                 api_key=os.environ["OPENAI_API_KEY"],
#                 http_client=httpx.Client(proxies=PROXY),
#             )
#         response = client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=self.max_tokens,
#             temperature=self.temperature
#         )
        
#         # Extract the full response text
#         self.chat_history = []
#         full_text = response.choices[0].message.content
#         # Append the user and assistant messages to the history
#         self.chat_history.append({"role": "user", "content": prompt})
#         self.chat_history.append({"role": "assistant", "content": full_text})
        
#         # Optionally return the full chat history
#         with jsonlines.open(self.history_file_name, 'a') as wid:
#             # print(self.chat_history)
#             wid.write(self.chat_history)
    
#         return full_text
        
        
# class DEEPINFRA:
#     def __init__(self, *args, **kwargs):
#         # Determine model type from the kwargs
#         self.model_name = kwargs.get('model_name', 'meta-llama/Meta-Llama-3.1-70B-Instruct')  
#         self.max_tokens = kwargs.get('max_tokens', 300) 
#         self.temperature = kwargs.get('temperature', 0) 
#         self.chat_history = []  # Empty list to store the chat history
            
#     def __call__(self, prompt: str, return_history=False):
#         client = OpenAI(
#             base_url="https://api.deepinfra.com/v1/openai",
#             api_key=os.environ["DEEPINFRA_API_KEY"],
#             http_client=httpx.Client(proxies="http://127.0.0.1:10190"),
#             )
#         response = client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=self.temperature,
#                     max_tokens=self.max_tokens,
#                 )
#         full_text = response.choices[0].message.content
        
#         self.chat_history.append({"role": "user", "content": prompt})
#         self.chat_history.append({"role": "assistant", "content": full_text})
        
#         # Optionally return the full chat history
#         if return_history:
#             return self.chat_history
#         else:
#             return full_text

def get_api_key(platform, model_name=None):
    if platform=="OpenAI":
        return os.environ["OpenAI_API_KEY"]
    elif platform=="DeepInfra":
        return os.environ["DEEPINFRA_API_KEY"]
    elif platform=="vllm":
        return os.environ["vllm_KEY"]
    elif platform=="SiliconFlow":
        if model_name in FREE_APIS:
            # 免费可以使用多个key
            keys = [os.environ["SiliconFlow_API_KEY"]]
            try:
                k2 = [os.environ["SiliconFlow_API_KEY_yuwei"]]
                k3 = [os.environ["SiliconFlow_API_KEY_tianhui"]]
                keys = keys+k2+k3
            except:
                pass
            return random.choice(keys)
        else:
            # 付费只走自己的key
            return os.environ["SiliconFlow_API_KEY"]


class LLMAPI:
    def __init__(self, model_name, platform=None):
        self.model_name = model_name
        
        # 优先SiliconFlow, 然后DeepInfra，除非指定platform
        self.platform_list = ["SiliconFlow", "OpenAI", "DeepInfra", 'vllm']
        self.model_platforms = {
                    # "SiliconFlow":  [
                    #     'llama3-8b', 'llama3-70b', 'gemma2-9b', 'gemma2-27b', 'mistral7bv2', 'qwen2-1.5b', 'qwen2-7b', 'qwen2-14b', 'qwen2-72b', 'glm4-9b', 'glm3-6b', 'deepseekv2', 'llama3.1-8b', 'llama3.1-70b', 'llama3.1-405b'] + [
                    #     'llama3-8b-pro', 'gemma2-9b-pro', 'mistral7bv2-pro', 'qwen2-1.5b-pro', 'qwen2-7b-pro', 'glm4-9b-pro', 'glm3-6b-pro'
                    # ],
                    "OpenAI":       ['gpt35turbo', 'gpt4turbo', 'gpt4o', 'gpt4omini'],
                    "DeepInfra":    ['llama3-8b', 'llama3-70b', 'gemma2-9b', 'gemma2-27b', 'mistral7bv2', 'qwen2-7b', 'qwen2-72b', 'llama3.1-8b', 'llama3.1-70b', 'mistral7bv3', 'llama3.1-405b','deepseek-v3','gemini'],
                    "vllm": ['llama3-8B-local', 'gemma2-2b-local', 'chatglm3-citygpt', 'chatglm3-6B-local']
                }
    #     
        self.model_mapper = {
            'gpt35turbo': 'gpt-3.5-turbo-0125',
            'gpt4turbo': 'gpt-4-turbo-2024-04-09',
            'gpt4o': 'gpt-4o-2024-05-13',
            'gpt4omini': 'gpt-4o-mini-2024-07-18',
            'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'llama3-8b-pro': 'Pro/meta-llama/Meta-Llama-3-8B-Instruct',
            'llama3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
            'llama3.1-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            'llama3.1-405b': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
            'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
            'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
            'llama2-70b': 'meta-llama/Llama-2-70b-chat-hf',
            'gemma2-9b': 'google/gemma-2-9b-it',
            'gemma2-9b-pro': 'Pro/google/gemma-2-9b-it',
            'gemma2-27b': 'google/gemma-2-27b-it',
            'mistral7bv2': 'mistralai/Mistral-7B-Instruct-v0.2',
            'mistral7bv3': 'mistralai/Mistral-7B-Instruct-v0.3',
            'mistral7bv2-pro': 'Pro/mistralai/Mistral-7B-Instruct-v0.2',
            'qwen2-1.5b': 'Qwen/Qwen2-1.5B-Instruct',
            'qwen2-1.5b-pro': 'Pro/Qwen/Qwen2-1.5B-Instruct',
            'qwen2-7b': 'Qwen/Qwen2-7B-Instruct',
            'qwen2-7b-pro': "Pro/Qwen/Qwen2-7B-Instruct",
            'qwen2-14b': 'Qwen/Qwen2-57B-A14B-Instruct',
            'qwen2-72b': 'Qwen/Qwen2-72B-Instruct',
            'glm4-9b': 'THUDM/glm-4-9b-chat',
            'glm4-9b-pro': 'Pro/THUDM/glm-4-9b-chat',
            'glm3-6b': 'THUDM/chatglm3-6b',
            'glm3-6b-pro': 'Pro/THUDM/chatglm3-6b',
            'deepseekv2': 'deepseek-ai/DeepSeek-V2-Chat',
            'llama3-8B-local':'llama3-8B-local',
            'gemma2-2b-local': 'gemma2-2b-local',
            'chatglm3-citygpt': 'chatglm3-citygpt',
            'chatglm3-6B-local': 'chatglm3-6B-local',
            'deepseek-v3': 'deepseek-ai/DeepSeek-V3-0324',
            'gemini': 'google/gemini-2.0-flash-001'
        }

        # 判断模型是否存在
        support_models = ";".join([";".join(self.model_platforms[k]) for k in self.model_platforms])
        if self.model_name not in support_models:
            raise ValueError('Invalid model name! Please use one of the following: {}'.format(support_models))
        
        # 优先指定平台，否则自动匹配
        if platform is not None and platform in self.platform_list:
            self.platform = platform
        else:
            for platform in self.platform_list:
                if self.model_name in self.model_platforms[platform]:
                    self.platform = platform
                    break
        # 平台匹配失败
        if self.platform is None:
            raise ValueError("'Invalid API platform:{} with model:{}".format(self.platform, self.model_name))
        # 模型与平台不匹配
        if self.model_name not in self.model_platforms[self.platform]:
            raise ValueError('Invalid model name! Please use one of the following: {} in API platform:{}'.format(support_models, self.platform))
        
        # 生成访问链接
        if self.platform == "OpenAI":
            self.client = OpenAI(
                base_url="https://api.deepbricks.ai/v1/",
                api_key=get_api_key(platform),
                http_client=httpx.Client(proxies=PROXY),
            )
        elif self.platform == "DeepInfra":
            self.client = OpenAI(
                base_url="https://api.deepinfra.com/v1/openai",
                api_key=get_api_key(platform)
                # http_client=httpx.Client(proxies=PROXY),
            )
        elif self.platform == "SiliconFlow":
            self.client = OpenAI(
                base_url="https://api.siliconflow.cn/v1",
                api_key=get_api_key(platform, model_name)
            )
    
    def get_client(self):
        return self.client
    
    def get_model_name(self):
        return self.model_mapper[self.model_name]
    
    def get_platform_name(self):
        return self.platform

    def get_supported_models(self):
        return self.model_platforms


class LLMWrapper:
    def __init__(self, model_name, platform=None, **kwargs):
        self.model_name = model_name
        self.hyperparams = {
            'temperature': 0.,  # make the LLM basically deterministic
            'max_new_tokens': 100 # not used in OpenAI API
        }
        self.chat_history = []  # Empty list to store the chat history
        
        # 初始化token统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        
        # 创建更详细的日志记录路径
        model_short_name = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
        history_path = os.path.join(DIAL_RESULT_PATH, model_short_name)
        if not os.path.exists(history_path):
            os.makedirs(history_path)
        
        # 使用全局会话ID确保一次执行过程中所有LLM交互记录在同一个文件
        if not hasattr(LLMWrapper, '_session_id'):
            LLMWrapper._session_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print(f"创建新的LLM会话: {LLMWrapper._session_id}")
        
        # 创建带会话ID的日志文件名
        self.history_file_name = os.path.join(history_path, f"{LLMWrapper._session_id}_history.jsonl")
        self.detailed_log_file = os.path.join(history_path, f"{LLMWrapper._session_id}_detailed_log.jsonl")
        
        # 初始化时打开JSON文件，准备追加记录
        self._init_log_files()
        
        self.llm_api = LLMAPI(self.model_name, platform=platform)
        self.client = self.llm_api.get_client()
        self.api_model_name = self.llm_api.get_model_name()
        self.max_tokens = kwargs.get('max_tokens', 300) 
        self.temperature = kwargs.get('temperature', 0)
        
        # 记录初始化信息
        self._log_initialization()

    def _init_log_files(self):
        """初始化日志文件，创建文件头信息"""
        try:
            # 创建详细日志文件并写入文件头
            with jsonlines.open(self.detailed_log_file, 'w') as writer:
                file_header = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": "file_header",
                    "session_id": LLMWrapper._session_id,
                    "model_name": self.model_name,
                    "file_type": "detailed_log",
                    "description": "LLM交互详细日志文件"
                }
                writer.write(file_header)
            
            # 创建历史文件并写入文件头
            with jsonlines.open(self.history_file_name, 'w') as writer:
                file_header = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": "file_header",
                    "session_id": LLMWrapper._session_id,
                    "model_name": self.model_name,
                    "file_type": "chat_history",
                    "description": "LLM聊天历史记录文件"
                }
                writer.write(file_header)
                
        except Exception as e:
            print(f"Warning: Failed to initialize log files: {e}")

    def _log_initialization(self):
        """记录LLM初始化信息"""
        init_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "initialization",
            "model_name": self.model_name,
            "api_model_name": self.api_model_name,
            "platform": self.llm_api.get_platform_name(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "hyperparams": self.hyperparams
        }
        
        # 直接追加到详细日志文件
        self._append_to_log_file(self.detailed_log_file, init_log, "initialization")

    def _log_interaction(self, prompt_text, response_text, response_obj=None, error=None, input_tokens=None, output_tokens=None, total_tokens=None):
        """记录LLM交互的详细信息"""
        # 如果没有提供token信息，使用估算
        if input_tokens is None:
            input_tokens = token_count(prompt_text)
        if output_tokens is None:
            output_tokens = token_count(response_text)
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        
        interaction_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "interaction",
            "model_name": self.model_name,
            "api_model_name": self.api_model_name,
            "platform": self.llm_api.get_platform_name(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "prompt_tokens": input_tokens,
            "response_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cumulative_input_tokens": self.total_input_tokens,
            "cumulative_output_tokens": self.total_output_tokens,
            "cumulative_total_tokens": self.total_tokens,
            "error": error
        }
        
        # 如果有完整的响应对象，记录更多信息
        if response_obj:
            interaction_log.update({
                "response_id": str(getattr(response_obj, 'id', '')),
                "response_usage": str(getattr(response_obj, 'usage', '')),
                "response_finish_reason": str(getattr(response_obj.choices[0], 'finish_reason', '')) if response_obj.choices else ''
            })
        
        # 直接追加到详细日志文件
        self._append_to_log_file(self.detailed_log_file, interaction_log, "interaction")

    def _append_to_log_file(self, file_path, log_entry, entry_type):
        """追加记录到日志文件"""
        try:
            # 简单地将所有值转换为字符串格式
            serializable_log_entry = self._convert_to_serializable(log_entry)
            
            with jsonlines.open(file_path, 'a') as writer:
                writer.write(serializable_log_entry)
        except Exception as e:
            print(f"Warning: Failed to log {entry_type}: {e}")

    def _convert_to_serializable(self, obj):
        """简单地将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {str(k): str(v) if not isinstance(v, (dict, list)) else self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [str(item) if not isinstance(item, (dict, list)) else self._convert_to_serializable(item) for item in obj]
        else:
            return str(obj)

    @retry(wait=wait_random_exponential(min=WAIT_TIME_MIN, max=WAIT_TIME_MAX), stop=stop_after_attempt(ATTEMPT_COUNTER))
    def get_response(self, prompt_text):
        if "gpt" in self.model_name:
            system_messages = [{"role": "system", "content": "You are a helpful assistant who predicts user next location."}]
        else:
            system_messages = []
        
        # 手动截断，取后面的部分
        original_prompt = prompt_text
        if token_count(prompt_text) > self.max_tokens:
            # 基于token预估一个系数
            prompt_text = prompt_text[-min(self.max_tokens*3, len(prompt_text)):]
        
        try:
            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=system_messages + [{"role": "user", "content": prompt_text}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            full_text = response.choices[0].message.content
            
            # 获取API返回的token使用情况
            if hasattr(response, 'usage') and response.usage:
                input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                output_tokens = getattr(response.usage, 'completion_tokens', 0)
                total_tokens = getattr(response.usage, 'total_tokens', 0)
            else:
                # 如果API没有返回usage信息，使用估算
                input_tokens = token_count(prompt_text)
                output_tokens = token_count(full_text)
                total_tokens = input_tokens + output_tokens
            
            # 更新累计token统计
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_tokens += total_tokens
            
            # 记录到聊天历史
            user_message = {"role": "user", "content": prompt_text}
            assistant_message = {"role": "assistant", "content": full_text}
            
            self.chat_history.append(user_message)
            self.chat_history.append(assistant_message)
            
            # 追加记录到历史文件
            self._append_to_log_file(self.history_file_name, user_message, "user_message")
            self._append_to_log_file(self.history_file_name, assistant_message, "assistant_message")
            
            # 记录详细的交互信息，包含token统计
            self._log_interaction(original_prompt, full_text, response, input_tokens, output_tokens, total_tokens)
            
            return full_text
            
        except Exception as e:
            # 记录错误信息
            self._log_interaction(original_prompt, "", error=str(e))
            raise e

    def get_log_files(self):
        """获取日志文件路径"""
        return {
            "history_file": self.history_file_name,
            "detailed_log_file": self.detailed_log_file
        }

    def get_interaction_count(self):
        """获取交互次数"""
        try:
            count = 0
            with jsonlines.open(self.detailed_log_file, 'r') as reader:
                for line in reader:
                    if line.get("type") == "interaction":
                        count += 1
            return count
        except:
            return 0

    @classmethod
    def reset_session(cls):
        """重置会话ID，开始新的会话"""
        if hasattr(cls, '_session_id'):
            old_session = cls._session_id
            delattr(cls, '_session_id')
            print(f"重置LLM会话: {old_session} -> 新会话")
        else:
            print("重置LLM会话: 无当前会话")

    @classmethod
    def get_session_id(cls):
        """获取当前会话ID"""
        return getattr(cls, '_session_id', None)

    def get_log_status(self):
        """获取日志文件状态信息"""
        status = {
            "session_id": LLMWrapper._session_id,
            "detailed_log_file": self.detailed_log_file,
            "history_file": self.history_file_name,
            "interaction_count": self.get_interaction_count(),
            "chat_history_length": len(self.chat_history),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens
        }
        
        # 检查文件是否存在
        status["detailed_log_exists"] = os.path.exists(self.detailed_log_file)
        status["history_file_exists"] = os.path.exists(self.history_file_name)
        
        return status

    def get_token_statistics(self):
        """获取token统计信息"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "interaction_count": self.get_interaction_count(),
            "average_input_tokens_per_interaction": self.total_input_tokens / max(self.get_interaction_count(), 1),
            "average_output_tokens_per_interaction": self.total_output_tokens / max(self.get_interaction_count(), 1),
            "average_total_tokens_per_interaction": self.total_tokens / max(self.get_interaction_count(), 1)
        }


if __name__ == "__main__":
    prompt_text = "请从下面的文字中抽取出地名及其别称，并以JOSON格式输出，比如 [{'地名':'xx', '别称':'xx'}].\n\n 太原，又称龙城，是唐太宗李世民的老家。北京，又称北平，是新中国的首都。石家庄，古称常州，是赵子龙的家乡。"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama3-8b")
    parser.add_argument("--platform", type=str, default="SiliconFlow", choices=["SiliconFlow", "OpenAI", "DeepInfra"])
    args = parser.parse_args()

    llm = LLMWrapper(temperature=0, model_name="llama3-70b", platform="DeepInfra", model_kwargs={"stop": "\n"})
                                                
    print(llm.get_response(prompt_text))
