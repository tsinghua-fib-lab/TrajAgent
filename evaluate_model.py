# evaluate_model.py

import os
import sys
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import concurrent.futures
from openai import OpenAI
import time

# Initialize OpenAI client at the top level
client = OpenAI(
    api_key="1717465539339440217",
    base_url="https://aigc.sankuai.com/v1/openai/native"
)

def setup_model(model_name):
    """Initialize tokenizer and LLM model."""
    print(f"\nSetting up model: {model_name}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Tokenizer loaded.")
    
    print("Initializing LLM...")
    if "phi-4" in model_name.lower() or "phi-3.5-mini" in model_name.lower() or "phi-3-mini" in model_name.lower() or "phi-3-small" in model_name.lower() or "phi-3-medium" in model_name.lower(): 
        for tensor_parallel_size in [2,1]:
            try:
                llm = LLM(
                    model=model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    max_model_len=None,
                    trust_remote_code=True,
                    enforce_eager=True
                )
                print(f"LLM initialized with tensor_parallel_size={tensor_parallel_size}.")
                return tokenizer, llm
            except Exception as e:
                print(f"Failed to initialize with tensor_parallel_size={tensor_parallel_size}: {e}")
    else:
        for tensor_parallel_size in [4,2,1]:
            try:
                llm = LLM(
                    model=model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    max_model_len=None,
                    trust_remote_code=True,
                    enforce_eager=True
                )
                print(f"LLM initialized with tensor_parallel_size={tensor_parallel_size}.")
                return tokenizer, llm
            except Exception as e:
                print(f"Failed to initialize with tensor_parallel_size={tensor_parallel_size}: {e}")
    raise RuntimeError(f"Failed to initialize LLM for model {model_name} with all tensor_parallel_sizes.")

def get_model_response(llm, tokenizer, question):
    """Get response from the local model for a single question."""
    prompt = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=3)
    
    outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)
    response = outputs[0].outputs[0].text.strip()
    print(response)
    return response

def get_completion(model, text):
    """
    向指定模型发送请求，处理响应或错误情况。
    如果遇到错误码450，返回'A'作为默认答案。
    """
    while True:
        try:
            result = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": text
                }],
                stream=False,
                temperature=0.1
            )
            content = result.choices[0].message.content
            print(f"Success: {model}\n{content}")
            return content
        except Exception as e:
            error_str = str(e)
            # Check for error code 450
            if "Error code: 450" in error_str or "Error code: 400" in error_str or "Error code: 451" in error_str:
                print(f"Received error for {model}, returning 'A' as default answer")
                return "A"
            else:
                print(f"Failure: {model}\nError: {e}\nRetrying in 5 seconds...")
                time.sleep(5)  # 等待后重试

def calculate_accuracy(predictions, ground_truth):
    """Calculate accuracy by comparing the first letter of predictions with ground truth."""
    total = len(ground_truth)
    if total == 0:
        return 0
        
    correct = 0
    for pred, true in zip(predictions, ground_truth):
        if pred and true:  # Ensure both are not empty
            if pred[0].upper() == true.strip()[0].upper():
                correct += 1
                
    return (correct / total) * 100

def process_dataset(dataset_path, llm, tokenizer):
    """Process a single dataset using the local model and return accuracy and total number of questions."""
    print(f"\nProcessing dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print("\nDataset columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    predictions = []
    total_questions = len(df)
    for i, question in enumerate(df['Question'].values, 1):
        try:
            print(f"Processing question {i}/{total_questions}")
            response = get_model_response(llm, tokenizer, question)
            predictions.append(response)
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            predictions.append("")
    
    accuracy = calculate_accuracy(predictions, df['Answer'].values)
    return accuracy, total_questions

def process_question_api(model_name, question, question_idx, total_questions):
    """Wrapper function to process a single question using the API."""
    try:
        print(f"Processing question {question_idx + 1}/{total_questions}")
        response = get_completion(model_name, question)
        return response
    except Exception as e:
        print(f"Error processing question {question_idx + 1}: {e}")
        return ""

def process_dataset_api(dataset_path, model_name):
    """Process a single dataset using the API with multi-threading and return accuracy and total number of questions."""
    print(f"\nProcessing dataset with API: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print("\nDataset columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    total_questions = len(df)
    max_workers = min(10, total_questions)
    
    # 创建固定大小的列表来存储有序结果
    predictions = [None] * total_questions
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务时包含问题索引和总问题数
        future_to_idx = {
            executor.submit(
                process_question_api, 
                model_name, 
                question, 
                idx,
                total_questions
            ): idx 
            for idx, question in enumerate(df['Question'].values)
        }
        
        # 按完成顺序处理结果，但保持原始顺序存储
        for future in concurrent.futures.as_completed(future_to_idx):
            original_idx = future_to_idx[future]
            try:
                response = future.result()
                predictions[original_idx] = response
            except Exception as e:
                print(f"Error processing question {original_idx + 1}: {e}")
                predictions[original_idx] = ""
    
    # 确保所有问题都被处理
    assert None not in predictions, "Some questions were not processed"
    
    accuracy = calculate_accuracy(predictions, df['Answer'].values)
    return accuracy, total_questions

def evaluate_model(model_name, task_numbers):
    """Evaluate the specified model on the given tasks."""
    os.makedirs('paper_results', exist_ok=True)
    model_basename = os.path.basename(model_name)
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    result_file = f"paper_results/{model_basename}_{timestamp}.txt"
    
    # Check if the model is an API-based model
    if 'gpt' in model_name.lower() or 'glm' in model_name.lower() or 'claude' in model_name.lower() or 'max' in model_name.lower() or 'plus-latest' in model_name.lower() or 'turbo' in model_name.lower() or 'abab' in model_name.lower() or 'moonshot' in model_name.lower() or 'doubao' in model_name.lower() or 'deepseek' in model_name.lower():
        with open(result_file, 'w', encoding='utf-8') as f:
            for task_num in task_numbers:
                dataset_path = f"new_benchmark/{task_num}_questions_answers.csv"  # Ensure .csv extension
                try:
                    accuracy, total_questions = process_dataset_api(dataset_path, model_name)
                    result_line = f"{task_num}: {accuracy:.2f}%, {total_questions}\n"
                    f.write(result_line)
                    print(f"Model: {model_name}, Task {task_num}: {accuracy:.2f}%, {total_questions} questions")
                except Exception as e:
                    error_msg = f"{task_num}: Error - {str(e)}\n"
                    f.write(error_msg)
                    print(f"Error processing task {task_num} for model {model_name}: {e}")        
    else: 
        tokenizer, llm = setup_model(model_name)
        with open(result_file, 'w', encoding='utf-8') as f:
            for task_num in task_numbers:
                dataset_path = f"new_benchmark/{task_num}_questions_answers.csv"
                try:
                    accuracy, total_questions = process_dataset(dataset_path, llm, tokenizer)
                    result_line = f"{task_num}: {accuracy:.2f}%, {total_questions}\n"
                    f.write(result_line)
                    print(f"Model: {model_name}, Task {task_num}: {accuracy:.2f}%, {total_questions} questions")
                except Exception as e:
                    error_msg = f"{task_num}: Error - {str(e)}\n"
                    f.write(error_msg)
                    print(f"Error processing task {task_num} for model {model_name}: {e}")        

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    if 'gemma' in model_name.lower():
        os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'
    # task_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 23, 25, 26, 27, 28, 29, 30, 38, 40, 47,51,52,53,54,55,56,57,58]
    # task_numbers=[51,52,53,54,55,56,57,58]
    # task_numbers = [28,29,30] 
    # task_numbers = [29,101] 
    # task_numbers=[66]
    task_numbers = [63,64,1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 23, 25, 26, 27, 28, 29, 30,31,34,35,59,60,61,62, 38, 40, 47,51,52,53,54,55,56,57,58]
    # task_numbers = [57,58]
    # task_numbers = [14, 15, 16, 23, 25, 26, 27, 28, 29, 30,31,34,35,59,60,61,62, 38, 40, 47,51,52,53,54,55,56,57,58]
    evaluate_model(model_name, task_numbers)