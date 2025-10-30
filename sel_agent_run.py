import argparse

from model_selection.sel_agent import SelReflectAgent

from UniEnv.etc.settings import *

def sel_main(trial_num: int, base_model: str, task:str,max_step=5, seed=42, gpu_id=0, n_cpu=3, result_path=RESULT_PATH,max_memory=4, augment_method="1000_1000", enhance=0.1):
    class Args:
        def __init__(self,
                    task: str,
                    seed: int,
                    gpu_id: int,
                    n_cpu: int,
                    result_path: str,
                    max_step: int, 
                    max_memory: int, 
                    augment_method: str,
                    enhance: str, 
                    base_model: str,
                    trial_num: int            
                    ):
                self.task=task
                self.seed=seed
                self.gpu_id=gpu_id
                self.n_cpu=n_cpu
                self.result_path=result_path
                self.max_step=max_step
                self.max_memory=max_memory
                self.augment_method=augment_method
                self.enhance=enhance
                self.base_model=base_model
                self.trial_num=trial_num
    args = Args(
                task=task,
                seed=seed,
                gpu_id=gpu_id,
                n_cpu=n_cpu,
                result_path=result_path,
                max_step=max_step,
                max_memory=max_memory,
                augment_method=augment_method,
                enhance=enhance,
                base_model=base_model,
                trial_num=trial_num
                ) 

    agent_cls = SelReflectAgent 
    # # TODO: 其他任务的base model设置
    # train_model(args, BASE_MODEL[args.task])   #获得DL Model的原始数据训练结果准确率
    # score = get_model_result(args, BASE_MODEL[args.task])

    agents = agent_cls(args = args, n_cpu = args.n_cpu,max_steps = args.max_step) 
    result_model = agents.run()
    return result_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--base_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_step", type=int)
    parser.add_argument("--trial_num", type=int, default=2)
    args = parser.parse_args()

    result_model = sel_main(trial_num=args.trial_num, task=args.task, base_model=args.base_model,max_step=args.max_step)
    print(result_model)
    
    


