from UniEnv.etc.settings import AUG_LIST
from result_optimize.optimize_agent import OptimizeAgent
import argparse

def get_model_aug(input_choice):
    model = input_choice["model"]
    aug_choices = []
    augmentation_methods = input_choice["augmentation_methods"]  
    for method in augmentation_methods:
        if method in AUG_LIST:
            idx = AUG_LIST.index(method) + 1  # +1 是因为索引从 0 开始
            aug_choices.append(str(idx))
    return model, aug_choices
# (city=args.city, dataset=args.source, task=args.task, aug_method = AUG_DICT[answer] )
def op_main(bs_model: str, trial_num:int, task:str, city:str, dataset:str, max_steps=2, aug_method=None):
    class Args:
        def __init__(self,
                    bs_model:str,
                    task: str,
                    dataset: str,
                    max_steps: int,
                    city: str, 
                    trial_num: int,
                    aug_method: str             
                    ):
            self.bs_model=bs_model
            self.city=city
            self.aug_method=aug_method
            self.task=task 
            self.dataset=dataset
            self.max_steps=max_steps
            self.trial_num=trial_num
    args = Args(
            city=city,
            bs_model=bs_model,
            aug_method=aug_method,
            task=task, 
            dataset=dataset,
            max_steps=max_steps,
            trial_num=trial_num
            )

    op_agent = OptimizeAgent(
                 args,
                 args.max_steps
                 )

    best_choice, suggestion, optimize_choice = op_agent.run()
    return best_choice, suggestion, optimize_choice
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)  #解析，将自然语言解析出以下参数
    parser.add_argument("--dataset", choices=["foursquare", "gowalla", "brightkite", "standard", "agentmove"])
    parser.add_argument("--city", type=str)
    parser.add_argument("--aug_method", type=str)
    parser.add_argument("--trial_num", type=int, default=2)
    parser.add_argument("--base_model", type=str)
    args = parser.parse_args()

    best_choice, suggestion, optimize_choice = op_main(trial_num=args.trial_num, city=args.city, dataset=args.dataset, task=args.task, bs_model=args.base_model, aug_method=args.aug_method)
    print("Done!!")

    