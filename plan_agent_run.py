import os
from da_agent_run import da_main
from fm_agent_run import fm_main 
from sel_agent_run import sel_main
from op_agent_run import op_main 
from param_agent_run import pa_main
from preprocess.traj_preprocess import pre_main_checkin
from preprocess.traj_preprocess_gps import pre_main_gps
from task_plan.plan_agent import PlanAgent
from UniEnv.etc.all_config import data_config, model_config
from UniEnv.etc.settings import *
from nl_input_parser import parse_with_fallback
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["Map_Matching", "Trajectory_Generation", "Trajectory_Representation","Trajectory_Recovery","Next_Location_Prediction","Trajectory_User_Linkage","Travel_Time_Estimation","Trajectory_Anomaly_Detection"])
    parser.add_argument("--source", choices=["foursquare", "gowalla", "brightkite", "agentmove","Earthquake","tencent","chengdu"])
    parser.add_argument("--target", choices=["foursquare", "gowalla", "brightkite",  "agentmove", "standard"])  #推荐用 standard 走通整个流程
    parser.add_argument("--city", type=str, default="Unknown", choices=['CapeTown','London', 'Moscow', 'Mumbai', 'Nairobi','NewYork','Paris', 'SanFrancisco', 'SaoPaulo', 'Sydney','Tokyo', 'Unknown'])  # agentmove需要指定城市，其他数据集默认此参数为Unknown
    parser.add_argument("--gpu_id", type=int, default=3) 
    parser.add_argument("--base_model", type=str, default="llama3-70b") 
    parser.add_argument("--trial_num", type=int, default=1)  
    parser.add_argument("--max_step", type=int, default=5)  #think+act+reflect总轮数
    parser.add_argument("--max_epoch", type=int, default=4)  #模型训练轮数
    parser.add_argument("--memory_length", type=int, default=1) 
    parser.add_argument("--query", type=str, default=None, help="natural language instruction to parse into arguments")
    args = parser.parse_args()

    # Natural language override/merge
    if args.query:
        nl = parse_with_fallback(args.query)
        for k, v in nl.items():
            # only update known keys
            if hasattr(args, k) and v is not None:
                setattr(args, k, v)
    if DATA_TYPE[args.source] == 'checkin':
        op_agent = PlanAgent(
                    task=args.task,
                    dataset=args.source,
                    )
        answer = op_agent.run()
    ## 其他种类数据没有数据增强
    else:
        answer = 'B'
    AUG_DICT = {'A':'DA','B':'PO','C':'Joint'}
    # use_da, use_pa, use_joint
    sel_model = sel_main(trial_num=args.trial_num, task=args.task, base_model=args.base_model, max_step=args.max_step)
    print(f"The name you choose:{sel_model}")
    fm_main(trial_num=args.trial_num, base_model=args.base_model, source=args.source,target=args.target, city=args.city)
    # 默认从fm_main()处理后的target.csv中读取数据，即处理成standard格式的源数据集.暂时不支持处理map类型的数据
    if DATA_TYPE[args.source] in ['checkin', 'gps']:
        eval(f"pre_main_{DATA_TYPE[args.source]}")(city=args.city, dataset=args.source, model=sel_model, data_config=data_config, model_config=model_config, base_model=args.base_model, memory_length=args.memory_length)
    if answer=='A':
    # TODO: plan模块在多种优化模式之间进行选择，包括da,pa,joint三种的排列组合。da可选是否使用算子参数搜索优化（pa_da).默认pa_da=True
        best_augment, result_score = da_main(gpu_id=args.gpu_id, trial_num=args.trial_num, model = sel_model, dataset = args.source, city = args.city, task = args.task, XR=False, memory_length=args.memory_length, max_epoch=args.max_epoch, max_step=args.max_step, base_model=args.base_model, pa_da=True)
    elif answer=='B':
        result_param, result_score = pa_main(memory_length=args.memory_length, trial_num = args.trial_num, XR=False,  gpu_id = args.gpu_id, model = sel_model, dataset = args.source, city = args.city, task = args.task, max_step=args.max_step, base_model=args.base_model, max_epoch=args.max_epoch, aug_methods_name = "1000_1000")
    elif answer=='C':
        best_augment, result_score = da_main(trial_num=args.trial_num, model = sel_model, dataset = args.source, city = args.city, task = args.task, XR=False, memory_length=args.memory_length, max_epoch=args.max_epoch, max_step=args.max_step, base_model=args.base_model, pa_da=True)
        result_param, result_score = pa_main(base_model=args.base_model, memory_length=args.memory_length, trial_num = args.trial_num, XR=False,  gpu_id = args.gpu_id, model = sel_model, dataset = args.source, city = args.city, task = args.task, max_step=args.max_step, max_epoch=args.max_epoch, aug_methods_name = best_augment)
    best_choice, suggestion, optimize_choice = op_main(trial_num=args.trial_num, city=args.city, dataset=args.source, task=args.task, bs_model=args.base_model, aug_method= AUG_DICT[answer])
    print(f"Best experiment of all experiments: {best_choice}\nExperience you have learned: {suggestion}\nOptimize suggestion: {optimize_choice}")

    
    
    
