
import os
import pandas as pd
import subprocess
import argparse
from typing import List
from data_format.format_agent import FormatAgent
from UniEnv.etc.settings import *
import argparse
import json

def fm_main(trial_num:int, base_model: str, source:str, target:str, city: str):
    all_results = []
    class Args:
        def __init__(self,
                    city: str, 
                    source: str,
                    target: str,
                    fm_data_path: str,
                    fm_output_path: str,
                    fm_code_path: str,
                    fm_code_file: str,   
                    base_model: str, 
                    all_results: List,
                    ):
            self.city=city
            self.source=source
            self.target=target  
            self.fm_data_path=fm_data_path
            self.fm_output_path=fm_output_path
            self.fm_code_path=fm_code_path 
            self.fm_code_file=fm_code_file
            self.base_model=base_model
            self.all_results = all_results

    args = Args(
            source=source,
            city=city,
            target=target,  
            fm_data_path=FORMAT_DATA_PATH,
            fm_output_path=FORMAT_DATA_OUTPUT_PATH, 
            fm_code_path=FORMAT_CODE_PATH, 
            fm_code_file="test.py",
            base_model=base_model,
            all_results=all_results,
            )
    agent_fm = FormatAgent(args.source, args.target, args)

    error_cnt = 0
    for i in range(trial_num):
        try:
            program = agent_fm.run()
            program_strp = program.strip('```python\n').strip('\n```')  # Remove the first line and last line
            with open(os.path.join(args.fm_code_path,args.fm_code_file),'w') as f:
                f.write(program_strp)
            subprocess.call(['sh', './data_format/run_format/run_format.sh', args.fm_code_file, args.fm_code_path])
        except:
            error_cnt += 1
    print("Error rate:", error_cnt/trial_num)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["foursquare", "gowalla", "libcity", "brightkite", "standard", "agentmove"])
    parser.add_argument("--target", choices=["foursquare", "gowalla", "libcity", "brightkite", "standard", "agentmove"])
    parser.add_argument("--base_model", type=str, default="llama3-70b")
    parser.add_argument("--trial_num", type=int, default=2)
    parser.add_argument("--city", type=str)
    args = parser.parse_args()
    
    fm_main(trial_num=args.trial_num, base_model=args.base_model, source=args.source,target=args.target, city=args.city)

