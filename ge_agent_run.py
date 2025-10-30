import os
import pandas as pd
import subprocess
import argparse
from data_generate.generate_agent import GenerateAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["FOURSQUARE", "GOWALLA", "LIBCITY", "BRIGHTKITE"])
    parser.add_argument("--target", choices=["FOURSQUARE", "GOWALLA", "LIBCITY", "BRIGHTKITE"])

    args = parser.parse_args()
    #  profile: List,react_llm
    # Occupation:{}\nGender:{}\nIncome Level:{}\nEducation Level:
    profile = ["doctor","female","middle","graduate degree"]
    agent_ge = GenerateAgent(profile=profile)
    program = agent_ge.run()
    print(program)
