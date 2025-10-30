import os
from typing import List, Any
from data_augmentation.utils.base_llm import AnyOpenAILLM,CityGPT
from data_generate.prompt import DAILY_PROMPT
# profile:occupation, gender, income, education
class GenerateAgent:
    def __init__(self,
                 profile: List,
                 react_llm: CityGPT = CityGPT(
                                            temperature=0,
                                            max_tokens=3000,
                                            model_name="citygpt-test:10086",
                                            openai_api_key=os.environ['OPENAI_API_KEY']),
                 ):
        self.occupation, self.gender, self.income, self.education = profile
        self.llm = react_llm
        self.daily_prompt = DAILY_PROMPT
        
    def get_global_prompt(self, day):
        user_profile = """User profile:\nOccupation:{}\nGender:{}\nIncome Level:{}\nEducation Level:{}\n""".format(self.occupation, self.gender, self.income, self.education)
        preference_prompt = user_profile + "Based on the user profile,please think about the daily routine of the user,list 3 places he\she most frequently visit,the action he most probably do in each place and the reason.Please output in the format of '1.Place1\naction:action1\nreason1'"
        user_preference = "Most frequently visit places:\n" + self.llm(preference_prompt)
        hobby_prompt = user_profile + user_preference + "Base on the user profile and user’s most frequently visit places,think about the hobbies he\she may have and the location each hobby may take place.Please output in the format of '1.Hobby:hobby1\nPlace:place1'"
        hobby_preference = "Hobbies and places:\n" + self.llm(hobby_prompt)
        traj_day = self.llm(user_profile+user_preference+hobby_preference+DAILY_PROMPT.format(day=day))
        
        return traj_day

    def run(self):
        day = "Sunday"
        traj = self.get_global_prompt(day=day)
        return traj       
    