from UniEnv.etc.model_data_config import MainTUL, DPLink, LibCity, GETNext, CACSR, S2TUL, TrajBERT, ActSTD, LLM
from UniEnv.etc.common_data_config import foursquare, brightkite, chengdu, porto, agentmove

model_config = {
    "MainTUL": MainTUL.config,
    "DPLink": DPLink.config,
    "LibCity": LibCity.config,
    "GETNext": GETNext.config,
    "CACSR": CACSR.config,
    "S2TUL": S2TUL.config,
    "TrajBERT": TrajBERT.config,
    "ActSTD": ActSTD.config,
    "LLM": LLM.config
}
data_config = {
    "foursquare": foursquare.config,
    "brightkite": brightkite.config,
    "porto": porto.config,
    "chengdu": chengdu.config,
    "agentmove": agentmove.config,
    
}
