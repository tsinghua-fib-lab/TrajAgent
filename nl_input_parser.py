import re
import os
import json
from typing import Dict, Optional, Tuple

try:
    # Prefer the project's unified LLM wrapper if available
    from data_augmentation.utils.base_llm import AnyOpenAILLM as _LLM
except Exception:  # pragma: no cover - optional dependency
    _LLM = None


TASK_ALIASES = {
    "map matching": "Map_Matching",
    "map-matching": "Map_Matching",
    "matching": "Map_Matching",
    "trajectory generation": "Trajectory_Generation",
    "generation": "Trajectory_Generation",
    "representation": "Trajectory_Representation",
    "trajectory representation": "Trajectory_Representation",
    "recovery": "Trajectory_Recovery",
    "trajectory recovery": "Trajectory_Recovery",
    "next location": "Next_Location_Prediction",
    "next location prediction": "Next_Location_Prediction",
    "nlp": "Next_Location_Prediction",
    "user linkage": "Trajectory_User_Linkage",
    "trajectory user linkage": "Trajectory_User_Linkage",
    "tte": "Travel_Time_Estimation",
    "travel time": "Travel_Time_Estimation",
    "travel time estimation": "Travel_Time_Estimation",
    "anomaly": "Trajectory_Anomaly_Detection",
    "anomaly detection": "Trajectory_Anomaly_Detection",
}


SOURCE_ALIASES = {
    "foursquare": "foursquare",
    "gowalla": "gowalla",
    "brightkite": "brightkite",
    "agentmove": "agentmove",
    "earthquake": "Earthquake",
    "tencent": "tencent",
    "chengdu": "chengdu",
    "libcity": "foursquare",  # fallback for legacy mentions
}


TARGET_ALIASES = {
    "foursquare": "foursquare",
    "gowalla": "gowalla",
    "brightkite": "brightkite",
    "agentmove": "agentmove",
    "standard": "standard",
}


CITY_CHOICES = {
    c.lower(): c
    for c in [
        "CapeTown",
        "London",
        "Moscow",
        "Mumbai",
        "Nairobi",
        "NewYork",
        "Paris",
        "SanFrancisco",
        "SaoPaulo",
        "Sydney",
        "Tokyo",
        "Unknown",
    ]
}

TASK_CHOICES = [
    "Map_Matching",
    "Trajectory_Generation",
    "Trajectory_Representation",
    "Trajectory_Recovery",
    "Next_Location_Prediction",
    "Trajectory_User_Linkage",
    "Travel_Time_Estimation",
    "Trajectory_Anomaly_Detection",
]

SOURCE_CHOICES = [
    "foursquare",
    "gowalla",
    "brightkite",
    "agentmove",
    "Earthquake",
    "tencent",
    "chengdu",
]

TARGET_CHOICES = [
    "foursquare",
    "gowalla",
    "brightkite",
    "agentmove",
    "standard",
]


def _find_first(query: str, mapping: Dict[str, str]) -> Optional[str]:
    q = query.lower()
    for k, v in mapping.items():
        if k in q:
            return v
    return None


def _find_city(query: str) -> Optional[str]:
    q = query.lower()
    for k, v in CITY_CHOICES.items():
        tokens = [k, k.replace("_", " "), k.replace("_", "")]  # handle underscores/spaces
        if any(t in q for t in tokens):
            return v
    return None


def _extract_int(query: str, patterns: Dict[str, str]) -> Dict[str, int]:
    found: Dict[str, int] = {}
    for key, pat in patterns.items():
        m = re.search(pat, query, flags=re.IGNORECASE)
        if m:
            try:
                found[key] = int(m.group(1))
            except Exception:
                pass
    return found


def parse_natural_language(query: str) -> Dict[str, object]:
    """Parse a natural-language instruction into plan args.

    Returns a partial args dict with any detected keys among:
    task, source, target, city, gpu_id, base_model, trial_num, max_step, max_epoch, memory_length
    """
    result: Dict[str, object] = {}

    task = _find_first(query, TASK_ALIASES)
    if task:
        result["task"] = task

    source = _find_first(query, SOURCE_ALIASES)
    if source:
        result["source"] = source

    target = _find_first(query, TARGET_ALIASES)
    if target:
        result["target"] = target

    city = _find_city(query)
    if city:
        result["city"] = city
        # Rule: when city specified, default source=agentmove and target=standard if not explicitly set
        result.setdefault("source", "agentmove")
        result.setdefault("target", "standard")
    else:
        # Rule: when city not specified, default target=standard if not explicitly set
        result.setdefault("target", "standard")

    # simple key phrase for base model
    m = re.search(r"model\s*[:=\s]+([\w\-\.]+)", query, flags=re.IGNORECASE)
    if m:
        result["base_model"] = m.group(1)

    # numeric fields
    nums = _extract_int(
        query,
        {
            "gpu_id": r"gpu[_\s-]*id\s*[:=\s]+(\d+)",
            "trial_num": r"trial[_\s-]*num\s*[:=\s]+(\d+)",
            "max_step": r"max[_\s-]*step\s*[:=\s]+(\d+)",
            "max_epoch": r"max[_\s-]*epoch\s*[:=\s]+(\d+)",
            "memory_length": r"memory[_\s-]*length\s*[:=\s]+(\d+)",
        },
    )
    result.update(nums)

    # lightweight fuzzy phrases
    fuzzy = _extract_int(
        query,
        {
            "gpu_id": r"gpu\s*(?:card|device)?\s*(\d+)",
            "max_epoch": r"epoch[s]?\s*(\d+)",
            "max_step": r"step[s]?\s*(\d+)",
        },
    )
    for k, v in fuzzy.items():
        result.setdefault(k, v)

    return result


def explain_parameters() -> str:
    return (
        "--task: which trajectory task to run. Choices: "
        "Map_Matching, Trajectory_Generation, Trajectory_Representation, "
        "Trajectory_Recovery, Next_Location_Prediction, Trajectory_User_Linkage, "
        "Travel_Time_Estimation, Trajectory_Anomaly_Detection.\n"
        "--source: input dataset name. Choices: foursquare, gowalla, brightkite, "
        "agentmove, Earthquake, tencent, chengdu.\n"
        "--target: target format. Choices: foursquare, gowalla, brightkite, agentmove, standard.\n"
        "--city: city name when source=agentmove. Choices: CapeTown, London, Moscow, Mumbai, Nairobi, "
        "NewYork, Paris, SanFrancisco, SaoPaulo, Sydney, Tokyo, Unknown.\n"
        "Other parameters: gpu_id, base_model, trial_num, max_step, max_epoch, memory_length."
    )


def explain_all_options() -> str:
    return (
        "Parameter semantics and supported options:\n"
        "- task: the trajectory task category to run.\n"
        "  * Map_Matching: align GPS points to road network (e.g., DeepMM, GraphMM)\n"
        "  * Trajectory_Generation: generate synthetic trajectories (ActSTD, DSTPP)\n"
        "  * Trajectory_Representation: learn trajectory embeddings (CACSR)\n"
        "  * Trajectory_Recovery: recover missing points (TrajBERT)\n"
        "  * Next_Location_Prediction: predict next location (DeepMove, RNN, FPMC, GETNext, LLM-ZS)\n"
        "  * Trajectory_User_Linkage: link trajectories to users (DPLink, MainTUL, S2TUL)\n"
        "  * Travel_Time_Estimation: estimate travel time (DeepTTE, DutyTTE, MulTTTE)\n"
        "  * Trajectory_Anomaly_Detection: detect anomalous trajectories (GMVSAE)\n"
        "- source: input dataset family.\n"
        "  * foursquare/gowalla/brightkite: check-in datasets\n"
        "  * agentmove: Foursquare split by city (requires city)\n"
        "  * Earthquake: time-series dataset\n"
        "  * tencent: map-type dataset\n"
        "  * chengdu: GPS trajectories\n"
        "- target: output/standardized format (standard recommended).\n"
        "- city: required only when source=agentmove. One of: CapeTown, London, Moscow, Mumbai, Nairobi, NewYork, Paris, SanFrancisco, SaoPaulo, Sydney, Tokyo, Unknown.\n"
        "- gpu_id/base_model/trial_num/max_step/max_epoch/memory_length: runtime and training controls.\n"
    )


def _llm_parse(query: str) -> Dict[str, object]:
    if _LLM is None:
        return {}
    model_name = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    llm = _LLM(model_name=model_name, temperature=0)
    schema = {
        "task": TASK_CHOICES,
        "source": SOURCE_CHOICES,
        "target": TARGET_CHOICES,
        "city": list(CITY_CHOICES.values()),
        "gpu_id": "int",
        "base_model": "str",
        "trial_num": "int",
        "max_step": "int",
        "max_epoch": "int",
        "memory_length": "int",
    }
    prompt = (
        "You are a strict argument parser. Read the user intent and return ONLY a compact JSON object "
        "with keys among: task, source, target, city, gpu_id, base_model, trial_num, max_step, max_epoch, memory_length. "
        "Values must respect these choices: " + json.dumps(schema) + ". If a value is unknown, omit the key.\n\n"
        "User instruction: " + query
    )
    try:
        text = llm(prompt)
        # find first JSON block
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        data = json.loads(m.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _validate(args: Dict[str, object]) -> Tuple[bool, Dict[str, object]]:
    out = dict(args)
    if "task" in out and out["task"] not in TASK_CHOICES:
        out.pop("task")
    if "source" in out and out["source"] not in SOURCE_CHOICES:
        out.pop("source")
    if "target" in out and out["target"] not in TARGET_CHOICES:
        out.pop("target")
    if "city" in out and out["city"] not in CITY_CHOICES.values():
        out.pop("city")
    return True, out


def parse_with_fallback(query: str) -> Dict[str, object]:
    """Rule-based parse first; if key params missing, use LLM to refine."""
    rule = parse_natural_language(query)
    _, rule = _validate(rule)

    need_city = rule.get("source") == "agentmove"
    missing_keys = [k for k in ("task", "source", "target") if k not in rule]
    if need_city and "city" not in rule:
        missing_keys.append("city")

    if not missing_keys:
        return rule

    llm_guess = _llm_parse(query)
    _, llm_guess = _validate(llm_guess)
    # merge: prefer rule-based; fill missing from llm
    for k, v in llm_guess.items():
        rule.setdefault(k, v)
    return rule


