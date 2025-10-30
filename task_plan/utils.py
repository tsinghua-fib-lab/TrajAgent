from thefuzz import process
import re 

def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^A-H]{0,20}?(?:n't|not))[^A-H]{0,10}?\b(?:|is|:|be))\b)[^A-H]{0,20}?\b([A-H])\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b([A-H])\b(?![^A-H]{0,8}?(?:n't|not)[^A-H]{0,5}?(?:correct|right))[^A-H]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^([A-H])(?:\.|,|:|$)", gen)

    # simply extract the first appeared letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])([A-H])(?![a-zA-Z=])", gen)

    if res is None:
        return choice_list[choice_list.index(process.extractOne(gen, choice_list)[0])]
    
    return res.group(1)