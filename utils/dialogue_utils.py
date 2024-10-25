import string 
import re 
import copy
from collections import defaultdict

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        #exclude = [ element for element in exclude if element != "?" ]
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def conversion(pred_dst):
    fixed = defaultdict(str)

    for d_s, value in pred_dst.items():
        if "|" in value:
            values = value.split("|")
            values = [ normalize_answer(v) for v in values ]
            value = "|".join(values)
        else: 
            value = normalize_answer(value)
        fixed[d_s] = value
    return fixed

def parse_t5(pred):
    tlb = defaultdict(str)
    if pred == "none":
        return tlb
    try:
        for d_s_v in pred.strip().split(","):
            d_s_v = d_s_v.strip()
            if d_s_v:
                d_s, v = d_s_v.split("=")
                tlb[d_s] = v
    except:
        pass
    return tlb

def compare(pred_dst, gold_dst):
    temp_pred_dst = copy.deepcopy(pred_dst)
    temp_gold_dst = copy.deepcopy(gold_dst)
    all_slots = list(set(temp_pred_dst.keys()) | set(temp_gold_dst.keys()))

    for slot in all_slots:
        gold_values = temp_gold_dst[slot].split("|")  
        if temp_pred_dst[slot] not in gold_values:
            return False
    return True
