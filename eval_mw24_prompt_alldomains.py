import json 
import string 
import re 
import copy
from collections import defaultdict
from utils.typo_fix import typo_fix
from utils.dialogue_utils import normalize_answer, conversion, parse_t5, compare
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_fn", type=str, default="./exps/t5base_mw24_prompt_alldomains/generated_predictions.txt", help="HF output file for the trained T5")
parser.add_argument("--gold_fn", type=str, default="./data/mw24_prompt_alldomains_test.json", help="gold test dialogues file path")
parser.add_argument("--ontology_fn", type=str, default="./data/ontology_mw24.json", help="schema of mwoz 2.4")

args = parser.parse_args()
pred_fn = args.pred_fn
gold_fn = args.gold_fn
ontology_fn = args.ontology_fn
ontology = json.load(open(ontology_fn))

mwz_ver = "2.4"
preds = open(pred_fn).readlines()
golds = open(gold_fn).readlines()



cur_dial_id = -1
t5_dst = defaultdict(str)
n_t5_dst_correct = 0
n_t5_tlb_correct = 0
for pred, gold in zip(preds, golds):
    gold = json.loads(gold)

    dial_id = gold["dialogue_id"]
    if dial_id != cur_dial_id:
        cur_dial_id = dial_id
        t5_dst = defaultdict(str)
        
    t5_tlb = parse_t5(pred.strip())
    t5_tlb = typo_fix(t5_tlb, ontology=ontology, version=mwz_ver)
    t5_tlb = conversion(t5_tlb)

    for d_s, v in t5_tlb.items():
        t5_dst[d_s] = v

    gold_dst_str  = gold["dst"]
    gold_dst = parse_t5(gold_dst_str.strip())
    gold_dst = typo_fix(gold_dst, ontology=ontology, version=mwz_ver)
    gold_dst = conversion(gold_dst)

    gold_tlb_str  = gold["tlb"]
    gold_tlb = parse_t5(gold_tlb_str.strip())
    gold_tlb = typo_fix(gold_tlb, ontology=ontology, version=mwz_ver)
    gold_tlb = conversion(gold_tlb)

    t5_dst_result = compare(t5_dst, gold_dst)
    t5_tlb_result = compare(t5_tlb, gold_tlb)

    if t5_dst_result:
        n_t5_dst_correct += 1

    if t5_tlb_result:
        n_t5_tlb_correct += 1

print("DST JGA score:", n_t5_dst_correct/len(golds))
print("TLB JGA score:", n_t5_tlb_correct/len(golds))
