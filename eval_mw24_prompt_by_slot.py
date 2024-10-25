import json
from operator import truediv 
import string 
import re 
import copy
import argparse
from collections import defaultdict
from utils.typo_fix import typo_fix
from utils.dialogue_utils import normalize_answer, conversion, parse_t5, compare

parser = argparse.ArgumentParser()
parser.add_argument("--pred_fn", type=str, default="./exps/t5base_mw24_prompt_by_slot/generated_predictions.txt", help="HF output file for the trained T5")
parser.add_argument("--gold_fn", type=str, default="./data/mw24_prompt_by_slot_test.json", help="gold test dialogues file path")
parser.add_argument("--ontology_fn", type=str, default="./data/ontology_mw24.json", help="schema of mwoz 2.4")


args = parser.parse_args()
pred_fn = args.pred_fn
gold_fn = args.gold_fn

preds = open(pred_fn).readlines()
golds = open(gold_fn).readlines()

ontology_fn = args.ontology_fn
ontology = json.load(open(ontology_fn))

mwz_ver = "2.4"

cur_dial_id = -1
cur_turn_id = -1
t5_dst = defaultdict(str)
n_t5_correct = 0
n_total_turns = 0
for pred, gold in zip(preds, golds):
    gold = json.loads(gold)
    dial_id = gold["dialogue_id"]
    dialogue = gold["dialogue"]
    turn_id = gold["turn_id"]
    domain_slot = gold["domain_slot"]

    if dial_id != cur_dial_id or turn_id != cur_turn_id:
        n_total_turns += 1
        if cur_turn_id != -1:  # avoid calculating on the first iteration
            t5_dst = typo_fix(t5_dst, ontology=ontology, version=mwz_ver)
            t5_dst = conversion(t5_dst)
            t5_dst_result = compare(t5_dst, gold_dst)
            if t5_dst_result:
                n_t5_correct += 1

        cur_dial_id = dial_id
        cur_turn_id = turn_id
        t5_dst = defaultdict(str)

    pred = normalize_answer(pred)
    if pred == "none":
        continue

    gold_dst_str = gold["dst"]
    gold_dst = parse_t5(gold_dst_str.strip())
    gold_dst = typo_fix(gold_dst, ontology=ontology, version=mwz_ver)
    gold_dst = conversion(gold_dst)

    t5_dst[domain_slot] = pred

print("DST JGA score: ", n_t5_correct / n_total_turns)
