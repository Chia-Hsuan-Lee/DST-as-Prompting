from tqdm import tqdm
import json
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--in_train_fn", type=str, default="./mw24/train_dials.json", help="Input training dialogues file path")
parser.add_argument("--in_test_fn", type=str, default="./mw24/test_dials.json", help="Input test dialogues file path")
parser.add_argument("--out_train_fn", type=str, default="./mw24/mw24_prompt_alldomains_train.json", help="Output training dialogues file path")
parser.add_argument("--out_test_fn", type=str, default="./mw24/mw24_prompt_alldomains_test.json", help="Output test dialogues file path")

args = parser.parse_args()
train_file_dials = args.in_train_fn
test_file_dials = args.in_test_fn
output_train_file = args.out_train_fn
output_test_file = args.out_test_fn

# Load data
train_dials = json.load(open(train_file_dials))
test_dials = json.load(open(test_file_dials))

mwoz_schema_str = (
    "| attraction : area , name , type | hotel : area , book day , book people , book stay , internet , name , parking , pricerange , stars , type "
    "| restaurant : area , book day , book people , book time , food , name , pricerange | taxi : arriveby , departure , destination , leaveat "
    "| train : arriveby , book people , day , departure , destination , leaveat"
)
def parse_slots_turn_label(slots_values):
    target_string = ""
    for slot_value in slots_values:
        if slot_value[1] != "none":
            target_string += ( slot_value[0] + "=" + slot_value[1] + ", ")
    return target_string

def parse_slots_belief_state(slots_values):
    target_string = ""
    for slot_value in slots_values:
        domain_slot, value = slot_value["slots"][0]
        target_string += f"{domain_slot}={value}, "
    return target_string

def process_and_write_dials(dials, output_file):
    with open(output_file, "w") as out:
        cur_dial_id = -1
        cur_dial = ""

        for dial in tqdm(dials):
            dial_id = dial["dialogue_idx"]

            if dial_id != cur_dial_id:
                cur_dial_id = dial_id
                prev_target_string = "none"
                cur_dial = ""

            for turn in dial["dialogue"]:
                cur_turn = ""
                tlb_target_string = ""
                turn_id = turn["turn_idx"]
                sys_uttr = turn.get("system_transcript", "")
                usr_uttr = turn.get("transcript", "")

                # Process slot values for TLB
                tlb_target_string = parse_slots_turn_label(turn["turn_label"]) or "none"
            
                # Process slot values for DST
                dst_target_string = parse_slots_belief_state(turn["belief_state"]) or "none"

                # Update current dialogue
                if sys_uttr:
                    cur_turn += f" [system] {sys_uttr}"
                    cur_dial += f" [system] {sys_uttr}"
                if usr_uttr:
                    cur_turn += f" [user] {usr_uttr}"
                    cur_dial += f" [user] {usr_uttr}"

                # Construct output line
                line = {
                    "dialogue": cur_dial,
                    "turn_pair": cur_turn,
                    "dst": dst_target_string,
                    "tlb": tlb_target_string,
                    "dialogue_id": dial_id,
                    "turn_id": turn_id,
                    "prev_dst": prev_target_string,
                    "prev_dst_turn_pair": f"[context] {prev_target_string} {cur_turn}",
                    "schema_prev_dst_turn_pair": f"[schema] {mwoz_schema_str} [context] {prev_target_string} {cur_turn}",
                    "schema_prev_dst_turn_pair_reversed": f"[context] {prev_target_string} {cur_turn} [schema] {mwoz_schema_str}"
                }

                prev_target_string = dst_target_string

                out.write(json.dumps(line) + "\n")

process_and_write_dials(train_dials, output_train_file)

process_and_write_dials(test_dials, output_test_file)
