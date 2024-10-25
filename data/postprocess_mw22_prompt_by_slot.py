#1 Use prediction file to update
#2 Use dialogue001 and dialogue 002 
import os
import json
import copy
from glob import glob
import argparse

def main(args):
    # create dummy frame as in MultiWOZ2.2 file format
    domains = ["train", "taxi", "bus", "police", "hotel", "restaurant", "attraction", "hospital"] 
    dummy_frames = []
    for domain in domains:
        dummy_frames.append({"service":domain, "state":{"slot_values":{}}})


    # Create dummy jsons to fill in later
    dummy_dial_file_json_1 = []
    dummy_dial_file_json_2 = []
    split = "test"
    target_jsons = glob(os.path.join(args.data_dir, "{}/*json".format(split)))
    for target_json_n in target_jsons:
        if target_json_n.split("/")[-1] == "schema.json":
            continue
        target_json = json.load(open(target_json_n)) 
        for dial_json in target_json:
            dial_id = dial_json["dialogue_id"] 
            dummy_dial_json = {"dialogue_id": dial_id, "turns":[]}

            for turn in dial_json["turns"]:
                turn_id = turn["turn_id"]
                if turn["speaker"] == "USER":
                    dummy_dial_json["turns"].append( {"turn_id":turn_id, "speaker":"USER", "frames":copy.deepcopy(dummy_frames)} )
                else:
                    dummy_dial_json["turns"].append(turn)

            if target_json_n.split("/")[-1] == "dialogues_001.json":
                dummy_dial_file_json_1.append(dummy_dial_json)
            elif target_json_n.split("/")[-1] == "dialogues_002.json":
                dummy_dial_file_json_2.append(dummy_dial_json)
            else:
                print("Exception! Exitting!")



    idx_lines = open(args.test_idx).readlines()
    out_lines = open(args.prediction_txt).readlines()

    # fill out dummy jsons with parsed predictions
    for _idx in range(len(idx_lines)):
        idx_list = idx_lines[_idx].strip()
        dial_json_n, dial_idx, turn_idx, frame_idx, d_name, s_name = idx_list.split("|||") 

        val = out_lines[_idx].strip()
        # For active slots, update values in the dummy jsons
        if val != "NONE":
            d_s_name = d_name + "-" + s_name 
            if dial_json_n == "dialogues_001.json":
                dummy_dial_file_json_1[int(dial_idx)]["turns"][int(turn_idx)]["frames"][int(frame_idx)]["state"]["slot_values"].update({d_s_name: [val]})
            elif dial_json_n == "dialogues_002.json":
                dummy_dial_file_json_2[int(dial_idx)]["turns"][int(turn_idx)]["frames"][int(frame_idx)]["state"]["slot_values"].update({d_s_name: [val]})
        # NONE token means the slot is non-active. Skip the updating option
        else:
            continue

    if not os.path.exists(args.out_dir):
                        os.mkdir(args.out_dir)

    # Create dummy json files for evaluation
    for n in ["dialogues_001", "dialogues_002"]:
        dummy_out_file = open(os.path.join(args.out_dir, "dummy_out_{n}.json".format(n=n)), "w")
        if n == "dialogues_001":
            json.dump(dummy_dial_file_json_1, dummy_out_file, indent=4)
        elif n == "dialogues_002":
            json.dump(dummy_dial_file_json_2, dummy_out_file, indent=4)
        dummy_out_file.close()
        
    

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./MultiWOZ_2.2/")
    parser.add_argument("--out_dir", type=str, default="./MultiWOZ_2.2/dummy/")

    parser.add_argument("--test_idx", type=str, default="./MultiWOZ_2.2/mw24_prompt_by_slot_test.idx")
    parser.add_argument("--prediction_txt", type=str, default="")
    args = parser.parse_args()

    main(args)
