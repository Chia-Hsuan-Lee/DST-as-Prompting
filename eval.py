import json
import os
import argparse
from glob import glob
import collections
import numpy as np

ALL_SERVICES = "#ALL_SERVICES"
SEEN_SERVICES = "#SEEN_SERVICES"
UNSEEN_SERVICES = "#UNSEEN_SERVICES"

# Name of the file containing all predictions and their corresponding frame
# metrics.
PER_FRAME_OUTPUT_FILENAME = "dialogues_and_metrics.json"


F1Scores = collections.namedtuple("F1Scores", ["f1", "precision", "recall"])

# Evaluation and other relevant metrics for DSTC8 Schema-guided DST.
# (1) Active intent accuracy.
ACTIVE_INTENT_ACCURACY = "active_intent_accuracy"
# (2) Slot tagging F1.
SLOT_TAGGING_F1 = "slot_tagging_f1"
SLOT_TAGGING_PRECISION = "slot_tagging_precision"
SLOT_TAGGING_RECALL = "slot_tagging_recall"
# (3) Requested slots F1.
REQUESTED_SLOTS_F1 = "requested_slots_f1"
REQUESTED_SLOTS_PRECISION = "requested_slots_precision"
REQUESTED_SLOTS_RECALL = "requested_slots_recall"
# (4) Average goal accuracy.
AVERAGE_GOAL_ACCURACY = "average_goal_accuracy"
AVERAGE_CAT_ACCURACY = "average_cat_accuracy"
AVERAGE_NONCAT_ACCURACY = "average_noncat_accuracy"
# (5) Joint goal accuracy.
JOINT_GOAL_ACCURACY = "joint_goal_accuracy"
JOINT_CAT_ACCURACY = "joint_cat_accuracy"
JOINT_NONCAT_ACCURACY = "joint_noncat_accuracy"

NAN_VAL = "NA"


def compute_f1(list_ref, list_hyp):
    """Compute F1 score from reference (grouth truth) list and hypothesis list.
    Args:
        list_ref: List of true elements.
        list_hyp: List of postive (retrieved) elements.
    Returns:
        A F1Scores object containing F1, precision, and recall scores.
    """

    ref = collections.Counter(list_ref)
    hyp = collections.Counter(list_hyp)
    true = sum(ref.values())
    positive = sum(hyp.values())
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:  # The F1-score is defined to be 0 if both precision and recall are 0.
        f1 = 0.0

    return F1Scores(f1=f1, precision=precision, recall=recall)


def fuzzy_string_match(str_ref, str_hyp):
    """Returns fuzzy string similarity score in range [0.0, 1.0]."""

    # The higher the score, the higher the similarity between the two strings.
    return fuzz.token_sort_ratio(str_ref, str_hyp) / 100.0


def noncat_slot_value_match(str_ref_list, str_hyp, use_fuzzy_match):
    """Calculate non-categorical slots correctness.
    Args:
        str_ref_list: a list of reference strings.
        str_hyp: the hypothesis string.
        use_fuzzy_match: whether to use fuzzy string matching.
    Returns:
        score: The highest fuzzy string match score of the references and hypotheis.
    """
    score = 0.0
    for str_ref in str_ref_list:
        if not use_fuzzy_match:
            match_score = float(str_ref == str_hyp)
        else:
            match_score = fuzzy_string_match(str_ref, str_hyp)
        score = max(score, match_score)
    return score


def compare_slot_values(slot_values_ref, slot_values_hyp, service,
                                                use_fuzzy_match):
    """Compare and get correctness of goal state's slot_values.
    Args:
        slot_values_ref: goal state slot_values from reference (ground truth).
        slot_values_hyp: goal state slot_values from hypothesis (prediction).
        service: a service data structure in the schema. We use it to obtain the
            list of slots in the service and infer whether a slot is categorical.
        use_fuzzy_match: whether to use fuzzy string matching for non-categorical
            slot values.
    Returns:
        (list_cor, slot_active, slot_cat)
        list_cor: list of corectness scores, each corresponding to one slot in the
                service. The score is a float either 0.0 or 1.0 for categorical slot,
                and in range [0.0, 1.0] for non-categorical slot.
        slot_active: list indicating whether the element in list_cor corresponds to
                an active ground-truth slot.
        slot_cat: list indicating whether the element in list_cor corresponds to a
                categorical slot.
    """
    list_cor = []
    slot_active = []
    slot_cat = []

    for slot in service["slots"]:
        slot_name = slot["name"]
        slot_cat.append(slot["is_categorical"])

        if slot_name in slot_values_ref:    # REF=active
            slot_active.append(True)
            if slot_name in slot_values_hyp:    # HYP=active, apply matching
                value_ref_list = slot_values_ref[slot_name]
                value_hyp = slot_values_hyp[slot_name][0]
                if slot["is_categorical"]:
                    cor = float(value_ref_list[0] == value_hyp)
                else:
                    cor = noncat_slot_value_match(value_ref_list, value_hyp,
                                                                                use_fuzzy_match)

                list_cor.append(cor)
            else:  # HYP=off
                list_cor.append(0.0)
        else:  # REF=off
            slot_active.append(False)
            if slot_name in slot_values_hyp:    # HYP=active
                list_cor.append(0.0)
            else:  # HYP=off
                list_cor.append(1.0)

    assert len(list_cor) == len(service["slots"])
    assert len(slot_active) == len(service["slots"])
    assert len(slot_cat) == len(service["slots"])
    return list_cor, slot_active, slot_cat

def get_average_and_joint_goal_accuracy(frame_ref, frame_hyp, service,
                                                                                use_fuzzy_match):
    """Get average and joint goal accuracies of a frame.
    Args:
        frame_ref: single semantic frame from reference (ground truth) file.
        frame_hyp: single semantic frame from hypothesis (prediction) file.
        service: a service data structure in the schema. We use it to obtain the
            list of slots in the service and infer whether a slot is categorical.
        use_fuzzy_match: whether to use fuzzy string matching for comparing
            non-categorical slot values.
    Returns:
        goal_acc: a dict whose values are average / joint
                all-goal / categorical-goal / non-categorical-goal accuracies.
    """
    goal_acc = {}

    list_acc, slot_active, slot_cat = compare_slot_values(
            frame_ref["state"]["slot_values"], frame_hyp["state"]["slot_values"],
            service, use_fuzzy_match)

    # (4) Average goal accuracy.
    active_acc = [acc for acc, active in zip(list_acc, slot_active) if active]
    goal_acc[AVERAGE_GOAL_ACCURACY] = np.mean(
            active_acc) if active_acc else NAN_VAL
    # (4-a) categorical.
    active_cat_acc = [
            acc for acc, active, cat in zip(list_acc, slot_active, slot_cat)
            if active and cat
    ]
    goal_acc[AVERAGE_CAT_ACCURACY] = (
            np.mean(active_cat_acc) if active_cat_acc else NAN_VAL)
    # (4-b) non-categorical.
    active_noncat_acc = [
            acc for acc, active, cat in zip(list_acc, slot_active, slot_cat)
            if active and not cat
    ]
    goal_acc[AVERAGE_NONCAT_ACCURACY] = (
            np.mean(active_noncat_acc) if active_noncat_acc else NAN_VAL)

    # (5) Joint goal accuracy.
    goal_acc[JOINT_GOAL_ACCURACY] = np.prod(list_acc) if list_acc else NAN_VAL
    # (5-a) categorical.
    cat_acc = [acc for acc, cat in zip(list_acc, slot_cat) if cat]
    goal_acc[JOINT_CAT_ACCURACY] = np.prod(cat_acc) if cat_acc else NAN_VAL
    # (5-b) non-categorical.
    noncat_acc = [acc for acc, cat in zip(list_acc, slot_cat) if not cat]
    goal_acc[JOINT_NONCAT_ACCURACY] = np.prod(
            noncat_acc) if noncat_acc else NAN_VAL

    return goal_acc

def get_service_set(schema_path):
    """Get the set of all services present in a schema."""
    service_set = set()
    #with tf.gfile.GFile(schema_path) as f:
    with open(schema_path) as f:
        schema = json.load(f)
        for service in schema:
            service_set.add(service["service_name"])
    return service_set


def get_in_domain_services(schema_path_1, schema_path_2):
    """Get the set of common services between two schemas."""
    return get_service_set(schema_path_1) & get_service_set(schema_path_2)


def get_dataset_as_dict(file_path_patterns):
    """Read the DSTC8 json dialog data as dictionary with dialog ID as keys."""
    dataset_dict = {}
    if isinstance(file_path_patterns, list):
        list_fp = file_path_patterns
    else:
        list_fp = sorted(glob(file_path_patterns))
    for fp in list_fp:
        if PER_FRAME_OUTPUT_FILENAME in fp:
            continue
        with open(fp) as f:
            data = json.load(f)
            if isinstance(data, list):
                for dial in data:
                    dataset_dict[dial["dialogue_id"]] = dial
            elif isinstance(data, dict):
                dataset_dict.update(data)
    return dataset_dict


def get_metrics(dataset_ref, dataset_hyp, service_schemas, in_domain_services):
    """Calculate the DSTC8 metrics.
    Args:
        dataset_ref: The ground truth dataset represented as a dict mapping dialogue
            id to the corresponding dialogue.
        dataset_hyp: The predictions in the same format as `dataset_ref`.
        service_schemas: A dict mapping service name to the schema for the service.
        in_domain_services: The set of services which are present in the training
            set.
    Returns:
        A dict mapping a metric collection name to a dict containing the values
        for various metrics. Each metric collection aggregates the metrics across
        a specific set of frames in the dialogues.
    """
    # Metrics can be aggregated in various ways, eg over all dialogues, only for
    # dialogues containing unseen services or for dialogues corresponding to a
    # single service. This aggregation is done through metric_collections, which
    # is a dict mapping a collection name to a dict, which maps a metric to a list
    # of values for that metric. Each value in this list is the value taken by
    # the metric on a frame.
    metric_collections = collections.defaultdict(
            lambda: collections.defaultdict(list))

    # Ensure the dialogs in dataset_hyp also occur in dataset_ref.
    assert set(dataset_hyp.keys()).issubset(set(dataset_ref.keys()))

    # Store metrics for every frame for debugging.
    per_frame_metric = {}
    for dial_id, dial_hyp in dataset_hyp.items():
        dial_ref = dataset_ref[dial_id]
        '''
        if set(dial_ref["services"]) != set(dial_hyp["services"]):
            raise ValueError(
                    "Set of services present in ground truth and predictions don't match "
                    "for dialogue with id {}".format(dial_id))
        '''
        joint_metrics = [
                JOINT_GOAL_ACCURACY, JOINT_CAT_ACCURACY,
                JOINT_NONCAT_ACCURACY
        ]
        for turn_id, (turn_ref, turn_hyp) in enumerate(
                zip(dial_ref["turns"], dial_hyp["turns"])):
            metric_collections_per_turn = collections.defaultdict(
                    lambda: collections.defaultdict(lambda: 1.0))
            if turn_ref["speaker"] != turn_hyp["speaker"]:
                raise ValueError(
                        "Speakers don't match in dialogue with id {}".format(dial_id))

            # Skip system turns because metrics are only computed for user turns.
            if turn_ref["speaker"] != "USER":
                continue

            hyp_frames_by_service = {
                    frame["service"]: frame for frame in turn_hyp["frames"]
            }

            # Calculate metrics for each frame in each user turn.
            for frame_ref in turn_ref["frames"]:
                service_name = frame_ref["service"]
                if service_name not in hyp_frames_by_service:
                    raise ValueError(
                            "Frame for service {} not found in dialogue with id {}".format(
                                    service_name, dial_id))
                service = service_schemas[service_name]
                frame_hyp = hyp_frames_by_service[service_name]
                
                goal_accuracy_dict = get_average_and_joint_goal_accuracy(
                        frame_ref, frame_hyp, service, args.use_fuzzy_match)
                #print(frame_ref)
                #print(frame_hyp)
                #print(goal_accuracy_dict)
                #print("-----------------------------------------")

                frame_metric = {}
                frame_metric.update(goal_accuracy_dict)

                frame_id = "{:s}-{:03d}-{:s}".format(dial_id, turn_id,
                                                                                         frame_hyp["service"])
                per_frame_metric[frame_id] = frame_metric
                # Add the frame-level metric result back to dialogues.
                frame_hyp["metrics"] = frame_metric

                # Get the domain name of the service.
                domain_name = frame_hyp["service"].split("_")[0]
                domain_keys = [ALL_SERVICES, frame_hyp["service"], domain_name]
                if frame_hyp["service"] in in_domain_services:
                    domain_keys.append(SEEN_SERVICES)
                else:
                    domain_keys.append(UNSEEN_SERVICES)
                for domain_key in domain_keys:
                    for metric_key, metric_value in frame_metric.items():
                        if metric_value != NAN_VAL:
                            if args.joint_acc_across_turn and metric_key in joint_metrics:
                                metric_collections_per_turn[domain_key][
                                        metric_key] *= metric_value
                            else:
                                metric_collections[domain_key][metric_key].append(metric_value)
            if args.joint_acc_across_turn:
                # Conduct multiwoz style evaluation that computes joint goal accuracy
                # across all the slot values of all the domains for each turn.
                for domain_key in metric_collections_per_turn:
                    for metric_key, metric_value in metric_collections_per_turn[
                            domain_key].items():
                        metric_collections[domain_key][metric_key].append(metric_value)
    all_metric_aggregate = {}
    for domain_key, domain_metric_vals in metric_collections.items():
        domain_metric_aggregate = {}
        for metric_key, value_list in domain_metric_vals.items():
            if value_list:
                # Metrics are macro-averaged across all frames.
                domain_metric_aggregate[metric_key] = float(np.mean(value_list))
            else:
                domain_metric_aggregate[metric_key] = NAN_VAL
        all_metric_aggregate[domain_key] = domain_metric_aggregate
    return all_metric_aggregate, per_frame_metric


def main(args):
    
    in_domain_services = get_in_domain_services(
        os.path.join(args.data_dir, args.eval_set, "schema.json"),
        os.path.join(args.data_dir, "train", "schema.json"))


    with open(os.path.join(args.data_dir, args.eval_set, "schema.json")) as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service

    dataset_ref = get_dataset_as_dict(
            os.path.join(args.data_dir, args.eval_set, "dialogues_*.json"))
    dataset_hyp = get_dataset_as_dict(
            os.path.join(args.prediction_dir, "*.json"))

    print("len(dataset_hyp)=%d, len(dataset_ref)=%d", len(dataset_hyp),
                                    len(dataset_ref))
    if not dataset_hyp or not dataset_ref:
        raise ValueError("Hypothesis and/or reference dataset are empty!")

    all_metric_aggregate, _ = get_metrics(dataset_ref, dataset_hyp, eval_services,
                                                                                in_domain_services)
    print("Dialog metrics: %s", str(all_metric_aggregate[ALL_SERVICES]))

    with open(args.output_metric_file, "w") as f:
        json.dump(
                all_metric_aggregate,
                f,
                indent=2,
                separators=(",", ": "),
                sort_keys=True)
        
    # Write the per-frame metrics values with the corrresponding dialogue frames.
    with open(      
            os.path.join(args.prediction_dir, PER_FRAME_OUTPUT_FILENAME), "w") as f:
        json.dump(dataset_hyp, f, indent=2, separators=(",", ": "))
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./MultiWOZ_2.2")
    parser.add_argument("--prediction_dir",  default="./MultiWOZ_2.2/dummy/")
    parser.add_argument("--eval_set",  default="test")
    parser.add_argument("--output_metric_file", default="./MultiWOZ_2.2/dummy/dummy_score")
    parser.add_argument("--joint_acc_across_turn", default=True, type=bool)      
    parser.add_argument("--use_fuzzy_match", default=False, type=bool)
    args = parser.parse_args()

    main(args)

