import numpy as np
from myutils import *
from easydict import EasyDict as edict

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def measure_rec_quality(path_data):
    attribute_list = get_attribute_list(path_data.dataset_name)
    metrics_names = ["ndcg", "hr", "recall", "precision"]
    metrics = edict()
    for metric in metrics_names:
        metrics[metric] = {"Overall": []}
        for values in attribute_list.values():
            attribute_to_name = values[1]
            for _, name in attribute_to_name.items():
                metrics[metric][name] = []

    topk_matches = path_data.uid_topk
    test_labels = path_data.test_labels

    test_user_idxs = list(test_labels.keys())
    invalid_users = []
    for uid in test_user_idxs:
        if uid not in topk_matches: continue
        if len(topk_matches[uid]) < 10:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid], test_labels[uid]
        if len(pred_list) == 0:
            continue

        k = 0
        hit_num = 0.0
        hit_list = []
        for pid in pred_list:
            k += 1
            if pid in rel_set:
                hit_num += 1
                hit_list.append(1)
            else:
                hit_list.append(0)

        ndcg = ndcg_at_k(hit_list, k)
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        # Based on attribute
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue #Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            metrics["ndcg"][attr_name].append(ndcg)
            metrics["recall"][attr_name].append(recall)
            metrics["precision"][attr_name].append(precision)
            metrics["hr"][attr_name].append(hit)
        metrics["ndcg"]["Overall"].append(ndcg)
        metrics["recall"]["Overall"].append(recall)
        metrics["precision"]["Overall"].append(precision)
        metrics["hr"]["Overall"].append(hit)

    return metrics

def print_rec_metrics(dataset_name, metrics):
    attribute_list = get_attribute_list(dataset_name)

    print("\n---Recommandation Quality---")
    print("Average for the entire user base:", end=" ")
    for metric, values in metrics.items():
        print("{}: {:.3f}".format(metric, np.array(values["Overall"]).mean()), end=" | ")
    print("")

    for attribute_category, values in attribute_list.items():
        print("\n-Statistic with user grouped by {} attribute".format(attribute_category))
        for attribute in values[1].values():
            print("{} group".format(attribute), end=" ")
            for metric_name, groups_values in metrics.items():
                print("{}: {:.3f}".format(metric_name, np.array(groups_values[attribute]).mean()), end=" | ")
            print("")
    print("\n")

"""
Explanation metrics
"""
def topk_ETD(path_data):
    ETDs = {}
    for uid, topk in path_data.uid_topk.items():
        if uid not in path_data.test_labels: continue
        unique_path_types = set()
        for pid in topk:
            if pid not in path_data.uid_pid_explaination[uid]:
                continue
            current_path = path_data.uid_pid_explaination[uid][pid]
            path_type = get_path_type(current_path)
            unique_path_types.add(path_type)
        ETD = len(unique_path_types) / TOTAL_PATH_TYPES[path_data.dataset_name]
        ETDs[uid] = ETD
    return ETDs

def get_attribute_list(dataset_name):
    if dataset_name == "ml1m":
        attribute_list = {"Gender": [], "Age": [], "Occupation": []}
    elif dataset_name == "lastfm":
        attribute_list = {"Gender": [], "Age": []}
    else:
        print("The dataset selected doesn't exist.")
        return

    for attribute in attribute_list.keys():
        if attribute == "Gender":
            user2attribute, attribute2name = get_user2gender(dataset_name)
        elif attribute == "Age":
            user2attribute, attribute2name = get_user2age(dataset_name)
        elif attribute == "Occupation":
            user2attribute, attribute2name = get_user2occupation(dataset_name)
        else:
            print("Unknown attribute")
        attribute_list[attribute] = [user2attribute, attribute2name]
    return attribute_list

def avg_ETD(path_data):
    uid_ETDs = topk_ETD(path_data)

    attribute_list = get_attribute_list(path_data.dataset_name)
    avg_groups_ETD = {}
    groups_ETD_scores = {}
    for attribute in attribute_list.keys():
        if "Overall" not in groups_ETD_scores:
            groups_ETD_scores["Overall"] = []
        for _, attribute_label in attribute_list[attribute][1].items():
            groups_ETD_scores[attribute_label] = []

    for uid, ETD in uid_ETDs.items():
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue #Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            groups_ETD_scores[attr_name].append(ETD)
        groups_ETD_scores["Overall"].append(ETD)


    for attribute_label, group_scores in groups_ETD_scores.items():
        avg_groups_ETD[attribute_label] = np.array(group_scores).mean()


    diversity_results = edict(
        avg_groups_ETD=avg_groups_ETD,
        groups_ETD_scores=groups_ETD_scores
    )
    return diversity_results

#Extract the value of LIR for the given user item path from the LIR_matrix
def LIR_single(path_data, path):
    uid = int(path[0][-1])
    if uid not in path_data.uid_timestamp or uid not in path_data.LIR_matrix or len(path_data.uid_timestamp[uid]) <= 1: return 0.
    path_data.uid_timestamp[uid].sort()

    predicted_path = path
    interaction = int(get_interaction_id(predicted_path))
    if interaction not in path_data.uid_pid_timestamp[uid]: return 0.0
    interaction_timestamp = path_data.uid_pid_timestamp[uid][interaction]

    LIR_ans = path_data.LIR_matrix[uid][interaction_timestamp] if interaction_timestamp in path_data.LIR_matrix[uid] else 0.0

    return LIR_ans



# Returns a dict where to every uid is associated a value of LIR calculated based on his topk
def topk_LIR(path_data):
    LIR_topk = {}

    # Precompute user timestamps weigths
    LIR_matrix = path_data.LIR_matrix

    #print(len(path_data.test_labels))
    count = 0
    count_pid = 0
    for uid in path_data.test_labels:
        LIR_single_topk = []
        if uid not in LIR_matrix or uid not in path_data.uid_topk:
            continue
        for pid in path_data.uid_topk[uid]:
            count_pid += 1
            if pid not in path_data.uid_pid_explaination[uid]:
                count += 1
                continue
            predicted_path = path_data.uid_pid_explaination[uid][pid]
            interaction = int(get_interaction_id(predicted_path))
            if interaction not in path_data.uid_pid_timestamp[uid]: continue
            interaction_timestamp = path_data.uid_pid_timestamp[uid][interaction]
            LIR = LIR_matrix[uid][interaction_timestamp]
            LIR_single_topk.append(LIR)

        LIR_topk[uid] = np.array(LIR_single_topk).mean() if len(LIR_single_topk) != 0 else 0
    #print(count_pid, count)
    return LIR_topk


# Returns an avg value for the LIR of a given group
def avg_LIR(path_data, attribute_name="Gender"):
    uid_LIR_score = topk_LIR(path_data)
    attribute_list = get_attribute_list(path_data.dataset_name)
    avg_groups_LIR = {}
    groups_LIR_scores = {}
    for attribute in attribute_list.keys():
        if "Overall" not in groups_LIR_scores:
            groups_LIR_scores["Overall"] = []
        for _, attribute_label in attribute_list[attribute][1].items():
            groups_LIR_scores[attribute_label] = []

    for uid, LIR_score in uid_LIR_score.items():
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue #Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            groups_LIR_scores[attr_name].append(LIR_score)
        groups_LIR_scores["Overall"].append(LIR_score)


    for attribute_label, group_scores in groups_LIR_scores.items():
         avg_groups_LIR[attribute_label] = np.array(group_scores).mean()

    LIR = edict(
        avg_groups_LIR=avg_groups_LIR,
        groups_LIR_scores=groups_LIR_scores,
    )

    return LIR

#Extract the value of SEP for the given user item path from the SEP_matrix
def SEP_single(path_data, path):
    related_entity_type, related_entity_id = get_related_entity(path)
    SEP = path_data.SEP_matrix[related_entity_type][related_entity_id]
    return SEP


def topks_SEP(path_data):
    SEP_topk = {}

    # Precompute entity distribution
    exp_serendipity_matrix = path_data.SEP_matrix

    #Measure explanation serendipity for topk
    for uid in path_data.test_labels:
        SEP_single_topk = []
        if uid not in path_data.uid_topk: continue
        for pid in path_data.uid_topk[uid]:
            if pid not in path_data.uid_pid_explaination[uid]:
                #print("strano 2")
                continue
            path = path_data.uid_pid_explaination[uid][pid]
            related_entity_type, related_entity_id = get_related_entity(path)
            SEP = exp_serendipity_matrix[related_entity_type][related_entity_id]
            SEP_single_topk.append(SEP)
        if len(SEP_single_topk) == 0: continue
        SEP_topk[uid] = np.array(SEP_single_topk).mean()
    return SEP_topk


def avg_SEP(path_data):
    uid_SEP = topks_SEP(path_data)
    attribute_list = get_attribute_list(path_data.dataset_name)
    avg_groups_SEP = {}
    groups_SEP_scores = {}
    for attribute in attribute_list.keys():
        if "Overall" not in groups_SEP_scores:
            groups_SEP_scores["Overall"] = []
        for _, attribute_label in attribute_list[attribute][1].items():
            groups_SEP_scores[attribute_label] = []

    for uid, SEP_score in uid_SEP.items():
        for attribute in attribute_list.keys():
            if uid not in attribute_list[attribute][0]: continue
            attr_value = attribute_list[attribute][0][uid]
            if attr_value not in attribute_list[attribute][1]: continue #Few users may have the attribute missing (LASTFM)
            attr_name = attribute_list[attribute][1][attr_value]
            groups_SEP_scores[attr_name].append(SEP_score)
        groups_SEP_scores["Overall"].append(SEP_score)

    for attribute_label, group_scores in groups_SEP_scores.items():
        avg_groups_SEP[attribute_label] = np.array(group_scores).mean()

    serendipity_results = edict(
        avg_groups_SEP=avg_groups_SEP,
        groups_SEP_scores=groups_SEP_scores,
    )
    return serendipity_results

def print_expquality_metrics(dataset_name, avg_groups_LIR, avg_groups_SEP, avg_groups_ETD):
    attribute_list = get_attribute_list(dataset_name)
    metric_values = {"LIR": avg_groups_LIR, "SEP": avg_groups_SEP, "ETD": avg_groups_ETD}
    print("\n---Explanation Quality---")
    print("Average for the entire user base:", end=" ")
    for metric, values in metric_values.items():
        print("{}: {:.3f}".format(metric, values["Overall"]), end= " | ")
    print("")

    for attribute_category, values in attribute_list.items():
        attributes = values[1].values()
        print("\n-Statistic with user grouped by {} attribute".format(attribute_category))
        for attribute in attributes:
            print("{} group".format(attribute), end=" ")
            for metric, values in metric_values.items():
                print("{}: {:.3f}".format(metric, values[attribute]), end=" | ")
            print("")

