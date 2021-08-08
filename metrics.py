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
    uid2gender, gender2name = get_user2gender(path_data.dataset_name)
    topk_matches = path_data.uid_topk
    test_labels = path_data.test_labels
    metrics = edict(
        ndcg=edict(
            Male=[],
            Female=[],
            Overall=[]
        ),
        hr=edict(
            Male=[],
            Female=[],
            Overall=[]
        ),
        precision=edict(
            Male=[],
            Female=[],
            Overall=[]
        ),
        recall=edict(
            Male=[],
            Female=[],
            Overall=[]
        ),

    )

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
        attribute_val = uid2gender[uid]
        gender = gender2name[attribute_val]
        all = "Overall"

        # According to gender
        metrics.ndcg[gender].append(ndcg)
        metrics.recall[gender].append(recall)
        metrics.precision[gender].append(precision)
        metrics.hr[gender].append(hit)

        # General
        metrics.ndcg[all].append(ndcg)
        metrics.hr[all].append(hit)
        metrics.recall[all].append(recall)
        metrics.precision[all].append(precision)

    return metrics

def print_rec_metrics(metrics):
    print("Recommandation quality:")
    for metric_name, groups_values in metrics.items():
        for group_name, group_values in groups_values.items():
            average = np.mean(group_values)
            print("Average {} {} noOfUsers {}: {:.3f} ".format(metric_name, group_name, len(group_values), average), end = ' | ')
        print()

"""
Explaination metrics
"""
def topk_diversity_score(path_data):
    diversity_scores = {}
    for uid, topk in path_data.uid_topk.items():
        if uid not in path_data.test_labels: continue
        unique_path_types = set()
        for pid in topk:
            if pid not in path_data.uid_pid_explaination[uid]:
                #print("strano 3")
                continue
            current_path = path_data.uid_pid_explaination[uid][pid]
            path_type = get_path_type(current_path)
            unique_path_types.add(path_type)
        diversity_score = len(unique_path_types) / TOTAL_PATH_TYPES[path_data.dataset_name]
        diversity_scores[uid] = diversity_score
    return diversity_scores

def avg_diversity_score(path_data, attribute_name="Gender"):
    uid_diversity_scores = topk_diversity_score(path_data)
    if attribute_name == "Gender":
        user2attribute, attribute2name = get_user2gender(path_data.dataset_name)
    elif attribute_name == "Age":
        user2attribute, attribute2name = get_user2age()
    elif attribute_name == "Occupation":
        user2attribute, attribute2name = get_user2occupation()
    else:
        print("The attribute selected doesn't exist.")
        return
    avg_groups_explaination_diversity = {}
    groups_explaination_diversity_scores = {"Overall": []}

    for _, attribute_label in attribute2name.items():
        groups_explaination_diversity_scores[attribute_label] = []

    for uid, diversity_score in uid_diversity_scores.items():
        attr_value = user2attribute[uid]
        attr_name = attribute2name[attr_value]
        groups_explaination_diversity_scores[attr_name].append(diversity_score)
        groups_explaination_diversity_scores["Overall"].append(diversity_score)

    for attribute_label, group_explaination_diversity_scores in groups_explaination_diversity_scores.items():
        avg_groups_explaination_diversity[attribute_label] = np.array(group_explaination_diversity_scores).mean()

    avg_groups_explaination_diversity["Overall"] = np.array(groups_explaination_diversity_scores["Overall"]).mean()

    diversity_results = edict(
        avg_groups_explaination_diversity=avg_groups_explaination_diversity,
        groups_explaination_diversity_scores=groups_explaination_diversity_scores
    )
    return diversity_results

#Extract the value of ETR for the given user item path from the ETR_matrix
def explantion_time_relevance_single(path_data, path):
    uid = int(path[0][-1])
    path_data.uid_timestamp[uid].sort()

    predicted_path = path
    interaction = int(get_interaction_id(predicted_path))
    interaction_timestamp = path_data.uid_pid_timestamp[uid][interaction]

    time_relevance_ans = path_data.ETR_matrix[uid][interaction_timestamp]

    return time_relevance_ans



# Returns a dict where to every uid is associated a value of time_relevance calculated based on his topk
def topk_time_relevance(path_data):
    time_relevance_topk = {}

    # Precompute user timestamps weigths
    time_relevance_matrix = path_data.ETR_matrix

    #print(len(path_data.test_labels))
    count = 0
    count_pid = 0
    for uid in path_data.test_labels:
        time_relevance_single_topk = []
        if uid not in time_relevance_matrix or uid not in path_data.uid_topk:
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
            time_relevance = time_relevance_matrix[uid][interaction_timestamp]
            time_relevance_single_topk.append(time_relevance)

        time_relevance_topk[uid] = np.array(time_relevance_single_topk).mean() if len(time_relevance_single_topk) != 0 else 0
    #print(count_pid, count)
    return time_relevance_topk


# Returns an avg value for the time_relevance of a given group
def avg_time_relevance(path_data, attribute_name="Gender"):
    topk_time_relevance_scores = topk_time_relevance(path_data)
    if attribute_name == "Gender":
        user2attribute, attribute2name = get_user2gender(path_data.dataset_name)
    elif attribute_name == "Age":
        user2attribute, attribute2name = get_user2age()
    elif attribute_name == "Occupation":
        user2attribute, attribute2name = get_user2occupation()
    elif attribute_name == None:
        pass
    else:
        print("The attribute selected doesn't exist.")
        return

    avg_groups_time_relevance = {}
    groups_time_relevance_scores = {"Overall": []}

    for _, attribute_label in attribute2name.items():
        groups_time_relevance_scores[attribute_label] = []

    for uid, time_relevance_score in topk_time_relevance_scores.items():
        attr_value = user2attribute[uid]
        attr_name = attribute2name[attr_value]
        groups_time_relevance_scores[attr_name].append(time_relevance_score)
        groups_time_relevance_scores["Overall"].append(time_relevance_score)

    for attribute_label, group_time_relevance_scores in groups_time_relevance_scores.items():
        avg_groups_time_relevance[attribute_label] = np.array(group_time_relevance_scores).mean()

    avg_groups_time_relevance["Overall"] = np.array(groups_time_relevance_scores["Overall"]).mean()

    time_relevance = edict(
        avg_groups_time_relevance=avg_groups_time_relevance,
        groups_time_relevance_scores=groups_time_relevance_scores,
    )

    return time_relevance


def explaination_serentipety_single(path_data, path):
    related_entity_type, related_entity_id = get_related_entity(path)
    explaination_serentipety = path_data.ES_matrix[related_entity_type][related_entity_id]
    return explaination_serentipety


def topks_explaination_serentipety(path_data):
    explaination_serentipety_topk = {}

    # Precompute entity distribution
    exp_serentipety_matrix = path_data.ES_matrix

    #Measure explaination serentipety for topk
    for uid in path_data.test_labels:
        explaination_serentipety_single_topk = []
        if uid not in path_data.uid_topk: continue
        for pid in path_data.uid_topk[uid]:
            if pid not in path_data.uid_pid_explaination[uid]:
                #print("strano 2")
                continue
            path = path_data.uid_pid_explaination[uid][pid]
            related_entity_type, related_entity_id = get_related_entity(path)
            explaination_serentipety = exp_serentipety_matrix[related_entity_type][related_entity_id]
            explaination_serentipety_single_topk.append(explaination_serentipety)
        if len(explaination_serentipety_single_topk) == 0: continue
        explaination_serentipety_topk[uid] = np.array(explaination_serentipety_single_topk).mean()
    return explaination_serentipety_topk


def avg_explaination_serentipety(path_data, attribute_name="Gender"):
    topks_explaination_serentipety_scores = topks_explaination_serentipety(path_data)
    if attribute_name == "Gender":
        user2attribute, attribute2name = get_user2gender(path_data.dataset_name)
    elif attribute_name == "Age":
        user2attribute, attribute2name = get_user2age()
    elif attribute_name == "Occupation":
        user2attribute, attribute2name = get_user2occupation()
    elif attribute_name == None:
        pass
    else:
        print("The attribute selected doesn't exist.")
        return
    avg_groups_explaination_serentipety = {}
    groups_explaination_serentipety_scores = {"Overall": []}

    for _, attribute_label in attribute2name.items():
        groups_explaination_serentipety_scores[attribute_label] = []

    for uid, explaination_serentipety in topks_explaination_serentipety_scores.items():
        attr_value = user2attribute[uid]
        attr_name = attribute2name[attr_value]
        groups_explaination_serentipety_scores[attr_name].append(explaination_serentipety)
        groups_explaination_serentipety_scores["Overall"].append(explaination_serentipety)

    for attribute_label, group_explaination_serentipety_scores in groups_explaination_serentipety_scores.items():
        avg_groups_explaination_serentipety[attribute_label] = np.array(
            group_explaination_serentipety_scores).mean()

    avg_groups_explaination_serentipety["Overall"] = np.array(groups_explaination_serentipety_scores["Overall"]).mean()
    serentipety_results = edict(
        avg_groups_explaination_serentipety=avg_groups_explaination_serentipety,
        groups_explaination_serentipety_scores=groups_explaination_serentipety_scores,
    )
    return serentipety_results

def print_expquality_metrics(avg_groups_time_relevance, avg_groups_explaination_serentipety, avg_groups_explaination_diversity):
    print("\nExplaination Quality:")
    print("Average time relevance after Male: {:.3f} | Female: {:.3f} | Overall: {:.3f}".format(
        avg_groups_time_relevance["Male"],
        avg_groups_time_relevance["Female"],
        avg_groups_time_relevance["Overall"],
    ))
    print("Average explaination serentipety after Male: {:.3f} | Female: {:.3f} | Overall: {:.3f}".format(
        avg_groups_explaination_serentipety["Male"],
        avg_groups_explaination_serentipety["Female"],
        avg_groups_explaination_serentipety["Overall"],
    ))
    print("Average diversity after Male: {:.3f} | Female: {:.3f} | Overall: {:.3f}".format(
        avg_groups_explaination_diversity["Male"],
        avg_groups_explaination_diversity["Female"],
        avg_groups_explaination_diversity["Overall"],
    ))
    print("\n")