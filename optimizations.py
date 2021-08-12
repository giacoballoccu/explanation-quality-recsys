from myutils import get_rec_pid, get_path_type, PATH_TYPES
from metrics import *

def sort_by_ETR(path_full):
    return explanation_time_relevance_single(path_full[2])

def sort_by_ES(path_full):
    return explanation_serendipity_single(path_full[2])

# Soft optimization ETR, for every user topk predicted by the baseline,
# get the predicted paths for every item and change the explanation according to ETR motivations
def soft_optimization_ETR(path_data):
    pred_paths = path_data.pred_paths
    for uid, topk in path_data.uid_topk.items():

        #Retrive topk explainations without changin the selected pids
        for pid in topk:
            pred_paths[uid][pid].sort(key=lambda x: explanation_time_relevance_single(path_data, x[-1]), reverse=True)
            path_data.uid_pid_explaination[uid][pid] = pred_paths[uid][pid][0][-1]

# Soft optimization ETR, for every user topk predicted by the baseline,
# get the predicted paths for every item and change the explanation according to ES motivations
def soft_optimization_ES(path_data):
    pred_path = path_data.pred_paths

    for uid, topk in path_data.uid_topk.items():

        #Retrive topk explainations without changin the selected pids
        for pid in topk:
            pred_path[uid][pid].sort(key=lambda x: explanation_serendipity_single(path_data, x[-1]), reverse=True)
            path_data.uid_pid_explaination[uid][pid] = pred_path[uid][pid][0][-1]



def soft_optimization_ED(path_data):
    pred_path = path_data.pred_paths

    for uid, topk in path_data.uid_topk.items():
        path_data.uid_pid_explaination[uid] = {}
        path_took = set()
        path_type_freq = [[path_type, 0] for path_type in PATH_TYPES[path_data.dataset_name]]
        for pid in topk:
            for path in pred_path[uid][pid]:
                path_type = get_path_type(path[-1])
                for t in path_type_freq:
                    if t[0] == path_type:
                        t[1] += 1
            path_type_freq.sort(key=lambda x: x[1])
            #Try to pick the most number of different paths
            path_took_len = len(path_took)
            for t in path_type_freq:
                type = t[0]
                count = t[1]
                if type in path_took: continue
                #Search for the first path of the new type
                for path in pred_path[uid][pid]:
                    if get_path_type(path[-1]) == type:
                        path_data.uid_pid_explaination[uid][pid] = path[-1]
                        path_took.add(type)
                    if pid in path_data.uid_pid_explaination[uid]:
                        break
            # If is true this mean we haven't found a good new path, so we took the most repeted infrequent
            if path_took_len == len(path_took):
                for t in path_type_freq:
                    type = t[0]
                    count = t[1]
                    # Search for the first path of the new type
                    for path in pred_path[uid][pid]:
                        if get_path_type(path[-1]) == type:
                            path_data.uid_pid_explaination[uid][pid] = path[-1]
                        if pid in path_data.uid_pid_explaination[uid]:
                            break

#ETR Alpha optimization
def optimize_ETR(path_data, alpha):
    pred_path = path_data.pred_paths

    #Pred_paths {uid: {pid: [[path_score, path_prob_path], ..., [path_score, path_prob_path]], ...,
    # pidn: [[path_score, path_prob_path], ..., [path_score, path_prob_path]]]}, ..., uidn: {pid: ...}}
    for uid, pid_list in pred_path.items():
        candidates = []
        best_candidates = []
        best_candidates_pids = set()

        #Create candidate list
        for pid, path_list in pid_list.items():
            candidates.extend(path_list)

        candidates.sort(key=lambda candidate: (candidate[0] * (1-alpha)) + (explanation_time_relevance_single(path_data, candidate[-1]) * alpha), reverse=True)

        #Pick the best items
        for candidate in candidates:
            rec_pid = get_rec_pid(candidate)
            if rec_pid in best_candidates_pids: continue
            best_candidates.append(candidate)
            best_candidates_pids.add(rec_pid)
            if len(best_candidates) == 10: break

        #if len(best_candidates) < 10:
        #    print("LESS THAN 10!")
        #Reorder topk by path_score
        best_candidates.sort(key=lambda candidate: candidate[0],
                             reverse=True)
        # Update the topk with the reranked one
        path_data.uid_topk[uid] = [get_rec_pid(candidate) for candidate in best_candidates]
        path_data.uid_pid_explaination[uid] = {get_rec_pid(candidate): candidate[-1] for candidate in best_candidates}

#ES Alpha optimization
def optimize_ES(path_data, alpha):
    pred_paths = path_data.pred_paths

    for uid, pid_list in pred_paths.items():
        candidates = []
        best_candidates = []
        best_candidates_pids = set()

        for pid, path_list in pid_list.items():
            path_list.sort(key=lambda x: x[1], reverse=True)
            candidates.extend(path_list)

        candidates.sort(key=lambda x: (x[0] * (1-alpha)) + (explanation_serendipity_single(path_data, x[-1]) * alpha), reverse=True)

        #Pick the best items
        for candidate in candidates:
            rec_pid = get_rec_pid(candidate)
            if rec_pid in best_candidates_pids: continue
            best_candidates.append(candidate)
            best_candidates_pids.add(rec_pid)
            if len(best_candidates) == 10: break

        #if len(best_candidates) < 10:
        #    print("LESS THAN 10!")

        #Reorder topk by path_score
        best_candidates.sort(key=lambda candidate: candidate[0],
                             reverse=True)
        # Update the topk with the reranked one
        path_data.uid_topk[uid] = [get_rec_pid(candidate) for candidate in best_candidates]
        path_data.uid_pid_explaination[uid] = {get_rec_pid(candidate): candidate[-1] for candidate in best_candidates}

#ED Alpha optimization
def optimize_ED(path_data, alpha):
    pred_path = path_data.pred_paths
    for uid, pid_list in pred_path.items():
        candidates = []
        best_candidates = []
        best_candidates_pids = set()

        for pid, path_list in pid_list.items():
            path_list = pred_path[uid][pid]
            candidates.extend(path_list)

        # Create a bin for every path type and insert in them every path of that type
        bins = {}
        for candidate in candidates:
            candidate_path_type = get_path_type(candidate[-1])
            if get_path_type(candidate[-1]) not in bins:
                bins[candidate_path_type] = []
            bins[candidate_path_type].append(candidate)

        # Sort every path type bin by a mixed score based on explaination time relevance and item relevance
        for bin_type, path_list in bins.items():
            path_list.sort(key=lambda x: x[0],
                           reverse=True)

        #Search the best for path_score among all the top of every bin, if the paths isn't already seen in the best_candidate give him a alpha bonus
        ptype_seen = set()
        while len(best_candidates) < 10:
            best_type = ""
            best_score = -1
            for bin_type, path_list in bins.items():
                if len(path_list) == 0: continue
                candidate = path_list[0]
                rec_pid = get_rec_pid(path_list[0][-1])
                while rec_pid in best_candidates_pids:
                    path_list.pop(0)
                    if len(path_list) == 0: break
                    candidate = path_list[0]
                    rec_pid = get_rec_pid(candidate[-1])
                if len(path_list) == 0: continue
                bonus = alpha if get_path_type(candidate[-1]) not in ptype_seen else 0
                score = candidate[0] + bonus
                if score > best_score:
                    best_score = score
                    best_type = get_path_type(candidate[-1])
            if best_type == "":
            #    print("Less than 10: {}".format(len(best_candidates)))
               break
            best_candidates.append(bins[best_type][0])
            best_candidates_pids.add(get_rec_pid(bins[best_type][0][-1]))
            ptype_seen.add(best_type)
            bins[best_type].pop(0)
            if len(bins[best_type]) == 0:
                bins.pop(best_type)

        # Rearrange the topk based on the metric
        best_candidates.sort(key=lambda x: x[0],
                             reverse=True)

        # Update the topk with the reranked one
        path_data.uid_topk[uid] = [get_rec_pid(candidate) for candidate in best_candidates]
        path_data.uid_pid_explaination[uid] = {get_rec_pid(candidate): candidate[-1] for candidate in best_candidates}

#ETR+ES Optimization
def optimize_ETR_ES(path_data, alpha):
    pred_path = path_data.pred_paths

    for uid, pid_list in pred_path.items():
        candidates = []
        best_candidates = []
        best_candidates_pids = set()

        # Create candidate list
        for pid, path_list in pid_list.items():
            candidates.extend(path_list)

        # Normalize scores between 0 and 1
        scores_list = [candidate[0] for candidate in candidates]
        min_score = min(scores_list)
        max_score = max(scores_list)
        for candidate in candidates:
            candidate[0] = (candidate[0] - min_score) / (max_score - min_score)

        candidates.sort(key=lambda x: (x[0] * (1-alpha)) + ((explanation_time_relevance_single(path_data, x[-1]) + explanation_serendipity_single(path_data, x[-1])) * alpha),
                        reverse=True)

        # Pick the best items
        for candidate in candidates:
            rec_pid = get_rec_pid(candidate)
            if rec_pid in best_candidates_pids: continue
            best_candidates.append(candidate)
            best_candidates_pids.add(rec_pid)
            if len(best_candidates) == 10: break

        #if len(best_candidates) < 10:
        #    print("LESS THAN 10!")

        # Reorder topk by path_score
        best_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        # Update the topk with the reranked one
        path_data.uid_topk[uid] = [get_rec_pid(candidate) for candidate in best_candidates]
        path_data.uid_pid_explaination[uid] = {get_rec_pid(candidate): candidate[-1] for candidate in best_candidates}

def optimize_ED_ETR(path_data, alpha):
    pred_path = path_data.pred_paths
    for uid, pid_list in pred_path.items():
        candidates = []
        best_candidates = []
        best_candidates_pids = set()

        for pid, path_list in pid_list.items():
            path_list = pred_path[uid][pid]
            candidates.extend(path_list)

        # Create a bin for every path type and insert in them every path of that type
        bins = {}
        for candidate in candidates:
            candidate_path_type = get_path_type(candidate[-1])
            if get_path_type(candidate[-1]) not in bins:
                bins[candidate_path_type] = []
            bins[candidate_path_type].append(candidate)

        # Sort every path type bin by a mixed score based on explaination time relevance and item relevance
        for bin_type, path_list in bins.items():
            path_list.sort(key=lambda x: (x[0] * (1-alpha)) + (explanation_time_relevance_single(path_data, x[-1]) * alpha),
                           reverse=True)

        ptype_seen = set()
        while len(best_candidates) < 10:
            best_type = ""
            best_score = -1
            for bin_type, path_list in bins.items():
                if len(path_list) == 0: continue
                candidate = path_list[0]
                rec_pid = get_rec_pid(path_list[0][-1])
                while rec_pid in best_candidates_pids:
                    path_list.pop(0)
                    if len(path_list) == 0: break
                    candidate = path_list[0]
                    rec_pid = get_rec_pid(candidate[-1])
                if len(path_list) == 0: continue
                bonus = alpha if get_path_type(candidate[-1]) not in ptype_seen else 0
                score = candidate[0] + bonus
                if score > best_score:
                    best_score = score
                    best_type = get_path_type(candidate[-1])
            if best_type == "":
            #    print("Less than 10: {}".format(len(best_candidates)))
               break
            best_candidates.append(bins[best_type][0])
            best_candidates_pids.add(get_rec_pid(bins[best_type][0][-1]))
            ptype_seen.add(best_type)
            bins[best_type].pop(0)
            if len(bins[best_type]) == 0:
                bins.pop(best_type)

        # Rearrange the topk based on path_score
        best_candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        # Update the topk with the reranked one
        path_data.uid_topk[uid] = [get_rec_pid(candidate) for candidate in best_candidates]
        path_data.uid_pid_explaination[uid] = {get_rec_pid(candidate): candidate[-1] for candidate in best_candidates}

#ED+ES Alpha optimization
def optimize_ED_ES(path_data, alpha):
    pred_path = path_data.pred_paths
    for uid, pid_list in pred_path.items():
        candidates = []
        best_candidates = []
        best_candidates_pids = set()

        for pid, path_list in pid_list.items():
            path_list = pred_path[uid][pid]
            candidates.extend(path_list)

        # Create a bin for every path type and insert in them every path of that type
        bins = {}
        for candidate in candidates:
            candidate_path_type = get_path_type(candidate[-1])
            if get_path_type(candidate[-1]) not in bins:
                bins[candidate_path_type] = []
            bins[candidate_path_type].append(candidate)

        # Sort every path type bin by a mixed score based on explaination time relevance and item relevance
        for bin_type, path_list in bins.items():
            path_list.sort(key=lambda x: (x[0] * (1-alpha)) + (explanation_serendipity_single(path_data, x[-1]) * alpha),
                           reverse=True)

        ptype_seen = set()
        while len(best_candidates) < 10:
            best_type = ""
            best_score = -1
            for bin_type, path_list in bins.items():
                if len(path_list) == 0: continue
                candidate = path_list[0]
                rec_pid = get_rec_pid(path_list[0][-1])
                while rec_pid in best_candidates_pids:
                    path_list.pop(0)
                    if len(path_list) == 0: break
                    candidate = path_list[0]
                    rec_pid = get_rec_pid(candidate[-1])
                if len(path_list) == 0: continue
                bonus = alpha if get_path_type(candidate[-1]) not in ptype_seen else 0
                score = candidate[0] + bonus
                if score > best_score:
                    best_score = score
                    best_type = get_path_type(candidate[-1])
            if best_type == "":
            #    print("Less than 10: {}".format(len(best_candidates)))
               break
            best_candidates.append(bins[best_type][0])
            best_candidates_pids.add(get_rec_pid(bins[best_type][0][-1]))
            ptype_seen.add(best_type)
            bins[best_type].pop(0)
            if len(bins[best_type]) == 0:
                bins.pop(best_type)

        # Rearrange the topk based on the metric
        best_candidates.sort(key=lambda x: x[0],
                             reverse=True)

        # Update the topk with the reranked one
        path_data.uid_topk[uid] = [get_rec_pid(candidate) for candidate in best_candidates]
        path_data.uid_pid_explaination[uid] = {get_rec_pid(candidate): candidate[-1] for candidate in best_candidates}

#ED+ES+ETR Alpha optimization
def optimize_ED_ES_ETR(path_data, alpha):
    pred_path = path_data.pred_paths
    for uid, pid_list in pred_path.items():
        candidates = []
        best_candidates = []
        best_candidates_pids = set()

        for pid, path_list in pid_list.items():
            path_list = pred_path[uid][pid]
            candidates.extend(path_list)

        # Create a bin for every path type and insert in them every path of that type
        bins = {}
        for candidate in candidates:
            candidate_path_type = get_path_type(candidate[-1])
            if get_path_type(candidate[-1]) not in bins:
                bins[candidate_path_type] = []
            bins[candidate_path_type].append(candidate)

        # Sort every path type bin by a mixed score based on explaination time relevance and item relevance
        for bin_type, path_list in bins.items():
            path_list.sort(
                key=lambda x: (x[0] * (1 - alpha)) + ((explanation_time_relevance_single(path_data, x[-1]) + explanation_serendipity_single(path_data, x[-1])) * alpha),
                reverse=True)

        ptype_seen = set()
        while len(best_candidates) < 10:
            best_type = ""
            best_score = -1
            for bin_type, path_list in bins.items():
                if len(path_list) == 0: continue
                candidate = path_list[0]
                rec_pid = get_rec_pid(path_list[0][-1])
                while rec_pid in best_candidates_pids:
                    path_list.pop(0)
                    if len(path_list) == 0: break
                    candidate = path_list[0]
                    rec_pid = get_rec_pid(candidate[-1])
                if len(path_list) == 0: continue
                bonus = alpha if get_path_type(candidate[-1]) not in ptype_seen else 0
                score = candidate[0] + bonus
                if score > best_score:
                    best_score = score
                    best_type = get_path_type(candidate[-1])
            if best_type == "":
                #    print("Less than 10: {}".format(len(best_candidates)))
                break
            best_candidates.append(bins[best_type][0])
            best_candidates_pids.add(get_rec_pid(bins[best_type][0][-1]))
            ptype_seen.add(best_type)
            bins[best_type].pop(0)
            if len(bins[best_type]) == 0:
                bins.pop(best_type)

        # Rearrange the topk based on the metric
        best_candidates.sort(key=lambda x: x[0],
                             reverse=True)

        # Update the topk with the reranked one
        path_data.uid_topk[uid] = [get_rec_pid(candidate) for candidate in best_candidates]
        path_data.uid_pid_explaination[uid] = {get_rec_pid(candidate): candidate[-1] for candidate in best_candidates}

