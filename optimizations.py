from myutils import get_rec_pid, get_path_type
from metrics import *

def sort_by_LIR(path_full):
    return LIR_single(path_full[2])

def sort_by_SEP(path_full):
    return SEP_single(path_full[2])

# Soft optimization LIR, for every user topk predicted by the baseline,
# get the predicted paths for every item and change the explanation according to LIR motivations
def soft_optimization_LIR(path_data):
    pred_paths = path_data.pred_paths
    for uid, topk in path_data.uid_topk.items():

        #Retrive topk explainations without changin the selected pids
        for pid in topk:
            pred_paths[uid][pid].sort(key=lambda x: LIR_single(path_data, x[-1]), reverse=True)
            path_data.uid_pid_explaination[uid][pid] = pred_paths[uid][pid][0][-1]

# Soft optimization LIR, for every user topk predicted by the baseline,
# get the predicted paths for every item and change the explanation according to SEP motivations
def soft_optimization_SEP(path_data):
    pred_path = path_data.pred_paths

    for uid, topk in path_data.uid_topk.items():

        #Retrive topk explainations without changin the selected pids
        for pid in topk:
            pred_path[uid][pid].sort(key=lambda x: SEP_single(path_data, x[-1]), reverse=True)
            path_data.uid_pid_explaination[uid][pid] = pred_path[uid][pid][0][-1]



def soft_optimization_ETD(path_data):
    pred_path = path_data.pred_paths
    for uid, topk in path_data.uid_topk.items():
        path_data.uid_pid_explaination[uid] = {}
        ptype_seen = set()
        for pid in topk:
            curr_size = len(path_data.uid_pid_explaination[uid])
            current_item_pred_paths = pred_path[uid][pid]
            current_item_pred_paths.sort(key=lambda x: x[1]) #Sort for probability
            for path in current_item_pred_paths:
                ptype = get_path_type(path[-1])
                if ptype not in ptype_seen:
                    path_data.uid_pid_explaination[uid][pid] = path[-1]
                    ptype_seen.add(ptype)
                    break
                    # No different path have been found
            if curr_size == len(path_data.uid_pid_explaination[uid]):
                path_data.uid_pid_explaination[uid][pid] = current_item_pred_paths[0][-1]

#LIR Alpha optimization
def optimize_LIR(path_data, alpha):
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

        candidates.sort(key=lambda candidate: (candidate[0] * (1-alpha)) + (LIR_single(path_data, candidate[-1]) * alpha), reverse=True)

        #Pick the best items
        for candidate in candidates:
            rec_pid = get_rec_pid(candidate)
            if rec_pid in best_candidates_pids: continue
            best_candidates.append(candidate)
            best_candidates_pids.add(rec_pid)
            if len(best_candidates) == 10: break

        #if len(best_candidates) < 10:
        #    print("LSEPS THAN 10!")
        #Reorder topk by path_score
        best_candidates.sort(key=lambda candidate: candidate[0],
                             reverse=True)
        # Update the topk with the reranked one
        path_data.uid_topk[uid] = [get_rec_pid(candidate) for candidate in best_candidates]
        path_data.uid_pid_explaination[uid] = {get_rec_pid(candidate): candidate[-1] for candidate in best_candidates}

#SEP Alpha optimization
def optimize_SEP(path_data, alpha):
    pred_paths = path_data.pred_paths

    for uid, pid_list in pred_paths.items():
        candidates = []
        best_candidates = []
        best_candidates_pids = set()

        for pid, path_list in pid_list.items():
            path_list.sort(key=lambda x: x[1], reverse=True)
            candidates.extend(path_list)

        candidates.sort(key=lambda x: (x[0] * (1-alpha)) + (SEP_single(path_data, x[-1]) * alpha), reverse=True)

        #Pick the best items
        for candidate in candidates:
            rec_pid = get_rec_pid(candidate)
            if rec_pid in best_candidates_pids: continue
            best_candidates.append(candidate)
            best_candidates_pids.add(rec_pid)
            if len(best_candidates) == 10: break

        #if len(best_candidates) < 10:
        #    print("LSEPS THAN 10!")

        #Reorder topk by path_score
        best_candidates.sort(key=lambda candidate: candidate[0],
                             reverse=True)
        # Update the topk with the reranked one
        path_data.uid_topk[uid] = [get_rec_pid(candidate) for candidate in best_candidates]
        path_data.uid_pid_explaination[uid] = {get_rec_pid(candidate): candidate[-1] for candidate in best_candidates}

#ETD Alpha optimization
def optimize_ETD(path_data, alpha):
    pred_path = path_data.pred_paths
    for uid, pid_list in pred_path.items():
        candidates = []
        best_candidates = []
        best_candidates_pids = set()

        #Populate candidate list
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

#LIR+SEP Optimization
def optimize_LIR_SEP(path_data, alpha):
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

        candidates.sort(key=lambda x: (x[0] * (1-alpha)) + ((LIR_single(path_data, x[-1]) + SEP_single(path_data, x[-1])) * alpha),
                        reverse=True)

        # Pick the best items
        for candidate in candidates:
            rec_pid = get_rec_pid(candidate)
            if rec_pid in best_candidates_pids: continue
            best_candidates.append(candidate)
            best_candidates_pids.add(rec_pid)
            if len(best_candidates) == 10: break

        #if len(best_candidates) < 10:
        #    print("LSEPS THAN 10!")

        # Reorder topk by path_score
        best_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        # Update the topk with the reranked one
        path_data.uid_topk[uid] = [get_rec_pid(candidate) for candidate in best_candidates]
        path_data.uid_pid_explaination[uid] = {get_rec_pid(candidate): candidate[-1] for candidate in best_candidates}

def optimize_ETD_LIR(path_data, alpha):
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
            path_list.sort(key=lambda x: (x[0] * (1-alpha)) + (LIR_single(path_data, x[-1]) * alpha),
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

#ETD+SEP Alpha optimization
def optimize_ETD_SEP(path_data, alpha):
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
            path_list.sort(key=lambda x: (x[0] * (1-alpha)) + (SEP_single(path_data, x[-1]) * alpha),
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

#ETD+SEP+LIR Alpha optimization
def optimize_ETD_SEP_LIR(path_data, alpha):
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
                key=lambda x: (x[0] * (1 - alpha)) + ((LIR_single(path_data, x[-1]) + SEP_single(path_data, x[-1])) * alpha),
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

