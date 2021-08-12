import os

import pandas as pd

from optimizations import *
import sys
from os.path import dirname
from scipy.stats import beta

class PathDataLoader(object):
    def __init__(self, args):
        self.model_dir = "/PGPR"
        self.dataset_name = args.dataset
        self.agent_topk = args.agent_topk
        if args.eval_baseline or args.opt in ["softED", "softES", "softETR"]:
            self.load_uid_topk()
            self.load_uid_pid_path()
        else:
            self.uid_topk = {}
            self.uid_pid_explanation = {}
        self.load_pred_paths()
        self.uid_pid_timestamp, self.uid_timestamp = get_interaction2timestamp(self.dataset_name)

        #Dependent by the model
        self.test_labels = load_labels(self.dataset_name, 'test')
        self.generate_ES_matrix()
        self.generate_ETR_matrix()

    # Returns a dict that map the uid to the topk obtained by the models
    def load_uid_topk(self):
        self.uid_topk = {}
        topk_labels_file = open("paths/" + self.dataset_name + "/agent_topk=" + self.agent_topk + "/uid_topk.csv", 'r')
        reader = csv.reader(topk_labels_file, delimiter=",")
        next(reader, None)  # skip the headers
        for row in reader:
            uid = int(row[0])
            topk = row[1].split(" ")
            topk = [int(x) for x in topk if x != '']
            self.uid_topk[uid] = topk


    # Returns a dict of dict where every uid pid represent the list of paths starting from a user ending in a given product
    def load_uid_pid_path(self):
        self.uid_pid_explanation = {}
        uid_pid_path_topk_file = open("paths/" + self.dataset_name + "/agent_topk=" + self.agent_topk + "/uid_pid_explanation.csv")
        reader = csv.reader(uid_pid_path_topk_file, delimiter=",")
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            normalized_path = normalize_path(row[2])

            if uid not in self.uid_pid_explanation:
                self.uid_pid_explanation[uid] = {}
            if pid not in self.uid_pid_explanation[uid]:
                self.uid_pid_explanation[uid][pid] = []

            self.uid_pid_explanation[uid][pid] = normalized_path
        uid_pid_path_topk_file.close()
        return self.uid_pid_explanation

    # Returns a dict where every
    def load_pred_paths(self):
        self.pred_paths = {}
        uid_pid_path_topk_file = open("paths/" + self.dataset_name + "/agent_topk=" + self.agent_topk + "/pred_paths.csv")
        reader = csv.reader(uid_pid_path_topk_file, delimiter=",")
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            path_score = float(row[2])
            path_prob = float(row[3])
            normalized_path = normalize_path(row[4])

            if uid not in self.pred_paths:
                self.pred_paths[uid] = {}
            if pid not in self.pred_paths[uid]:
                self.pred_paths[uid][pid] = []
            self.pred_paths[uid][pid].append([path_score, path_prob, normalized_path])

        uid_pid_path_topk_file.close()
        return self.pred_paths

    def load_best_pred_paths(self):
        self.best_pred_paths = {}
        uid_pid_path_topk_file = open("paths/" + self.dataset_name + "/agent_topk=" + self.agent_topk + "/best_pred_paths.csv")
        reader = csv.reader(uid_pid_path_topk_file, delimiter=",")
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            path_score = float(row[2])
            path_prob = float(row[3])
            normalized_path = normalize_path(row[4])

            if uid not in self.best_pred_paths:
                self.best_pred_paths[uid] = {}
            if pid not in self.best_pred_paths[uid]:
                self.best_pred_paths[uid][pid] = []
            self.best_pred_paths[uid][pid].append([path_score, path_prob, normalized_path])

        uid_pid_path_topk_file.close()
        return self.best_pred_paths

    def generate_ETR_matrix(self):
        self.ETR_matrix = {}
        time_relevance_matrix = {}
        for uid in self.test_labels:
            timestamp_tr_value = {}
            if uid not in self.uid_timestamp: continue #77 invalid users for lastfm
            self.uid_timestamp[uid].sort()
            def normalized_ema(values):
                values = np.array([i for i in range(len(values))])
                values = pd.Series(values)
                ema_vals = values.ewm(span=10).mean().tolist() #CHECKKKKKKKKKKKKK
                min_res = min(ema_vals)
                max_res = max(ema_vals)
                return [(x - min_res) / (max_res - min_res) for x in ema_vals]

            ema_timestamps = normalized_ema(self.uid_timestamp[uid]) if len(self.uid_timestamp[uid]) > 1 else [0.5]
            for idx, timestamp in enumerate(self.uid_timestamp[uid]):
                timestamp_tr_value[timestamp] = ema_timestamps[idx]
            time_relevance_matrix[uid] = timestamp_tr_value
        self.ETR_matrix = time_relevance_matrix

    def generate_ES_matrix(self):
        # Precompute entity distribution
        exp_serentipety_matrix = {}
        degrees = load_kg(self.dataset_name).degrees
        for type, eid_indegree in degrees.items():
            pid_indegree_list = []
            for pid, indegree in eid_indegree.items():
                pid_indegree_list.append(indegree)  # idx = pid

            #Normalize indegree between 0 and 1
            normalized_indegree_list = [
                (indegree - min(pid_indegree_list)) / (max(pid_indegree_list) - min(pid_indegree_list)) for indegree in
                pid_indegree_list]

            #Generate function tha maps indegree value between 0-1 to ES value
            def get_weigths(normalized_indegree_list):
                a, b = 5, 2
                mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
                x = np.linspace(beta.ppf(0, a, b),
                                beta.ppf(1, a, b), len(set(normalized_indegree_list)))
                y = beta.pdf(x, a, b) / max(beta.pdf(x, a, b))

                return y, x

            x, y = get_weigths(normalized_indegree_list)
            pid_weigth = {}
            for pid, indegree in enumerate(normalized_indegree_list):
                index = np.argmin(np.abs(x - indegree))
                pid_weigth[pid] = y[index]
            exp_serentipety_matrix[type] = pid_weigth
        self.ES_matrix = exp_serentipety_matrix


