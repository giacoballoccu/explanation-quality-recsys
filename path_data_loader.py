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


'''
if __name__ == '__main__':
    sys.path.append(dirname("PGPR"))
    pred_path = path_data.pred_paths
    #with open('log/ES_mitigation_alphas.txt', 'w') as f:
        #sys.stdout = f

    uid_gender, gender2name = get_user2gender()
    n_male = []
    n_female = []
    n_labels_uid = {}
    for uid, labels in path_data.test_labels.items():
        gender_value = uid_gender[uid]
        n_labels = len(labels)
        if gender_value == 0:
            n_male.append(n_labels)
        else:
            n_female.append(n_labels)
        if n_labels not in n_labels_uid:
            n_labels_uid[n_labels] = []
        n_labels_uid[n_labels].append(uid)
    train_size_male = [(n*100)/20 for n in n_male]
    train_size_female = [(n*100)/20 for n in n_female]
    print("Average train size for {:.3f}, average train size for female {:.3f}".format(np.mean(train_size_male), np.mean(train_size_female)))
    print("Average test size for male {:.3f}, average test size for female {:.3f}".format(np.mean(n_male), np.mean(n_female)))

    n_labels_keys = [x for x in n_labels_uid.keys()]
    n_labels_keys.sort()
    differences = []
    for size_test in n_labels_keys:
        uids = n_labels_uid[size_test]
        metrics = measure_rec_quality_group(path_data, uids)
        print("User with train_size={}, test_size={}, Male: {}, Female: {}, Total: {}".format((100*size_test)/20, size_test, metrics.n_male, metrics.n_female, metrics.n_male+metrics.n_female))
        print("NDCG Male: {:.3f}, NDCG Female: {:.3f}, NDCG Total: {:.3f} DIFF(Male-Female): {:.3f}".format(
            np.mean(metrics.ndcg["Male"]), np.mean(metrics.ndcg["Female"]), np.mean(metrics.ndcg["All"]), (np.mean(metrics.ndcg["Male"]) - np.mean(metrics.ndcg["Female"]))
        ))
        print()
        diff = (np.mean(metrics.ndcg["Male"]) - np.mean(metrics.ndcg["Female"]))
        if np.math.isnan(diff): continue
        differences.append(diff)
    print("Mean: {:.7f}".format(np.mean(differences)))


    avg_scores = {}
    for uid, pred_paths in path_data.best_pred_paths.items():
        uid_scores = []
        gender_value = uid_gender[uid]
        gender_name = gender2name[gender_value]
        for pred_pid, pred_path in pred_paths.items():
            uid_scores.append(pred_path[0][0])
        if gender_name not in avg_scores:
            avg_scores[gender_name] = []
        avg_scores[gender_name].append(np.mean(uid_scores))
    print("Average scores for best paths Male: {:.3f}, Female: {:.3f}".format(np.mean(avg_scores["Male"]), np.mean(avg_scores["Female"])))

    scores_male = []
    scores_female = []
    for uid, topk in path_data.uid_topk.items():
        gender = uid_gender[uid]
        topk_scores = []
        for pid in topk:
            topk_scores.append(path_data.best_pred_paths[uid][pid][0][0])
        if gender == 0:
            scores_male.append(np.mean(topk_scores))
        else:
            scores_female.append(np.mean(topk_scores))
    print("Average scores in topk Male: {:.3f}, Female: {:.3f}".format(np.mean(scores_male), np.mean(scores_female)))


    # with open("models/PGPR/results/before_mitigation_group_results.csv", "w+") as file:
    #     writer = csv.writer(file)
    #     header = ["metric", "group", "data"]
    #     writer.writerow(header)
    #     for group_name, values in tr_before_mitigation.groups_time_relevance_scores.items():
    #         if group_name == "General": continue
    #         for value in values:
    #             writer.writerow(["time_relevance", group_name, value])
    #     for group_name, values in es_before_mitigation.groups_explaination_serentipety_scores.items():
    #         if group_name == "General": continue
    #         for value in values:
    #             writer.writerow(["serentipety", group_name, value])
    #     for group_name, values in ed_before_mitigation.groups_explaination_diversity_scores.items():
    #         if group_name == "General": continue
    #         for value in values:
    #             writer.writerow(["diversity", group_name, value])
    # exit(0)

    # with open("models/PGPR/results/baseline_moving_alpha.csv", "w+") as file:
    #     writer = csv.writer(file)
    #     header = ["alpha", "metric", "group", "data"]
    #     writer.writerow(header)
    #     for alpha in [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    #                   0.85, 0.90, 0.95, 1.]:
    #         for metric_name, group_values in exp_metrics_before.items():
    #             for group_name, value in group_values.items():
    #                 writer.writerow([alpha, metric_name, group_name, value])
    #         for metric_name, group_values in rec_metrics.items():
    #             for group_name, value in group_values.items():
    #                 writer.writerow([alpha, metric_name, group_name, np.mean(value)])
    # file.close()


    # uid_gender, gender2name = get_user2gender()
    # scores_male = []
    # scores_female = []
    # for uid, topk in path_data.uid_topk.items():
    #     gender = uid_gender[uid]
    #     topk_scores = []
    #     for pid in topk:
    #         topk_scores.append(path_data.best_pred_paths[uid][pid][0][0])
    #     if gender == 0:
    #         scores_male.append(np.mean(topk_scores))
    #     else:
    #         scores_female.append(np.mean(topk_scores))
    #print(
    #    "Average scores in topk Male: {:.3f}, Female: {:.3f}".format(np.mean(scores_male), np.mean(scores_female)))

    #alpha = 0.2
    #mitigation = "ED_ES_ETR_opt_alpha={:.2f}".format(alpha)
    mitigation = "ndcg"
    log_path = "./log/agent_topk=" + path_data.agent_topk + "/" + mitigation + ".txt"
    result_path = "./results/agent_topk=" + path_data.agent_topk + "/" + mitigation + "_avg.csv"
    distribution_path = "./results/agent_topk=" + path_data.agent_topk + "/" + mitigation + "_distribution.csv"
    if not os.path.exists("./log/agent_topk=" + path_data.agent_topk):
        os.makedirs("./log/agent_topk=" + path_data.agent_topk)
    if not os.path.exists("./results/agent_topk=" + path_data.agent_topk):
            os.makedirs("./results/agent_topk=" + path_data.agent_topk)

    # ALpha test
     #result_file = open(result_path, "w+")
    #distribution_file = open(distribution_path, "w+")
    #writer = csv.writer(result_file)
    with open(log_path, "w+") as log_file:
        sys.stdout = log_file
        print("--- BEFORE MITIGATION ---")
        rec_metrics = measure_rec_quality(path_data)
        print_rec_metrics(rec_metrics)
        tr_before_mitigation = avg_time_relevance(path_data)
        es_before_mitigation = avg_explaination_serentipety(path_data)
        ed_before_mitigation = avg_diversity_score(path_data)
        print_expquality_metrics(tr_before_mitigation.avg_groups_time_relevance,
                                 es_before_mitigation.avg_groups_explaination_serentipety,
                                 ed_before_mitigation.avg_groups_explaination_diversity)
        exp_metrics_before = {}
        exp_metrics_before["ETR"] = tr_before_mitigation.avg_groups_time_relevance
        exp_metrics_before["ES"] = es_before_mitigation.avg_groups_explaination_serentipety
        exp_metrics_before["ED"] = ed_before_mitigation.avg_groups_explaination_diversity
        with open(result_path, "w+") as file:
            writer = csv.writer(file)
            header = ["alpha", "metric", "group", "data", "opt"]
            writer.writerow(header)
            for alpha in [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,0.80, 0.85, 0.90, 0.95, 1.]:

                print("--- AFTER {} optimization with alpha={}---".format(mitigation, alpha))
                optimize_ED(path_data, alpha)
                rec_metrics_after = measure_rec_quality(path_data)
                print_rec_metrics(rec_metrics_after)
                exp_metrics_after = {}
                exp_metrics_after["ETR"] = avg_time_relevance(path_data).avg_groups_time_relevance
                exp_metrics_after["ES"] = avg_explaination_serentipety(path_data).avg_groups_explaination_serentipety
                exp_metrics_after["ED"] = avg_diversity_score(path_data).avg_groups_explaination_diversity
                print_expquality_metrics(exp_metrics_after["ETR"],
                                         exp_metrics_after["ES"],
                                         exp_metrics_after["ED"])
                print("\n")
                #for metric_name, group_values in exp_metrics_after.items():
                #    for group_name, value in group_values.items():
                #       writer.writerow([alpha, metric_name, group_name, value, mitigation])
                #for metric_name, group_values in rec_metrics_after.items():
                #   for group_name, value in group_values.items():
                #       writer.writerow([alpha, metric_name, group_name, np.mean(value), mitigation])
        file.close()
    log_file.close()
    exit()

    # Distribution Saving
    distribution_file = open(distribution_path, "w+")
    writer_distribution = csv.writer(distribution_file)
    header = ["metric", "group", "data", "opt"]
    writer_distribution.writerow(header)
    rec_metrics = measure_rec_quality(path_data)
    tr_before_mitigation = avg_time_relevance(path_data)
    es_before_mitigation = avg_explaination_serentipety(path_data)
    ed_before_mitigation = avg_diversity_score(path_data)

    exp_metrics_before = {}
    distributions_exp_metrics_before = {}

    exp_metrics_before["ETR"] = tr_before_mitigation.avg_groups_time_relevance
    exp_metrics_before["ES"] = es_before_mitigation.avg_groups_explaination_serentipety
    exp_metrics_before["ED"] = ed_before_mitigation.avg_groups_explaination_diversity

    distributions_exp_metrics_before["ETR"] = tr_before_mitigation.groups_time_relevance_scores
    distributions_exp_metrics_before["ES"] = es_before_mitigation.groups_explaination_serentipety_scores
    distributions_exp_metrics_before["ED"] = ed_before_mitigation.groups_explaination_diversity_scores

    optimize_ED_ES_ETR(path_data, alpha)

    exp_metrics_after = {}
    distributions_exp_metrics_after = {}
    tr_after_mitigation = avg_time_relevance(path_data)
    es_after_mitigation = avg_explaination_serentipety(path_data)
    ed_after_mitigation = avg_diversity_score(path_data)

    distributions_exp_metrics_after["ETR"] = tr_after_mitigation.groups_time_relevance_scores
    distributions_exp_metrics_after["ES"] = es_after_mitigation.groups_explaination_serentipety_scores
    distributions_exp_metrics_after["ED"] = ed_after_mitigation.groups_explaination_diversity_scores

    for metric_name, group_values in distributions_exp_metrics_after.items():
        for group_name, values in group_values.items():
            if group_name == "Overall": continue
            for value in values:
                writer_distribution.writerow([metric_name, group_name, value, mitigation])
    distribution_file.close()
    exit()








    #Baseline Saving
    result_file = open(result_path, "w+")
    distribution_file = open(distribution_path, "w+")
    writer = csv.writer(result_file)
    writer_distribution = csv.writer(distribution_file)
    header = ["metric","group","data","opt"]
    writer_distribution.writerow(header)
    header = ["alpha","metric","group","data","opt"]
    writer.writerow(header)
    rec_metrics = measure_rec_quality(path_data)
    tr_before_mitigation = avg_time_relevance(path_data)
    es_before_mitigation = avg_explaination_serentipety(path_data)
    ed_before_mitigation = avg_diversity_score(path_data)

    exp_metrics_before = {}
    distributions_exp_metrics_before = {}

    exp_metrics_before["ETR"] = tr_before_mitigation.avg_groups_time_relevance
    exp_metrics_before["ES"] = es_before_mitigation.avg_groups_explaination_serentipety
    exp_metrics_before["ED"] = ed_before_mitigation.avg_groups_explaination_diversity

    distributions_exp_metrics_before["ETR"] = tr_before_mitigation.groups_time_relevance_scores
    distributions_exp_metrics_before["ES"] = es_before_mitigation.groups_explaination_serentipety_scores
    distributions_exp_metrics_before["ED"] = ed_before_mitigation.groups_explaination_diversity_scores
    for metric_name, group_values in distributions_exp_metrics_before.items():
       for group_name, values in group_values.items():
           if group_name == "Overall": continue
           for value in values:
               writer_distribution.writerow([metric_name, group_name, value, mitigation])
    for alpha in [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                  0.75,0.80, 0.85, 0.90, 0.95, 1.]:
        for metric_name, group_values in exp_metrics_before.items():
            for group_name, value in group_values.items():
                writer.writerow([alpha, metric_name, group_name, value, mitigation])
        for metric_name, group_values in rec_metrics.items():
            for group_name, value in group_values.items():
                writer.writerow([alpha, metric_name, group_name, np.mean(value), mitigation])
    result_file.close()
    distribution_file.close()
    exit()



    # SOFT TEST
    with open(log_path, "w+") as log_file:
        sys.stdout = log_file
        print("--- BEFORE MITIGATION ---")
        rec_metrics = measure_rec_quality(path_data)
        print_rec_metrics(rec_metrics)
        tr_before_mitigation = avg_time_relevance(path_data)
        es_before_mitigation = avg_explaination_serentipety(path_data)
        ed_before_mitigation = avg_diversity_score(path_data)
        print_expquality_metrics(tr_before_mitigation.avg_groups_time_relevance,
                                 es_before_mitigation.avg_groups_explaination_serentipety,
                                 ed_before_mitigation.avg_groups_explaination_diversity)

        exp_metrics_before = {}
        distributions_exp_metrics_before = {}

        exp_metrics_before["ETR"] = tr_before_mitigation.avg_groups_time_relevance
        exp_metrics_before["ES"] = es_before_mitigation.avg_groups_explaination_serentipety
        exp_metrics_before["ED"] = ed_before_mitigation.avg_groups_explaination_diversity

        # distributions_exp_metrics_before["ETR"] = tr_before_mitigation.groups_time_relevance_scores
        # distributions_exp_metrics_before["ES"] = es_before_mitigation.groups_explaination_serentipety_scores
        # distributions_exp_metrics_before["ED"] = ed_before_mitigation.groups_explaination_diversity_scores
        # distribution_file = open(distribution_path, "w+")
        # writer_distribution = csv.writer(distribution_file)
        #
        # for metric_name, group_values in distributions_exp_metrics_before.items():
        #     for group_name, values in group_values.items():
        #         if group_name == "Overall": continue
        #         for value in values:
        #             writer_distribution.writerow([metric_name, group_name, value, mitigation])

        results_file = open(result_path, "w+")
        distribution_file = open(distribution_path, "w+")
        writer_results = csv.writer(results_file)
        writer_distribution = csv.writer(distribution_file)
        header = ["metric", "group", "data", "opt"]
        writer_results.writerow(header)
        writer_distribution.writerow(header)

        print("--- AFTER {} optimization---".format(mitigation))
        soft_optimization_ED(path_data)
        time_relevance_after = avg_time_relevance(path_data)
        explanation_serendipity_after = avg_explaination_serentipety(path_data)
        explanation_diversity_after = avg_diversity_score(path_data)
        rec_metrics_after = measure_rec_quality(path_data)
        print_rec_metrics(rec_metrics_after)

        avg_exp_metrics_after = {}
        distributions_exp_metrics_after = {}

        avg_exp_metrics_after["ETR"] = time_relevance_after.avg_groups_time_relevance
        avg_exp_metrics_after["ES"] = explanation_serendipity_after.avg_groups_explaination_serentipety
        avg_exp_metrics_after["ED"] = explanation_diversity_after.avg_groups_explaination_diversity

        distributions_exp_metrics_after["ETR"] = time_relevance_after.groups_time_relevance_scores
        distributions_exp_metrics_after["ES"] = explanation_serendipity_after.groups_explaination_serentipety_scores
        distributions_exp_metrics_after["ED"] = explanation_diversity_after.groups_explaination_diversity_scores

        print_expquality_metrics(avg_exp_metrics_after["ETR"],
                                 avg_exp_metrics_after["ES"],
                                 avg_exp_metrics_after["ED"])
        print("\n")
        #for metric_name, group_values in distributions_exp_metrics_after.items():
        #    for group_name, values in group_values.items():
        #        if group_name == "Overall": continue
        #        for value in values:
        #            writer_distribution.writerow([metric_name, group_name, value, mitigation])
        for metric_name, group_values in avg_exp_metrics_after.items():
            for group_name, value in group_values.items():
                writer_results.writerow([metric_name, group_name, value, mitigation])
        for metric_name, group_values in rec_metrics_after.items():
            for group_name, value in group_values.items():
                writer_results.writerow([metric_name, group_name, np.mean(value), mitigation])
    results_file.close()
    distribution_file.close()
    exit()

    mitigation = "Soft-Mitigation ED"
    with open("./results/" + path_data.agent_topk + ".csv", "w+") as file:
        writer = csv.writer(file)
        header = ["alpha", "metric", "group", "data", "opt"]
        writer.writerow(header)
        for alpha in [1.]: #[0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.]:
            print("--- AFTER {} optimization alpha={}---".format(mitigation, alpha))
            maximize_ED_optimize_ETR(path_data)
            rec_metrics_after = measure_rec_quality(path_data)
            print_rec_metrics(rec_metrics_after)
            exp_metrics_after = {}
            exp_metrics_after["ETR"] = avg_time_relevance(path_data).avg_groups_time_relevance
            exp_metrics_after["ES"] = avg_explaination_serentipety(path_data).avg_groups_explaination_serentipety
            exp_metrics_after["ED"] = avg_diversity_score(path_data).avg_groups_explaination_diversity
            print_expquality_metrics(exp_metrics_after["ETR"],
                          exp_metrics_after["ES"],
                          exp_metrics_after["ED"])
            print("\n")
            for metric_name, group_values in exp_metrics_after.items():
                for group_name, value in group_values.items():
                    writer.writerow([alpha, metric_name, group_name, value, "soft_etr"])
            for metric_name, group_values in rec_metrics_after.items():
                for group_name, value in group_values.items():
                    writer.writerow([alpha, metric_name, group_name, np.mean(value), "soft_etr"])
        file.close()

        # uid_gender, gender2name = get_user2gender()
        # avg_scores = {}
        # for uid, pred_paths in path_data.best_pred_paths.items():
        #     uid_scores = []
        #     gender_value = uid_gender[uid]
        #     gender_name = gender2name[gender_value]
        #     for pred_pid, pred_path in pred_paths.items():
        #         uid_scores.append(pred_path[0][0])
        #     if gender_name not in avg_scores:
        #         avg_scores[gender_name] = []
        #     avg_scores[gender_name].append(np.mean(uid_scores))
        # print("Average scores for best paths Male: {:.3f}, Female: {:.3f}".format(np.mean(avg_scores["Male"]),
        #                                                                           np.mean(avg_scores["Female"])))
        #
        # scores_male = []
        # scores_female = []
        # for uid, topk in path_data.uid_topk.items():
        #     gender = uid_gender[uid]
        #     topk_scores = []
        #     for pid in topk:
        #         topk_scores.append(path_data.best_pred_paths[uid][pid][0][0])
        #     if gender == 0:
        #         scores_male.append(np.mean(topk_scores))
        #     else:
        #         scores_female.append(np.mean(topk_scores))
        # print(
        #     "Average scores in topk Male: {:.3f}, Female: {:.3f}".format(np.mean(scores_male), np.mean(scores_female)))


    #f.close()
    '''