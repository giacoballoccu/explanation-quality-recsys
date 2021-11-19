import argparse
import os

from metrics import *
from optimizations import *

from path_data_loader import PathDataLoader

if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="lastfm", help='One of {ml1m, lastfm}')
    parser.add_argument('--agent_topk', type=str, default="25-50-1", help='One of {25-50-1, 10-12-1} or more if you compute the predpaths with PGPR')
    parser.add_argument('--opt', type=str, default="LIRopt", help='One of ["softETD", "softSEP", "softLIR", "ETDopt", "SEPopt", "LIRopt", "ETD_SEP_opt", "ETD_LIR_opt", "SEP_LIR_opt", "ETD_SEP_LIR_opt"]')
    parser.add_argument('--alpha', type=float, default=-1, help="Determine the weigth of the optimized explanation metric/s in reranking, -1 means test all alpha from 0. to 1. at step of 0.05")
    parser.add_argument('--eval_baseline', type=bool, default=False, help='If True compute rec quality metrics and explanation quality metrics from the extracted paths')
    parser.add_argument('--log_enabled', type=bool, default=True, help='If true save log files instead of printing results')
    parser.add_argument('--save_baseline_rec_quality_avgs', type=bool, default=True, help='If true save a csv with the average baseline values for rec metrics and groups')
    parser.add_argument('--save_baseline_exp_quality_avgs', type=bool, default=True, help='If true save a csv with the average baseline values for exp metrics and groups')
    parser.add_argument('--save_baseline_rec_quality_distributions', type=bool, default=True, help='If true save a csv with the distribution of baseline values for the rec metrics and groups')
    parser.add_argument('--save_baseline_exp_quality_distributions', type=bool, default=True, help='If true save a csv with the distribution of baseline values for the exp metrics and groups')
    parser.add_argument('--save_after_rec_quality_avgs', type=bool, default=True, help='If true save a csv with the distribution of after-opt values for rec metrics and groups')
    parser.add_argument('--save_after_exp_quality_avgs', type=bool, default=True, help='If true save a csv with the distribution of after-opt values for exp metrics and groups')
    parser.add_argument('--save_after_rec_quality_distributions', type=bool, default=True, help='If true save a csv with the distribution of after-opt values for the rec metrics and groups')
    parser.add_argument('--save_after_exp_quality_distributions', type=bool, default=True, help='If true save a csv with the distribution of after-opt values for the exp metrics and groups')
    parser.add_argument('--save_overall', type=bool, default=True, help='If true saves the avgs and distribution also for the overall group')
    args = parser.parse_args()

    sys.path.append(r'models/PGPR')

    #Creation of results folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/" + args.dataset):
        os.makedirs("./results/" + args.dataset)
    if not os.path.exists("./results/" + args.dataset + "/agent_topk=" + args.agent_topk):
        os.makedirs("./results/" + args.dataset + "/agent_topk=" + args.agent_topk)
    result_base_path = "./results/" + args.dataset + "/agent_topk=" + args.agent_topk + "/"

    #Creation of log folders
    if not os.path.exists("./log"):
        os.makedirs("./log")
    if not os.path.exists("./log/" + args.dataset):
        os.makedirs("./log/" + args.dataset)
    if not os.path.exists("./log/" + args.dataset + "/agent_topk=" + args.agent_topk):
        os.makedirs("./log/" + args.dataset + "/agent_topk=" + args.agent_topk)
    log_base_path = "./log/" + args.dataset + "/agent_topk=" + args.agent_topk + "/"

    soft_optimizations = ["softETD", "softSEP", "softLIR"]
    alpha_optimizations = ["ETDopt", "SEPopt", "LIRopt", "ETD_SEP_opt", "ETD_LIR_opt", "SEP_LIR_opt", "ETD_SEP_LIR_opt"]

    #Load paths
    path_data = PathDataLoader(args)

    '''uid_gender, gender2name = get_user2gender(path_data.dataset_name)
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
    train_size_male = [(n * 100) / 20 for n in n_male]
    train_size_female = [(n * 100) / 20 for n in n_female]
    print("Average train size for {:.3f}, average train size for female {:.3f}".format(np.mean(train_size_male),
                                                                                       np.mean(train_size_female)))
    print("Average test size for male {:.3f}, average test size for female {:.3f}".format(np.mean(n_male),
                                                                                          np.mean(n_female)))

    n_labels_keys = [x for x in n_labels_uid.keys()]
    n_labels_keys.sort()
    differences = []
    carry_male = []
    carry_female = []
    for size_test in n_labels_keys:
        uids = n_labels_uid[size_test]
        metrics = measure_rec_quality_group(path_data, uids)
        if metrics.n_female == 0 and carry_male == []:
            carry_male = metrics.ndcg["Male"]
            continue
        if metrics.n_male == 0 and carry_male == []:
            carry_female = metrics.ndcg["Female"]
            continue
        if metrics.n_female == 0 and carry_male != []:
            carry_male += metrics.ndcg["Male"]
            continue
        if metrics.n_male == 0 and carry_male != []:
            carry_female += metrics.ndcg["Female"]
            continue
        ndcg_male = metrics.ndcg["Male"] + carry_male
        ndcg_female = metrics.ndcg["Female"] + carry_female
        print("User with train_size={}, test_size={}, Male: {}, Female: {}, Total: {}".format((100 * size_test) / 20,
                                                                                              size_test, metrics.n_male,
                                                                                              metrics.n_female,
                                                                                              metrics.n_male + metrics.n_female))
        print("NDCG Male: {:.3f}, NDCG Female: {:.3f}, NDCG Total: {:.3f} DIFF(Male-Female): {:.3f}".format(
            np.mean(ndcg_male), np.mean(ndcg_female), np.mean(metrics.ndcg["Overall"]),
            (np.mean(ndcg_male) - np.mean(ndcg_female))
        ))
        carry_male = []
        carry_female = []
        print()
        diff = np.mean(ndcg_male) - np.mean(ndcg_female)
        if np.math.isnan(diff): continue
        differences.append(diff)
    print("Mean: {:.7f}".format(np.mean(differences)))

    exit(0)'''

    #Enable logging to file
    if args.log_enabled == True and args.eval_baseline:
        log_path = log_base_path + "/baseline.txt"
        log_file = open(log_path, "w+")
        sys.stdout = log_file

    #BASELINE
    if args.eval_baseline == True:
        print("--- Baseline---")
        #Rec Quality
        rec_metrics_before = measure_rec_quality(path_data)
        print_rec_metrics(path_data.dataset_name,rec_metrics_before)
        #Exp Quality
        exp_metrics_before = {}
        distributions_exp_metrics_before = {}
        # Save average of values in topk for each metric
        tr_before_mitigation = avg_LIR(path_data)
        es_before_mitigation = avg_SEP(path_data)
        ed_before_mitigation = avg_ETD(path_data)

        exp_metrics_before["LIR"] = dict(tr_before_mitigation.avg_groups_LIR)
        exp_metrics_before["SEP"] = dict(es_before_mitigation.avg_groups_SEP)
        exp_metrics_before["ETD"] = dict(ed_before_mitigation.avg_groups_ETD)
        # Save distributions of values in topk for each metric
        distributions_exp_metrics_before["LIR"] = dict(tr_before_mitigation.groups_LIR_scores)
        distributions_exp_metrics_before["SEP"] = dict(es_before_mitigation.groups_SEP_scores)
        distributions_exp_metrics_before["ETD"] = dict(ed_before_mitigation.groups_ETD_scores)
        print_expquality_metrics(path_data.dataset_name, exp_metrics_before["LIR"],
                                 exp_metrics_before["SEP"],
                                 exp_metrics_before["ETD"])

        # Initialize file to save .csv with avg values of topk recommandation quality metrics
        if args.save_baseline_rec_quality_avgs or args.save_baseline_exp_quality_avgs:
            filename = result_base_path + "baseline_avg.csv"
            avg_metrics_file = open(filename, 'w+')
            writer = csv.writer(avg_metrics_file)
            header = ["alpha", "metric", "group", "data", "opt"]
            writer.writerow(header)
            # Write on file avg values for rec quality metrics after optimization
            for alpha in [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,0.80, 0.85, 0.90, 0.95, 1.]:
                if args.save_baseline_rec_quality_avgs:
                    for metric_name, group_values in rec_metrics_before.items():
                        for group_name, value in group_values.items():
                            writer.writerow([alpha, metric_name, group_name, np.mean(value), "baseline"])
                # Write on file avg values for exp quality metrics after optimization
                if args.save_baseline_exp_quality_avgs:
                    for metric_name, group_values in exp_metrics_before.items():
                        for group_name, value in group_values.items():
                            writer.writerow([alpha, metric_name, group_name, value, "baseline"])
            avg_metrics_file.close()
        # Initialize file to save .csv with avg values of topk recommandation quality metrics
        if args.save_baseline_rec_quality_distributions or args.save_baseline_exp_quality_distributions:
            filename = result_base_path + "baseline_distribution.csv"
            avg_distribution_file = open(filename, 'w+')
            writer_distribution = csv.writer(avg_distribution_file)
            header = ["metric", "group", "data", "opt"]
            writer_distribution.writerow(header)
            # Write distribution of values for topk rec quality metrics
            if args.save_baseline_rec_quality_distributions:
                for metric_name, group_avg_values in rec_metrics_before.items():
                    for group_name, values in group_avg_values.items():
                        if args.save_overall and group_name == "Overall": continue
                        for value in values:
                            writer_distribution.writerow([metric_name, group_name, value, "baseline"])
            # Write distribution of values for topk exp quality metrics
            if args.save_baseline_exp_quality_distributions:
                for metric_name, group_values in distributions_exp_metrics_before.items():
                    for group_name, values in group_values.items():
                        if args.save_overall and group_name == "Overall": continue
                        for value in values:
                            writer_distribution.writerow([metric_name, group_name, value, "baseline"])


    #Optimization
    chosen_optimization = args.opt
    if chosen_optimization not in alpha_optimizations and chosen_optimization not in soft_optimizations:
        print("The chosen optimization doesn't exist...")


    #Performing Soft-Optimization

    if chosen_optimization in soft_optimizations:


        for optimization in soft_optimizations:
            if args.log_enabled == True:
                log_path = log_base_path + chosen_optimization + ".txt"
                log_file = open(log_path, "w+")
                sys.stdout = log_file
            print("Performing Soft-Optimization...")
            chosen_optimization = optimization
            if chosen_optimization == "softETD":
                soft_optimization_ETD(path_data)
            elif chosen_optimization == "softSEP":
                soft_optimization_SEP(path_data)
            elif chosen_optimization == "softLIR":
                soft_optimization_LIR(path_data)

            LIR_after = avg_LIR(path_data)
            SEP_after = avg_SEP(path_data)
            ETD_after = avg_ETD(path_data)
            rec_metrics_after = measure_rec_quality(path_data)
            print_rec_metrics(path_data.dataset_name,rec_metrics_after)

            avg_exp_metrics_after = {}
            distributions_exp_metrics_after = {}

            # Save average of values in topk for each metric
            avg_exp_metrics_after["LIR"] = dict(LIR_after.avg_groups_LIR)
            avg_exp_metrics_after["SEP"] = dict(SEP_after.avg_groups_SEP)
            avg_exp_metrics_after["ETD"] = dict(ETD_after.avg_groups_ETD)

            # Save distributions of values in topk for each metric
            distributions_exp_metrics_after["LIR"] = dict(LIR_after.groups_LIR_scores)
            distributions_exp_metrics_after["SEP"] = dict(SEP_after.groups_SEP_scores)
            distributions_exp_metrics_after["ETD"] = dict(ETD_after.groups_ETD_scores)
            print_expquality_metrics(path_data.dataset_name, avg_exp_metrics_after["LIR"],
                                     avg_exp_metrics_after["SEP"],
                                     avg_exp_metrics_after["ETD"])

            # Initialize file to save .csv with avg values of topk recommandation quality metrics
            if args.save_after_exp_quality_avgs:
                filename = result_base_path + chosen_optimization + "_avg.csv"
                avg_metrics_file = open(filename, 'w+')
                writer = csv.writer(avg_metrics_file)
                header = ["metric", "group", "data", "opt"]
                writer.writerow(header)
                # Write on file avg values for exp quality metrics after optimization
                for metric_name, group_values in avg_exp_metrics_after.items():
                    for group_name, value in group_values.items():
                        if args.save_overall and group_name == "Overall": continue
                        writer.writerow([metric_name, group_name, np.mean(value), chosen_optimization])
                avg_metrics_file.close()

            # Initialize file to save .csv with distribution values of topk recommandation quality metrics
            if args.save_after_exp_quality_distributions:
                filename = result_base_path + chosen_optimization + "_distribution.csv"
                avg_distribution_file = open(filename, 'w+')
                writer_distribution = csv.writer(avg_distribution_file)
                header = ["metric", "group", "data", "opt"]
                writer_distribution.writerow(header)
                # Write distribution of values for topk exp quality metrics
                for metric_name, group_values in distributions_exp_metrics_after.items():
                    for group_name, values in group_values.items():
                        if args.save_overall and group_name == "Overall": continue
                        for value in values:
                            writer_distribution.writerow([metric_name, group_name, value, chosen_optimization])
                avg_distribution_file.close()
            log_file.close()
    if chosen_optimization in alpha_optimizations:
        for chosen_optimization in alpha_optimizations:
                #Performing Alpha-Optimization
            if args.log_enabled == True:
                log_path = log_base_path + chosen_optimization + ".txt"
                log_file = open(log_path, "w+")
                sys.stdout = log_file
            print("Performing Alpha-Optimization...")
            if args.alpha == -1:
                alphas = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,0.80, 0.85, 0.90, 0.95, 1.]
            else:
                alphas = [args.alpha]

            #Initialize file to save .csv with avg values of topk recommandation quality metrics
            if args.save_after_rec_quality_avgs or args.save_after_exp_quality_avgs:
                filename = chosen_optimization + "_moving_alpha_avg.csv" if args.alpha == -1 else chosen_optimization + "_alpha=" + args.alpha + "_avg.csv"
                file_path = result_base_path + filename
                avg_metrics_file = open(file_path, 'w+')
                writer = csv.writer(avg_metrics_file)
                header = ["alpha","metric","group","data","opt"]
                writer.writerow(header)

            # Initialize file to save .csv with distribution values of topk recommandation quality metrics
            if args.save_after_rec_quality_distributions or args.save_after_exp_quality_distributions:
                filename = chosen_optimization + "_moving_alpha_distribution.csv" if args.alpha == -1 else chosen_optimization + "_alpha=" + args.alpha + "_distribution.csv"
                file_path = result_base_path + filename
                distribution_file = open(file_path, 'w+')
                writer_distribution = csv.writer(distribution_file)
                header = ["alpha", "metric", "group", "data", "opt"]
                writer_distribution.writerow(header)

            #Apply the chosen optimization for chosen value of alpha
            for alpha in alphas:
                print("--- AFTER {} optimization with alpha={}---".format(chosen_optimization, alpha))

                if chosen_optimization == "ETDopt":
                    optimize_ETD(path_data, alpha)
                elif chosen_optimization == "SEPopt":
                    optimize_SEP(path_data, alpha)
                elif chosen_optimization == "LIRopt":
                    optimize_LIR(path_data, alpha)
                elif chosen_optimization == "ETD_SEP_opt":
                    optimize_ETD_SEP(path_data, alpha)
                elif chosen_optimization == "ETD_LIR_opt":
                    optimize_ETD_LIR(path_data, alpha)
                elif chosen_optimization == "SEP_LIR_opt":
                    optimize_LIR_SEP(path_data, alpha)
                elif chosen_optimization == "ETD_SEP_LIR_opt":
                    optimize_ETD_SEP_LIR(path_data, alpha)

                rec_metrics_after = measure_rec_quality(path_data)
                print_rec_metrics(path_data.dataset_name,rec_metrics_after)

                exp_metrics_after = {}
                distributions_exp_metrics_after = {}

                #Save average of values in topk for each metric
                tr_after_mitigation = avg_LIR(path_data)
                es_after_mitigation = avg_SEP(path_data)
                ed_after_mitigation = avg_ETD(path_data)

                # Save average of values in topk for each metric
                exp_metrics_after["LIR"] = dict(tr_after_mitigation.avg_groups_LIR)
                exp_metrics_after["SEP"] = dict(es_after_mitigation.avg_groups_SEP)
                exp_metrics_after["ETD"] = dict(ed_after_mitigation.avg_groups_ETD)

                #Save distributions of values in topk for each metric
                distributions_exp_metrics_after["LIR"] = dict(tr_after_mitigation.groups_LIR_scores)
                distributions_exp_metrics_after["SEP"] = dict(es_after_mitigation.groups_SEP_scores)
                distributions_exp_metrics_after["ETD"] = dict(ed_after_mitigation.groups_ETD_scores)
                print_expquality_metrics(path_data.dataset_name, exp_metrics_after["LIR"],
                                         exp_metrics_after["SEP"],
                                         exp_metrics_after["ETD"])

                # Write on file avg values for exp quality metrics after optimization
                if args.save_after_exp_quality_avgs:
                    for metric_name, group_values in rec_metrics_after.items():
                      for group_name, value in group_values.items():
                          if not args.save_overall and group_name == "Overall": continue
                          writer.writerow([alpha, metric_name, group_name, np.mean(value), chosen_optimization])

                # Write on file avg values for rec quality metrics after optimization
                if args.save_after_rec_quality_avgs:
                    for metric_name, group_values in exp_metrics_after.items():
                        for group_name, value in group_values.items():
                            if not args.save_overall and group_name == "Overall": continue
                            writer.writerow([alpha, metric_name, group_name, value, chosen_optimization])

                # Write distribution of values for topk rec quality metrics
                if args.save_after_rec_quality_distributions:
                    for metric_name, group_values in rec_metrics_after.items():
                        for group_name, values in group_values.items():
                            if group_name == "Overall": continue
                            for value in values:
                                writer_distribution.writerow([alpha, metric_name, group_name, value])
                # Write distribution of values for topk exp quality metrics
                if args.save_after_exp_quality_distributions:
                    for metric_name, group_values in distributions_exp_metrics_after.items():
                        for group_name, values in group_values.items():
                            if group_name == "Overall": continue
                            for value in values:
                                writer_distribution.writerow([alpha, metric_name, group_name, value])
            if args.log_enabled:
                log_file.close()
            #Close files
            if args.save_after_rec_quality_avgs or args.save_after_exp_quality_avgs:
                avg_metrics_file.close()
            if args.save_after_exp_quality_distributions or args.save_after_rec_quality_distributions:
                distribution_file.close()