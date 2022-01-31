# Incorporating Quality of Explanations in Recommender Systems with Knowledge Graphs
This repository contains the source code of the paper "Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations", where we proposed three quantitive explanation metrics and proposed a framework for path-based explanable RCMSYS over KG capable of optimizing both explanbility quality and recommandation quality. 

THE OTHER BASELINES ARE LOCATED IN THE OTHER REPOSITORY: [https://anonymous.4open.science/r/KA-RC-Baselines-0652/README.md](https://anonymous.4open.science/r/KA-RC-Baselines-0652/README.md)

# Table of Content
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Paths](#paths)
  * [Requirements for Alpha Optimization](#requirements-for-alpha-optimization)
    + [pred_paths.csv](#pred-pathscsv)
  * [Requirements for Soft Optimization and Baseline Evaluation](#requirements-for-soft-optimization-and-baseline-evaluation)
    + [uid_topk.csv](#uid-topkcsv)
    + [uid_pid_explanation.csv](#uid-pid-explanationcsv)
- [Usage](#usage)
- [Supplementary Material](#supplementary-material)
  * [Double metric weighted optimization heatmap.](#double-metric-weighted-optimization-heatmap)
  * [Soft-Optimizations Results.](#soft-optimizations-results)
    + [ML1M](#ml1m)
    + [LASTFM](#lastfm)
  * [Age Fairness (NDCG, LIR, SEP, ETD).](#age-fairness--ndcg--lir--sep--etd-)
    + [ML1M](#ml1m-1)
    + [LASTFM](#lastfm-1)
- [References](#references)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


# Dataset
The two dataset used have records of sensible attributes of users and they are **Movielens 1million (ML1M)** a well known dataset for movie recommandation and a subset of **LAST-FM 1billion (LAST-FM)** for music recommandation.  
The correspondet Knowledge Graph completion derive from two important state of art explanable recommender system, **Joint-KG \[2\]** for ml1m and **KGAT \[3\]** for Last-FM.  
The datasets are preprocessed with the dataset_mapper.py in order to compute various mappings, clean the data and formatting it for being read from our baseline model.  
You can download the preprocessed dataset directly from there: [preprocessed-datasets](https://drive.google.com/file/d/1yRaGIsAkxrifhdusC7rvdo6zgzJ0K0D6/view?usp=sharing). The datasets folder must stay in "./\<main-project-folder\>/"  
If you wish to download the original datasets you can find them there [ML1M](https://grouplens.org/datasets/movielens/) [LAST-FM](http://www.cp.jku.at/datasets/LFM-1b/).

# Requirements
python >= 3.6  

You can install the other requirements using: 
```
pip install -r requirements.txt
```
# Paths
In order to apply an optimization or measure a baseline with our proposed metrics you will need to store the paths in csv files with "," as delimiter.  

If you want to reproduce the results you can download the already computed paths from here: [ML1M](https://drive.google.com/file/d/1b6HgNJvHGPZs6q3PMaMBHT89pW46Lw7J/view?usp=sharing) [LAST-FM](https://drive.google.com/file/d/1gf9TyRN39Tc0I8immOzn9FK3e14pUpvi/view?usp=sharing) paths. The path files must stay in the following location: "\<main-project-dir\>/paths/agent-topk=\<your-agent-topk\>/\<dataset-name\>/"
If you wish to apply in-train mitigation on the baseline, produce more paths or change the metaparameters you can retrain it and produce the paths, they will be automatically saved using the path_extractor.py file.  
You can downloaded the precomputed TransE embeddings, agent-policy and agent cpkt used for the experiments from there: [ML1M](https://drive.google.com/file/d/1HWp7I-0qW1XesUE_WZ6nZ0DHFALnfRrJ/view?usp=sharing) [LAST-FM](https://drive.google.com/file/d/17EUgh299U8y0bqPYT39sdMzhjjzlahSG/view?usp=sharing). This files must stay in the following location: "\<main-project-dir\>/models/PGPR/tmp/\<dataset-name\>"  
Instead if you wish to use this framework with other path-based explanable algorithm make sure to extract the paths and have them on this form:
## Requirements for Alpha Optimization
In order to performe the reranking you would need a pred_path.csv. Files must follow this format:    

### pred_paths.csv  

|user_id|product_id|path_score|path_prob|path|
|---|---|---|---|---|
|4942|998|0.6242884741109818|1.6026814|self_loop user 4942 watched movie 328 produced_by_producer producer 197 produced_by_producer movie 998 | 
|...|...|...|...|...|


The path_score is usually computed in most of the path-based baselines usually is a score given by the KG embeddings.  
The path_prob is not mandatory, if your baseline doesn't produce probability just use -1 as a placeholder, since our reranking doesn't use it.  
The path must have len 3 in order for our algorithm to individuate the interaction, the related entity and recommendation properly.  

## Requirements for Soft Optimization and Baseline Evaluation
If you want to perfome also the evaluation of the baseline using our proposed metric or you want to apply a soft optimization you will need also a:

### uid_topk.csv
|uid|	top10|
|---|---|
|1|	946 518 513 309 742 93 31 944 274 417|
|...|...|
### uid_pid_explanation.csv

|uid|	pid|	path|
|---|---|---|
|1	|946	|self_loop user 1 watched movie 2289 watched user 266 watched movie 946|
|...|...|...|


# Usage
If you wish to execute the adapted PGPR baseline \[1\] refer to the original documentation [HERE](https://github.com/orcax/PGPR)

To perfome the optimization:
```
python main.py --dataset=dataset_name --opt=opt_name
```

You can define which optimization to use, the alpha value and more using these flags:
- --dataset: One between {ml1m, lastfm}

- --opt: One of ["softED", "softES", "softETR", "EDopt", "ESopt", "ETRopt", "ED_ES_opt", "ED_ETR_opt", "ES_ETR_opt", "ED_ES_ETR_opt"]

- --alpha: Determine the weigth of the optimized explaination metric/s in reranking, -1 means test all alpha from 0. to 1. at step of 0.05

- --eval_baseline:   If True compute rec quality metrics and explaination quality metrics from the extracted paths

- --log_enabled:   If true save log files instead of printing results

- --save_baseline_rec_quality_avgs: If true save a csv with the average baseline values for rec metrics and groups

- --save_baseline_exp_quality_avgs: If true save a csv with the average baseline values for exp metrics and groups

- --save_baseline_rec_quality_distributions: If true save a csv with the distribution of baseline values for the rec metrics and groups

- --save_baseline_exp_quality_distributions: If true save a csv with the distribution of baseline values for the exp metrics and groups

- --save_after_rec_quality_avgs: If true save a csv with the distribution of after-opt values for rec metrics and groups

- --save_after_exp_quality_avgs: If true save a csv with the distribution of after-opt values for exp metrics and groups

- --save_after_rec_quality_distributions: If true save a csv with the distribution of after-opt values for the rec metrics and groups

- --save_after_exp_quality_distributions: If true save a csv with the distribution of after-opt values for the exp metrics and groups

- --save_overall: If true saves the avgs and distribution also for the overall group

# Supplementary Material


## Double metric weighted optimization heatmap.

![Heatmap double metric weighted optimization](https://ibb.co/nCpBt5n)

## Soft-Optimizations Results.

### ML1M
|   |  NDCG | EXP  |  LIR |  SEP |  ETD | 
|---|---|---|---|---|---|
|PGPR |      0.33 | 0.81 | 0.43 | 0.26 | 0.12   |
|T-PGPR |    0.33 | 1.17 | 0.67 | 0.33 | 0.16   |
|P-PGPR |    0.33 | 1.03 | 0.44 | 0.41 | 0.17   |
|D-PGPR |    0.33 | 1.00 | 0.44 | 0.34 | 0.20   |

### LASTFM
|   |  NDCG | EXP  |  LIR |  SEP |  ETD | 
|---|---|---|---|---|---|
|PGPR |     0.15 | 1.07 | 0.56 | 0.38 | 0.13  |
|T-PGPR |   0.15 | 1.35 | 0.79 | 0.41 | 0.14  |
|P-PGPR |   0.15 | 1.26 | 0.55 | 0.54 | 0.16   |
|D-PGPR |   0.15 | 1.13 | 0.56 | 0.40 | 0.17  |

## Age Fairness (NDCG, LIR, SEP, ETD).
Average difference between age groups for the 3 metrics.
### ML1M
|   |  delta NDCG| delta LIR| delta SEP| delta ETD|
|---|---|---|---|---|
|PGPR |   0.006 | -0.027 | -0.010 | 0.000  |
|T-PGPR |   0.004 | -0.016 | -0.010 | -0.004 |
|P-PGPR |   0.006 | -0.012 | -0.023 | -0.002 |
|D-PGPR |   0.035 | -0.025 | -0.007 | -0.012 |
|DP-PGPR |  0.043 | -0.027 | -0.012 | -0.004 |
|PR-PGPR |  0.018 | -0.025 | 0.002 | -0.005  |
|DR-PGPR |  -0.002 | 0.002 | -0.010 | -0.013 |
|DPR-PGPR |  0.035 | -0.019 | -0.008 | -0.005|

        
### LASTFM
|   |  delta NDCG| delta LIR| delta SEP| delta ETD|
|---|---|---|---|---|
|PGPR     |  0.008 | 0.049 | -0.020 | -0.014 |
|T-PGPR   | 0.007  | 0.018 | -0.025 | -0.017 |
|P-PGPR   | 0.010  | 0.039 | -0.020 | -0.015 |
|D-PGPR   | -0.007 | 0.045 | -0.015 | -0.080 |
|DP-PGPR  | -0.006 | 0.034 | -0.013 | -0.054 |
|PR-PGPR  | 0.009  | 0.020 | -0.027 | -0.017 |
|DR-PGPR  | -0.003 | 0.030 | -0.018 | -0.076 |
|DPR-PGPR |  0.004 | 0.028 | -0.021 | -0.049 |

# References
\[1\] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, and Yongfeng Zhang. 2019. Reinforcement knowledge graph reasoning for explainable recommendation. In Proceedings of the 42nd International ACM SIGIR (Paris, France) https://github.com/orcax/PGPR 
\[2\] Cao, Yixin and Wang, Xiang and He, Xiangnan and Hu, Zikun and Chua Tat-seng. 2019. Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preference https://github.com/TaoMiner/joint-kg-recommender
\[3\] Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.  https://github.com/xiangwang1223/knowledge_graph_attention_network
