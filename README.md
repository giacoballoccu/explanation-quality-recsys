# Post Processing Explanations in Path Reasoning Recommender Systems with Knowledge Graphs
This repository contains the source code of the SIGIR 2022 paper ["Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations"](https://dl.acm.org/doi/10.1145/3477495.3532041).

![Pipeline Summary](/SIGIR22-pipeline-summary.png)

In this paper we performed a mixed approach compining literature review and user's studies to explore and conceptualize the space
of relevant explanation types comprehensively. As a result of this first phase, we identified and operationalized three expla-
nation properties. Recommendations and explainable paths returned by pre-trained models were re-ranked to optimize the
explanation properties, and evaluated on recommendation utility, explanation quality, and fairness.

This framework can be trasfered to any Path Reasoning Explainable Recommender Systems, provided that you save the predicted paths candidates in csv files standardized as shown in the upcoming sections. 

The other baselines are located in the other repository: [Knowlede-Aware-Recommender-Systems-Baselines](https://github.com/giacoballoccu/Knowlede-Aware-Recommender-Systems-Baselines)

# Table of Content
- [Acknowledgement](#acknowledgement)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Precomputed Paths](#precomputed-paths)
  * [Requirements for Weigthed Optimization](#requirements-for-weigthed-optimization)
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


# Acknowledgement
Any scientific publications that use our datasets should cite the following paper as the reference:
```
@inproceedings{10.1145/3477495.3532041,
author = {Balloccu, Giacomo and Boratto, Ludovico and Fenu, Gianni and Marras, Mirko},
title = {Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3532041},
doi = {10.1145/3477495.3532041},
abstract = {Existing explainable recommender systems have mainly modeled relationships between recommended and already experienced products, and shaped explanation types accordingly (e.g., movie "x" starred by actress "y" recommended to a user because that user watched other movies with "y" as an actress). However, none of these systems has investigated the extent to which properties of a single explanation (e.g., the recency of interaction with that actress) and of a group of explanations for a recommended list (e.g., the diversity of the explanation types) can influence the perceived explaination quality. In this paper, we conceptualized three novel properties that model the quality of the explanations (linking interaction recency, shared entity popularity, and explanation type diversity) and proposed re-ranking approaches able to optimize for these properties. Experiments on two public data sets showed that our approaches can increase explanation quality according to the proposed properties, fairly across demographic groups, while preserving recommendation utility. The source code and data are available at https://github.com/giacoballoccu/explanation-quality-recsys.},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {646–656},
numpages = {11},
keywords = {knowledge graphs, fairness, recommender systems, explainability},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```

Nobody guarantees the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:
* The user must acknowledge the use of the data set in publications resulting from the use of the data set.
* The user may not redistribute the data without separate permission.
* The user may not try to deanonymise the data.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.


# Datasets
Two datasets with sensible attributes Movielens 1 Million (ML1M) and a subset of LAST-FM 1b.

The datasets were augmented using the Knowledge Graph completion extracted from Freebase and DBpedia from two stat-of-the-art knowledge-aware recommender systems, **Joint-KG \[2\]** for ML1M and **KGAT \[3\]** for LAST-FM.  

The datasets are preprocessed with the dataset_mapper.py in order to compute various mappings, clean the data and formatting it for being read from our baseline model. All the details are avaiable in the manuscript.

You can download the preprocessed dataset directly from there: [preprocessed-datasets](https://drive.google.com/file/d/1yRaGIsAkxrifhdusC7rvdo6zgzJ0K0D6/view?usp=sharing). The datasets folder must stay in "./\<main-project-folder\>/" 

If you wish to download the original datasets you can find them there [ML1M](https://grouplens.org/datasets/movielens/) [LAST-FM](http://www.cp.jku.at/datasets/LFM-1b/).

# Requirements
python >= 3.6  

You can install the other requirements using: 
```
pip install -r requirements.txt
```

# Precomputed Paths
In order to apply an optimization or measure a baseline with our proposed metrics you will need to store the predicted paths in csv files with "," as delimiter. In general we distinguish 3 files:
- pred_paths.csv: Which contains all the extracted paths for users (Candidate Selection Step).  
- uid_topk: Which containes the top-k recommended items for every user.  
- uid_pid_explanation.csv: Which containes the top-k recommended items and the associated explaination path.  

**Precomputed Paths**: If you want to reproduce the results you can download the already computed paths from here: [ML1M](https://drive.google.com/file/d/1b6HgNJvHGPZs6q3PMaMBHT89pW46Lw7J/view?usp=sharing) [LAST-FM](https://drive.google.com/file/d/1gf9TyRN39Tc0I8immOzn9FK3e14pUpvi/view?usp=sharing) paths. The path files must stay in the following location: "\<main-project-dir\>/paths/agent-topk=\<your-agent-topk\>/\<dataset-name\>/"

**Retrain Original Model**: If you wish to apply in-train mitigation on the baseline, produce more paths or change the metaparameters you can retrain it and produce the paths, they will be automatically saved using the path_extractor.py file.  

**Precomputed TransE Embeddings**: You can downloaded the precomputed TransE embeddings, agent-policy and agent cpkt used for the experiments from there: [ML1M](https://drive.google.com/file/d/1HWp7I-0qW1XesUE_WZ6nZ0DHFALnfRrJ/view?usp=sharing) [LAST-FM](https://drive.google.com/file/d/17EUgh299U8y0bqPYT39sdMzhjjzlahSG/view?usp=sharing). This files must stay in the following location: "\<main-project-dir\>/models/PGPR/tmp/\<dataset-name\>"  

**Trasfer the framework to other baselines**: Instead if you wish to use this framework with other path-based explanable algorithm make sure to extract the paths and have them on this form:

## Requirements for Weigthed Optimization
In order to performe the reranking you would need a pred_path.csv. Files must follow this format:    

### pred_paths.csv  

|user_id|product_id|path_score|path_prob|path|
|---|---|---|---|---|
|4942|998|0.6242884741109818|1.6026814|self_loop user 4942 watched movie 328 produced_by_producer producer 197 produced_by_producer movie 998 | 
|...|...|...|...|...|


The **path_score** is usually computed using KG embedding tecniques.  

The **path_prob** is not mandatory, if your baseline doesn't produce probability just use -1 as a placeholder, since our reranking doesn't use it.  

The **path must have length of 3** in order for our algorithm to individuate the interaction, the related entity and recommendation properly.  

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

Where opt is one of: ``` ["softETD", "softSEP", "softLIR", "ETDopt", "SEPopt", "LIRopt", "ETD_SEP_opt", "ETD_LIR_opt", "SEP_LIR_opt", "ETD_SEP_LIR_opt"] ```


You can define which optimization to use, the alpha value and more using these flags:   
- ``` --dataset```, string, dataset to use one among {ml1m, lastfm}    
- ``` --agent_topk```, string, determine which agent top-k folder from which take the predicted paths.
- ``` --opt```, string, determine which type of optimization one among {soft, weighted}
- ``` --alpha```, float, used only by weighted optimization, determine the weighting coefficient for the explanation metric in reranking process.
- ```--metrics'```, list, determine which metric optimize, possible values are: {LIR, SEP, ETD, [ETD, LIR], [ETD, SEP], [SEP, LIR], [SEP, LIR, ETD]}
- ```--eval_baseline'```, boolean, determine if calculate the stats for baseline or not.  
- ```--log_enabled'```, boolean, if true save outputs in log files instead of printing results.

More flags can be find in main.py args.


# Supplementary Material

## Double metric weighted optimization heatmap.

[![Double metric weighted optimization heatmap.](https://www.linkpicture.com/q/sens-attributes-doubleopt-interplay-heatmap-1_2.png)](https://www.linkpicture.com/view.php?img=LPic61f7ffcc930d7633188407)

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

