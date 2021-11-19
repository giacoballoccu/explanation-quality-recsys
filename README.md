# Incorporating Quality of Explanations in Recommender Systems with Knowledge Graphs
This repository contains the source code of the paper "Incorporating Quality of Explanations in Recommender Systems with Knowledge Graphs", where we proposed three quantitive explanation metrics and proposed a framework for path-based explanable RCMSYS over KG capable of optimizing both explanbility quality and recommandation quality. 
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

# References
\[1\] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, and Yongfeng Zhang. 2019. Reinforcement knowledge graph reasoning for explainable recommendation. In Proceedings of the 42nd International ACM SIGIR (Paris, France) https://github.com/orcax/PGPR 
\[2\] Cao, Yixin and Wang, Xiang and He, Xiangnan and Hu, Zikun and Chua Tat-seng. 2019. Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preference https://github.com/TaoMiner/joint-kg-recommender
\[3\] Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.  https://github.com/xiangwang1223/knowledge_graph_attention_network
