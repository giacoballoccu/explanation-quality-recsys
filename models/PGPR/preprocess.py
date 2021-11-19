from __future__ import absolute_import, division, print_function

import os
import pickle
import gzip
import argparse
#from models.PGPR.utils import *
from utils import *
from data_utils import AmazonDataset
from knowledge_graph import KnowledgeGraph


def generate_labels(dataset, mode='train'):
    review_file = '{}/{}.txt.gz'.format(DATASET_DIR[dataset], mode)
    user_products = {}  # {uid: [pid,...], ...}
    id2kgid = get_product_id_kgid_mapping(dataset)
    uid2kg_uid = get_uid_to_kgid_mapping(dataset)
    with gzip.open(review_file, 'r') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            arr = line.split(' ')
            user_idx = uid2kg_uid[int(arr[0])]
            product_idx = int(arr[1])
            if product_idx not in id2kgid: continue
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(id2kgid[product_idx])
    save_labels(dataset, user_products, mode=mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='ML1M')
    args = parser.parse_args()

    # Create AmazonDataset instance for dataset.
    # ========== BEGIN ========== #
    print('Load', args.dataset, 'dataset from file...')
    if not os.path.isdir(TMP_DIR[args.dataset]):
        os.makedirs(TMP_DIR[args.dataset])
    dataset = AmazonDataset(DATASET_DIR[args.dataset])
    save_dataset(args.dataset, dataset)
    # Generate knowledge graph instance.
    # ========== BEGIN ========== #
    print('Create', args.dataset, 'knowledge graph from dataset...')
    dataset = load_dataset(args.dataset)
    kg = KnowledgeGraph(dataset)
    kg.compute_degrees()
    save_kg(args.dataset, kg)
    # =========== END =========== #

    # Generate train/test labels.
    # ========== BEGIN ========== #
    print('Generate', args.dataset, 'train/test labels.')
    generate_labels(args.dataset, 'train')
    generate_labels(args.dataset, 'test')
    # =========== END =========== #


if __name__ == '__main__':
    main()
