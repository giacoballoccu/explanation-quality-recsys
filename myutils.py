import csv

# Dataset names.
import gzip

#from mapper import get_movie_mapping, get_mapping
import pickle
import sys

from models.PGPR.utils import get_entities_without_user
from models.PGPR import knowledge_graph
ML1M = 'ml1m'
LASTFM = 'lastfm'

# Models

# Dataset directories.
KG_COMPLETATION_DATASET_DIR = {
    ML1M: './datasets/ml1m/joint-kg',
    LASTFM: './datasets/lastfm/kg-completion'
}

DATASET_DIR = {
    ML1M: './datasets/ml1m',
    LASTFM: './datasets/lastfm'
}

LABELS_DIR = {
    ML1M: {
        "kg": "models/PGPR/tmp/ml1m/kg.pkl",
        "train": "models/PGPR/tmp/ml1m/train_label.pkl",
        "test": "models/PGPR/tmp/ml1m/test_label.pkl",
    },
    LASTFM: {
        "kg": "models/PGPR/tmp/lastfm/kg.pkl",
        "train": "models/PGPR/tmp/lastfm/train_label.pkl",
        "test": "models/PGPR/tmp/lastfm/test_label.pkl",
    }
}

PGPR_MODEL_DIR = "models/PGPR"


# Selected relationships.
SELECTED_RELATIONS = {
    ML1M: [0, 1, 8, 10, 14, 15, 16, 18],
    LASTFM: [0, 1, 2, 3,4, 5, 6, 7, 8]
}

TOTAL_PATH_TYPES = {
    ML1M: len(SELECTED_RELATIONS[ML1M]),
    LASTFM: len(SELECTED_RELATIONS[LASTFM]),
}
PATH_TYPES = {
    ML1M: ['watched', 'directed_by', 'produced_by_company', 'produced_by_producer', 'starring', 'belong_to', 'edited_by', 'wrote_by', 'cinematography'],
    LASTFM: ['listened', 'belong_to', 'related_to','sang_by', 'mixed_by', 'produced_by_producer', 'original_version_of' ,'related_to', 'alternative_version_of', 'featured_by']

}
# Model result directories.
TMP_DIR = {
    ML1M: './tmp/ml1m',
    LASTFM: './tmp/lastfm'
}


def get_user2gender(dataset_name):
    file = open(DATASET_DIR[dataset_name] + "/mappings/user2gender_map.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_gender = {}
    gender2name = {-1: "All", 0: "Male", 1: "Female"}
    for row in csv_reader:
        row = row[0].strip().split('\t')
        uid_gender[int(row[0])] = 0 if row[1] == 'M' else 1
    return uid_gender, gender2name

def get_user2age(dataset_name):
    file = open(DATASET_DIR[dataset_name] + "/mappings/user2age_map.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_age = {}
    age2name = {-1: "All", 1:  "Under 18", 18:  "18-24", 25:  "25-34", 35:  "35-44", 45:  "45-49", 50:  "50-55", 56:  "56+"}
    for row in csv_reader:
        row = row[0].strip().split('\t')
        uid_age[int(row[0])] = int(row[1])
    return uid_age, age2name

def get_user2occupation(dataset_name):
    file = open(DATASET_DIR[dataset_name] + "/mappings/user2occupation_map.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_occ = {}
    occ2name = {-1: "All", 0:  "other", 1:  "academic/educator",  2:  "artist",  3:  "clerical/admin",  4:  "college/grad student",  5:  "customer service",  6:  "doctor/health care",  7:  "executive/managerial",  8:  "farmer",  9:  "homemaker", 10:  "K-12 student", 11:  "lawyer", 12:  "programmer", 13:  "retired", 14:  "sales/marketing", 15:  "scientist", 16:  "self-employed", 17:  "technician/engineer", 18:  "tradesman/craftsman", 19:  "unemployed", 20:  "writer"}
    for row in csv_reader:
        row = row[0].strip().split('\t')
        uid_occ[int(row[0])] = int(row[1])
    return uid_occ, occ2name

def get_review_uid_kg_uid_mapping(dataset_name):
    review_uid_kg_uid = {}
    with open(DATASET_DIR[dataset_name] + "/mappings/review_uid_kg_uid_mapping.txt", 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)
        for row in reader:
            uid_review = int(row[0])
            uid_kg = int(row[1])
            review_uid_kg_uid[uid_review] = uid_kg
    return review_uid_kg_uid

def get_interaction2timestamp(dataset_name):
    user2timestamp = {}
    interaction2timestamp = {}
    dataset2kg = get_dataset2kgid_mapping(dataset_name)
    file = open(DATASET_DIR[dataset_name] + "/train.txt", 'r')
    csv_reader = csv.reader(file, delimiter=' ')
    if dataset_name == "lastfm":
        uid_mapping = get_review_uid_kg_uid_mapping(dataset_name)
    for row in csv_reader:
        uid = int(row[0]) if dataset_name == "ml1m" else uid_mapping[int(row[0])]
        movie_id_ml = int(row[1])
        if movie_id_ml not in dataset2kg: continue
        movie_id_kg = dataset2kg[movie_id_ml]
        timestamp = int(row[3]) if dataset_name == "ml1m" else int(row[2])
        if uid not in user2timestamp:
            user2timestamp[uid] = []
        if uid not in interaction2timestamp:
            interaction2timestamp[uid] = {}
        user2timestamp[uid].append(timestamp)
        interaction2timestamp[uid][movie_id_kg] = timestamp
    return interaction2timestamp, user2timestamp


#Return the mapping between the id of the entity in the knowledge graph and his original entity id from the jointkg
def get_mapping(dataset_name, entity_name, old_id_as_key=False):
    mapping = {}
    file = open(DATASET_DIR[dataset_name] + "/mappings/" + entity_name + "id2dbid.txt", "r")
    csv_reader = csv.reader(file, delimiter='\n')
    next(csv_reader, None)
    for row in csv_reader:
        row = row[0].strip().split("\t")
        kg_id = int(row[0])
        old_entity_id = int(row[1])
        if old_id_as_key:
            mapping[old_entity_id] = kg_id
        else:
            mapping[kg_id] = old_entity_id
    return mapping

def get_all_entity_mappings(dataset_name):
    mappings = {}
    for entity in get_entities_without_user(dataset_name):
        if entity == 'movie':
            mappings[entity] = get_movie_mapping(dataset_name)
            continue
        if entity == 'song':
            mappings[entity] = get_song_mapping(dataset_name)
            continue
        mappings[entity] = get_mapping(dataset_name, entity, True)
    return mappings

def get_movie_mapping(dataset_name):
    valid = {}
    file = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "r")
    csv_reader = csv.reader(file, delimiter='\n')
    for row in csv_reader:
        row = row[0].strip().split("\t")
        valid[int(row[2])] = [int(row[0]), int(row[1])] #key: entityid, value: {kgid, movielandid}
    return valid

def get_song_mapping(dataset_name):
    valid = {}
    file = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "r")
    csv_reader = csv.reader(file, delimiter='\n')
    next(csv_reader, None)
    for row in csv_reader:
        row = row[0].strip().split("\t")
        valid[int(row[2])] = [int(row[0]), int(row[1])] #key: entityid, value: {kgid, movielandid}
    return valid

def get_dataset2kgid_mapping(dataset_name):
    file = open("./datasets/" + dataset_name + "/mappings/product_mappings.txt", "r")
    csv_reader = csv.reader(file, delimiter='\n')
    mlId_to_kgId = {}
    next(csv_reader, None)
    for row in csv_reader:
        row = row[0].strip().split("\t")
        mlId_to_kgId[int(row[1])] = int(row[0])
    file.close()
    return mlId_to_kgId

def zip_file(filename):
    with open(filename, 'rb') as file:
        zipped = gzip.open(filename + '.gz', 'wb')
        zipped.writelines(file)
        zipped.close()
    file.close()

#Returns a string representing the path type
def get_path_type(path):
    return path[-1][0]

def get_interaction_id(path):
    return path[1][-1]

def get_rec_pid(path):
    return int(path[-1][-1][-1])

def get_related_entity(path):
    related_entity_name = path[-2][1]
    related_entity_id = int(path[-2][-1])
    return related_entity_name, related_entity_id

#Trasform a string separeted by space that rapresent the path in a list composed by triplets
def normalize_path(path_str):
    path = path_str.split(" ")
    normalized_path = []
    for i in range(0, len(path), 3):
        normalized_path.append((path[i], path[i + 1], path[i + 2]))
    return normalized_path

def get_uidreview2uidkg(dataset_name):
    file = open(DATASET_DIR[dataset_name] + "/mappings/review_uid_kg_uid_mapping.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\t')
    review_uid2kg_uid = {}
    uid_kg2uid_review = {}
    next(csv_reader, None)
    for row in csv_reader:
        review_uid2kg_uid[int(row[1])] = int(row[0])
        uid_kg2uid_review[int(row[0])] = int(row[1])
    return review_uid2kg_uid, uid_kg2uid_review

def get_user2gender(dataset_name):
    file = open(DATASET_DIR[dataset_name] + "/mappings/uid2gender.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_gender = {}
    gender2name = {-1: "Overall", 0: "Male", 1: "Female"}
    if dataset_name == "lastfm":
        _, uid_mapping = get_uidreview2uidkg(dataset_name)
    for row in csv_reader:
        row = row[0].strip().split('\t')
        if dataset_name == "ml1m":
            uid_gender[int(row[0])] = 0 if row[1] == 'M' else 1
        else:
            uid_gender[uid_mapping[int(row[0])]] = 0 if row[1] == 'm' else 1
    return uid_gender, gender2name

# Returns a dict that maps the user uid with his age
def get_user2age():
    file = open("datasets/ml1m/mappings/uid2age.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_age = {}
    age2name = {1:  "Under 18", 18:  "18-24", 25:  "25-34", 35:  "35-44", 45:  "45-49", 50:  "50-55", 56:  "56+"}
    for row in csv_reader:
        row = row[0].strip().split('\t')
        uid_age[int(row[0])] = int(row[1])
    return uid_age, age2name

# Returns a dict that maps the user uid with his occupation
def get_user2occupation():
    file = open("datasets/ml1m/mappings/uid2occupation.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_occ = {}
    occ2name = {0:  "other", 1:  "academic/educator",  2:  "artist",  3:  "clerical/admin",  4:  "college/grad student",  5:  "customer service",  6:  "doctor/health care",  7:  "executive/managerial",  8:  "farmer",  9:  "homemaker", 10:  "K-12 student", 11:  "lawyer", 12:  "programmer", 13:  "retired", 14:  "sales/marketing", 15:  "scientist", 16:  "self-employed", 17:  "technician/engineer", 18:  "tradesman/craftsman", 19:  "unemployed", 20:  "writer"}
    for row in csv_reader:
        row = row[0].strip().split('\t')
        uid_occ[int(row[0])] = int(row[1])
    return uid_occ, occ2name

def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS_DIR[dataset][mode]
        # CHANGED
    elif mode == 'test':
        label_file = LABELS_DIR[dataset][mode]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products

def load_kg(dataset):
    kg_file = LABELS_DIR[dataset]["kg"]
    # CHANGED
    sys.path.append(r'models/PGPR')
    kg = pickle.load(open(kg_file, 'rb'))
    return kg