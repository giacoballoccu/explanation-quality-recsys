import argparse
import os
from easydict import EasyDict as edict
from models.PGPR.utils import get_tail_entity_name, LASTFM_RELATION_NAME
from myutils import *

#Generate the mapping from the KG Completation of KGAT completion to a PGPR readable dataset
class LastFmDatasetMapper(object):
    def __init__(self, args):
        self.args = args
        self.generate_train_test_split()
        self.generate_user_attributes_mappings()
        self.generate_kg_entities()
        self.generate_kg_relations()

    def generate_train_test_split(self):
        dataset_name = self.args.dataset
        uid_review_tuples = {}
        dataset_size = 0
        print("Loading reviews...")
        with open(DATASET_DIR[dataset_name] + "/ratings.dat", 'r', encoding='latin-1') as reviews_file:
            reader = csv.reader(reviews_file, delimiter=',')
            next(reader, None)
            for row in reader:
                uid = int(row[0])
                if uid not in uid_review_tuples:
                    uid_review_tuples[uid] = []
                uid_review_tuples[uid].append((row[0], row[3], row[4]))
                dataset_size += 1
        reviews_file.close()
        train_size = 0.8
        print("Performing split {}/{}...".format(train_size * 100, 100 - train_size * 100))
        for uid, reviews in uid_review_tuples.items():
            reviews.sort(key=lambda x: int(x[-1]))  # sorting from recent to older

        train = []
        test = []
        for uid, reviews in uid_review_tuples.items():  # python dict are sorted, 1...nuser
            n_elements_test = int(len(reviews) * train_size)
            train.append(reviews[:n_elements_test])
            test.append(reviews[n_elements_test:])

        print("Writing train...")
        with open(DATASET_DIR[dataset_name] + "/train.txt", 'w+') as file:
            for user_reviews in train:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()

        print("Writing test...")
        with open(DATASET_DIR[dataset_name] + "/test.txt", 'w+') as file:
            for user_reviews in test:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()
        print("Zipping train and test...")
        zip_file(DATASET_DIR[dataset_name] + "/train.txt")
        zip_file(DATASET_DIR[dataset_name] + "/test.txt")
        print("Loading reviews.. DONE")

    def generate_kg_relations(self):
        dataset_name = args.dataset
        mappings = get_all_entity_mappings(dataset_name)
        if dataset_name == ML1M:
            product = 'movie'
        else:
            product = 'song'
        no_of_movies = len(mappings[product])+1
        movie_id_entity = edict(
            sang_by=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/sang_by_s_a.txt'),
            featured_by = ([[] for _ in range(no_of_movies)],DATASET_DIR[dataset_name] + '/relations/featured_by_s_a.txt'),
            belong_to = ([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/belong_to_s_ca.txt'),
            mixed_by = ([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/mixed_by_s_e.txt'),
            related_to = ([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/related_to_s_rs.txt'),
            alternative_version_of = ([[] for _ in range(no_of_movies)],DATASET_DIR[dataset_name] + '/relations/alternative_version_of_s_rs.txt'),
            original_version_of = ([[] for _ in range(no_of_movies)],DATASET_DIR[dataset_name] + '/relations/orginal_version_of_s_rs.txt'),
            produced_by_producer=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/produced_by_producer_s_pr.txt'),
        )
        relations_path = DATASET_DIR[dataset_name] + "/relations/"
        if not os.path.isdir(relations_path):
            os.makedirs(relations_path)

        print("Inserting relations inside buckets...\n")
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + '/kg_final.txt', 'r') as file:
            csv_reader = csv.reader(file, delimiter=' ')
            invalid = 0
            for row in csv_reader:
                row[0] = int(row[0])
                if row[0] not in mappings[product]:
                    invalid += 1
                    continue
                head = mappings[product][row[0]][0] #id of the movie in the kg
                relation = int(row[1])
                tail = int(row[2])

                if relation not in SELECTED_RELATIONS[dataset_name]: continue
                tail_entity_name = get_tail_entity_name(dataset_name, relation)
                relation_name = LASTFM_RELATION_NAME[relation]
                if tail not in mappings[tail_entity_name]: continue
                kg_id_tail = mappings[tail_entity_name][tail]
                movie_id_entity[relation_name][0][head].append(kg_id_tail)
        file.close()
        #print(invalid)
        for relation_name in movie_id_entity.keys():
            relationship_filename = movie_id_entity[relation_name][1]
            associated_entity_list = movie_id_entity[relation_name][0]
            print("Populating " + relationship_filename + "...\n")
            with open(relationship_filename, 'w+') as file:
                for entitylist_for_movie in associated_entity_list:
                    s = ' '.join([str(entitity) for entitity in entitylist_for_movie])
                    file.writelines(s)
                    file.write("\n")
            zip_file(relationship_filename)

    #Generate mappings from uid to sensible attributes for gender, age and occupation
    def generate_user_attributes_mappings(self):
        dataset_name = self.args.dataset
        uid_attributes = {}
        #user_id, country, age, gender, playcount, registered_unixtime
        with open(DATASET_DIR[dataset_name] + "/users.dat", 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                uid = row[0]
                country = row[1]
                age = row[2]
                gender = row[3]
                uid_attributes[uid] = [country, age, gender]
        file.close()

        if not os.path.exists(DATASET_DIR[dataset_name] + "/mappings/"):
            os.makedirs(DATASET_DIR[dataset_name] + "/mappings/")

        # Write user_occupation mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2country.txt", 'w+') as file:
            for uid, attributes in uid_attributes.items():
                country = attributes[0]
                file.write(uid + "\t" + country + "\n")
        file.close()

        # Write user_age mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2age_map.txt", 'w+') as file:
            for uid, attributes in uid_attributes.items():
                age = attributes[1]
                file.write(uid + "\t" + age + "\n")
        file.close()

        #Write user_gender mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2gender.txt", 'w+') as file:
            for uid, attributes in uid_attributes.items():
                gender = attributes[2]
                file.write(uid + "\t" + gender + "\n")
        file.close()

    def generate_kg_entities(self):
        dataset_name = self.args.dataset
        #Creates a dict of sets to store all the extracted entitities for every differnt type
        kg_entities = edict(
            user=(set(), 'user.txt'),
            song=(set(), 'song.txt'),
            artist=(set(), 'artist.txt'),
            engineer=(set(), 'engineer.txt'),
            producer=(set(), 'producer.txt'),
            category=(set(), 'category.txt'),
            related_song=(set(), 'related_song.txt'),
        )
        entity_path = DATASET_DIR[dataset_name] + "/entities/"
        if not os.path.isdir(entity_path):
            os.makedirs(entity_path)

        lastid2name = {}
        with open(DATASET_DIR[dataset_name] + "/tracks.txt") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader, None)
            for row in reader:
                track_id = int(row[0])
                lastid2name[track_id] = row[1]
        file.close()

        file = open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/item_list.txt", "r")
        csv_reader = csv.reader(file, delimiter=' ')
        dbid2lastid = {}
        lastid2dbid = {}
        lastid2freebase = {}
        next(csv_reader, None)
        for row in csv_reader:
            last_id = int(row[0])
            if last_id not in lastid2name: continue
            dbid2lastid[int(row[1])] = int(row[0])
            lastid2dbid[int(row[0])] = int(row[1])
            lastid2freebase[int(row[0])] = row[2]
        file.close()

        kgid2freebase = {}
        file = open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/entity_list.txt", "r")
        csv_reader = csv.reader(file, delimiter=' ')
        next(csv_reader, None)
        for row in csv_reader:
            kgid2freebase[int(row[1])] = row[0]
        file.close()

        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg_final.txt", 'r') as file:
            csv_reader = csv.reader(file, delimiter=' ')
            invalid = 0
            count = 0
            for row in csv_reader:
                head = int(row[0])
                relation = int(row[1])
                tail = int(row[2])
                if head not in dbid2lastid:
                    invalid += 1
                    continue

                movie_id = head #OCCHIO
                tail_name = get_tail_entity_name(dataset_name, relation) #Retriving what is the tail of that relation
                kg_entities['song'][0].add(movie_id)
                kg_entities[tail_name][0].add(tail)
        file.close()
        #print(invalid, count)

        review_uid_kg_uid = {}
        # Write user entity
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/user_list.txt", 'r') as file:
            csv_reader = csv.reader(file, delimiter=' ')
            next(csv_reader, None)
            with open(entity_path + "/user.txt", 'w+') as file:
                for row in csv_reader:
                    review_uid = int(row[0])
                    kg_uid = int(row[1])
                    if review_uid in kg_entities.user[0]: continue
                    kg_entities.user[0].add(review_uid)
                    review_uid_kg_uid[review_uid] = kg_uid
                    file.writelines(str(kg_uid))
                    file.write("\n")
        file.close()
        zip_file(entity_path + "user.txt")

        with open(DATASET_DIR[dataset_name] + "/mappings/review_uid_kg_uid_mapping.txt", 'w+') as file:
            header = ["kgid", "lastfmid"]
            file.write(' '.join(header) + "\n")
            for review_id, kg_id in review_uid_kg_uid.items():
                file.write('\t'.join([str(review_id), str(kg_id), "\n"]))
        file.close()

        #Populate movie entity file (Done by itself due to is different structure)
        new_id2old_id = {}
        with open(entity_path + "/song.txt", 'w+') as file:
            for idx, movie in enumerate(kg_entities['song'][0]):
                new_id2old_id[idx] = int(movie)
                file.write(str(idx) + "\n")
        file.close()
        zip_file(entity_path + "song.txt")

        # newId (0...n), oldId(movilandID), entityId(jointkgentityid), trackname, freebase id
        with open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", 'w+') as file:
            header = ["kgid", "lastfmid", "kgcompletionid", "trackname", "freebaseid"]
            file.write(' '.join(header) + "\n")
            for new_id, db_id in new_id2old_id.items():
                last_id = dbid2lastid[db_id]
                track_name = lastid2name[last_id]
                freebase_id = lastid2freebase[last_id]
                file.write('\t'.join([str(new_id), str(last_id), str(db_id), track_name, freebase_id, "\n"]))
        file.close()

        #Populating other entities
        for entity_name in get_entities_without_user(dataset_name):
            if entity_name == 'song': continue
            new_id2old_id = {}
            filename = entity_path + entity_name + '.txt'
            #Populate entities
            with open(filename, 'w+') as file:
                for idx, entity in enumerate(kg_entities[entity_name][0]):
                    new_id2old_id[idx] = int(entity)
                    file.write(str(idx) + "\n")
            file.close()

            # newId (0...n), entityId(jointkgentityid), entityNameDBPEDIA
            with open(DATASET_DIR[dataset_name] + "/mappings/" + entity_name + 'id2dbid.txt', 'w+') as file:
                header = ["kg_id", "kg_completion_id", "freebase_id"]
                file.write(' '.join(header) + "\n")
                for new_id, old_id in new_id2old_id.items():
                    entity_dblink = kgid2freebase[old_id]
                    file.write(str(new_id) + '\t' + str(old_id) + '\t' + entity_dblink + "\n")
            file.close()

            # Zip entities
            zip_file(filename)



#Generate the mapping from the KG Completation of Joint-KG to a PGPR readable dataset
class ML1MDatasetMapper(object):
    def __init__(self, args):
        self.args = args
        self.generate_train_test_split()
        self.generate_dbid_mlid_mapping()
        self.generate_kg_entities()
        self.generate_kg_relations()
        self.generate_user_attributes_mappings()

    def generate_train_test_split(self):
        dataset_name = self.args.dataset
        uid_review_tuples = {}
        dataset_size = 0
        print("Loading reviews...")
        with open(DATASET_DIR[dataset_name] + "/ratings.dat", 'r', encoding='latin-1') as reviews_file:
            reader = csv.reader(reviews_file, delimiter='\n')
            for row in reader:
                row = ''.join(row).strip().split("::")
                if row[0] not in uid_review_tuples:
                    uid_review_tuples[row[0]] = []
                uid_review_tuples[row[0]].append((row[0], row[1], row[2], row[3]))
                dataset_size += 1
        reviews_file.close()
        train_size = 0.8
        print("Performing split {}/{}...".format(train_size*100, 100-train_size*100))
        for uid, reviews in uid_review_tuples.items():
            reviews.sort(key=lambda x: int(x[3])) #sorting from recent to older

        train = []
        test = []
        for uid, reviews in uid_review_tuples.items():  # python dict are sorted, 1...nuser
            n_elements_test = int(len(reviews) * train_size)
            train.append(reviews[:n_elements_test])
            test.append(reviews[n_elements_test:])

        print("Writing train...")
        with open(DATASET_DIR[dataset_name] + "/train.txt", 'w+') as file:
            for user_reviews in train:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()

        print("Writing test...")
        with open(DATASET_DIR[dataset_name] + "/test.txt", 'w+') as file:
            for user_reviews in test:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()
        print("Zipping train and test...")
        zip_file(DATASET_DIR[dataset_name] + "/train.txt")
        zip_file(DATASET_DIR[dataset_name] + "/test.txt")
        print("Loading reviews.. DONE")

    #Generate mappings from uid to sensible attributes for gender, age and occupation
    def generate_user_attributes_mappings(self):
        dataset_name = self.args.dataset
        users_id = []
        genders = []
        ages = []
        occupations = []
        with open(DATASET_DIR[dataset_name] + "/users.dat", 'r') as file:
            csv_reader = csv.reader(file, delimiter='\n')
            for row in csv_reader:
                attributes = row[0].strip().split('::')
                users_id.append(attributes[0])
                genders.append(attributes[1])
                ages.append(attributes[2])
                occupations.append(attributes[3])
        file.close()

        #Write user_gender mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2gender.txt", 'w+') as file:
            for user, gender in zip(users_id, genders):
                file.write(user + "\t" + gender + "\n")
        file.close()

        # Write user_occupation mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2occupation.txt", 'w+') as file:
            for user, occupation in zip(users_id, occupations):
                file.write(user + "\t" + occupation + "\n")
        file.close()

        # Write user_age mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2age_map.txt", 'w+') as file:
            for user, age in zip(users_id, ages):
                file.write(user + "\t" + age + "\n")
        file.close()

    def generate_kg_entities(self):
        dataset_name = self.args.dataset
        #Creates a dict of sets to store all the extracted entitities for every differnt type
        kg_entities = edict(
            user=(set(), 'user.txt'),
            movie=(set(), 'movie.txt'),
            actor=(set(), 'actor.txt'),
            director=(set(), 'director.txt'),
            producer=(set(), 'producer.txt'),
            production_company=(set(), 'production_company.txt'),
            category=(set(), 'category.txt'),
            editor=(set(), 'editor.txt'),
            writter=(set(), 'writter.txt'),
            cinematographer=(set(), 'cinematographer.txt'),
        )
        entity_path = DATASET_DIR[dataset_name] + "/entities/"
        if not os.path.isdir(entity_path):
            os.makedirs(entity_path)

        file = open(DATASET_DIR[dataset_name] + "/mappings/movie_dbid2mlid.txt", "r")
        csv_reader = csv.reader(file, delimiter='\n')
        dbid2mlid = {}
        mlid2dbid = {}
        for i, row in enumerate(csv_reader):
            row = row[0].strip().split("\t")
            dbid2mlid[int(row[1])] = int(row[0])
            mlid2dbid[int(row[0])] = int(row[1])
        file.close()

        file = open(DATASET_DIR[dataset_name] + "/mappings/MappingMovielens2DBpedia-1.2.tsv", 'r')
        csv_reader = csv.reader(file, delimiter='\n')
        mlid2entityname = {}
        for row in csv_reader:
            row = row[0].strip().split("\t")
            mlid2entityname[int(row[0])] = row[1]
        file.close()

        file = open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg/e_map.dat", 'r')
        csv_reader = csv.reader(file, delimiter='\t')
        mlid2entityname = {}
        for row in csv_reader:
            mlid2entityname[int(row[0])] = row[1]
        file.close()

        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/dataset.dat", 'r') as file:
            csv_reader = csv.reader(file, delimiter='\t')
            for row in csv_reader:
                head = int(row[0])
                tail = row[1]
                relation = int(row[2])
                if head not in dbid2mlid: continue

                movie_id = dbid2mlid[head]
                tail_name = get_tail_entity_name(dataset_name, relation) #Retriving what is the tail of that relation

                kg_entities['movie'][0].add(movie_id)
                kg_entities[tail_name][0].add(tail)
        file.close()

        # Write user entity
        with open(DATASET_DIR[dataset_name] + "/users.dat", 'r') as file:
            csv_reader = csv.reader(file, delimiter='\n')
            for row in csv_reader:
                row = row[0].strip().split('::')
                uid = int(row[0])
                kg_entities.user[0].add(uid)

        with open(entity_path + "/user.txt", 'w+') as file:
            for u in kg_entities.user[0]:
                file.writelines(str(u))
                file.write("\n")
        file.close()

        zip_file(entity_path + "/user.txt")

        #Populate movie entity file (Done by itself due to is different structure)
        new_id2old_id = {}
        with open(entity_path + "/movie.txt", 'w+') as file:
            for idx, movie in enumerate(kg_entities['movie'][0]):
                new_id2old_id[idx] = int(movie)
                file.write(str(idx) + "\n")
        file.close()

        # newId (0...n), oldId(movilandID), entityId(jointkgentityid), entityNameDBPEDIA
        with open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", 'w+') as file:
            header = ["kg_id", "ml1m_id", "kg_completion_id", "dbpedia_url"]
            file.write('\t'.join(header))
            for new_id, old_id in new_id2old_id.items():
                entity_id = mlid2dbid[old_id]
                movie_dblink = mlid2entityname[old_id]
                file.write(str(new_id) + '\t' + str(old_id) + '\t' + str(entity_id) + '\t' + movie_dblink + "\n")
        file.close()

        zip_file(entity_path + "/movie.txt")

        #Populating other entities
        for entity_name in get_entities_without_user(dataset_name):
            if entity_name == 'movie': continue
            new_id2old_id = {}
            filename = entity_path + entity_name + '.txt'
            #Populate entities
            with open(filename, 'w+') as file:
                for idx, entity in enumerate(kg_entities[entity_name][0]):
                    new_id2old_id[idx] = int(entity)
                    file.write(str(idx) + "\n")
            file.close()

            # newId (0...n), entityId(jointkgentityid), entityNameDBPEDIA
            with open(DATASET_DIR[dataset_name] + "/mappings/" + entity_name + 'id2dbid.txt', 'w+') as file:
                for new_id, old_id in new_id2old_id.items():
                    entity_dblink = mlid2entityname[old_id]
                    file.write(str(new_id) + '\t' + str(old_id) + '\t' + entity_dblink + "\n")
            file.close()

            # Zip entities
            zip_file(filename)

    def generate_kg_relations(self):
        dataset_name = args.dataset
        mappings = get_all_entity_mappings(dataset_name)

        no_of_movies = len(mappings['movie'])+1
        movie_id_entity = edict(
            production_company=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/produced_by_company_m_pc.txt'),
            category=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/belong_to_p_ca.txt'),
            director=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/directed_by_m_d.txt'),
            actor=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/starring_m_a.txt'),
            cinematographer=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/cinematography_m_ci.txt'),
            editor=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/edited_by_m_ed.txt'),
            producer=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/produced_by_producer_m_pr.txt'),
            writter=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/wrote_by_m_w.txt'),
        )
        relations_path = DATASET_DIR[dataset_name] + "/relations/"
        if not os.path.isdir(relations_path):
            os.makedirs(relations_path)

        print("Inserting relations inside buckets...\n")
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + '/dataset.dat', 'r') as file:
            csv_reader = csv.reader(file, delimiter='\n')
            for row in csv_reader:
                row = row[0].strip().split("\t")
                row[0] = int(row[0])
                if row[0] not in mappings['movie']: continue
                head = mappings['movie'][row[0]][0] #id of the movie in the kg
                tail = int(row[1])
                relation = int(row[2])

                if relation not in SELECTED_RELATIONS[dataset_name]: continue
                tail_entity_name = get_tail_entity_name(dataset_name, relation)
                if tail not in mappings[tail_entity_name]: continue
                kg_id_tail = mappings[tail_entity_name][tail]
                movie_id_entity[tail_entity_name][0][head].append(kg_id_tail)
        file.close()

        for entitity_name in get_entities_without_user(dataset_name):
            if entitity_name == 'movie': continue
            relationship_filename = movie_id_entity[entitity_name][1]
            associated_entity_list = movie_id_entity[entitity_name][0]
            print("Populating " + relationship_filename + "...\n")
            with open(relationship_filename, 'w+') as file:
                for entitylist_for_movie in associated_entity_list:
                    s = ' '.join([str(entitity) for entitity in entitylist_for_movie])
                    file.writelines(s)
                    file.write("\n")
            zip_file(relationship_filename)

    def generate_dbid_mlid_mapping(self):
        dataset_name = args.dataset
        dblink_id = {}
        dbid_mlid = {}
        tsv_file = open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/i2kg_map.tsv")
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for row in read_tsv:
            #print(row)
            dblink_id[row[2]] = [row[0], row[1]]
        tsv_file.close()

        count = 0
        hit = 0
        dat_file = open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg/e_map.dat")
        read_dat = csv.reader(dat_file, delimiter="\t")
        for row in read_dat:
            if row[1] not in dblink_id:
                #print("Not a movie #" + str(count))
                count = count+1
            else:
                #print("Movie #" + str(hit))
                hit = hit + 1
                dbid_mlid[row[0]] = int(dblink_id[row[1]][0])
        dat_file.close()
        dbid_mlid_sorted = {k: v for k, v in sorted(dbid_mlid.items(), key=lambda item: item[1])}
        #K is the id in movielands v is the id in entities
        with open(DATASET_DIR[dataset_name] + "/mappings/movie_dbid2mlid.txt", 'w+') as file:
            for k,v in dbid_mlid_sorted.items():
                file.writelines(str(v) + "\t" + k)
                file.write("\n")
        file.close()



def unify_dataset(args):
    dataset_name = args.dataset
    selected_relationship = SELECTED_RELATIONS[dataset_name]
    print("Unifying dataset from joint-kg Knowledge graph completation for {}...".format(dataset_name))
    with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/dataset.dat", 'w+', newline='\n') as dataset_file:
        print("Loading joint-kg train...")
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg/train.dat") as joint_kg_train:
            csv_reader = csv.reader(joint_kg_train, delimiter='\t')
            for row in csv_reader:
                relation = int(row[2])
                if relation not in selected_relationship: continue
                dataset_file.writelines('\t'.join(row))
                dataset_file.write("\n")
        joint_kg_train.close()
        print("Loading joint-kg valid...")
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg/valid.dat") as joint_kg_valid:
            csv_reader = csv.reader(joint_kg_valid, delimiter='\t')
            for row in csv_reader:
                relation = int(row[2])
                if relation not in selected_relationship: continue
                dataset_file.writelines('\t'.join(row))
                dataset_file.write("\n")
        joint_kg_valid.close()
        print("Loading joint-kg test...")
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg/test.dat") as joint_kg_test:
            csv_reader = csv.reader(joint_kg_test, delimiter='\t')
            for row in csv_reader:
                relation = int(row[2])
                if relation not in selected_relationship: continue
                dataset_file.writelines('\t'.join(row))
                dataset_file.write("\n")
        joint_kg_test.close()
        print("Unifying dataset from joint-kg Knowledge graph completation... DONE")
    dataset_file.close()

if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {ML1M, LASTFM}')
    args = parser.parse_args()


    if args.dataset == ML1M:
        ML1MDatasetMapper(args)
    elif args.dataset == LASTFM:
        LastFmDatasetMapper(args)
    else:
        print("Invalid dataset string, chose one between [ml1m, lastfm]")
