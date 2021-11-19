from __future__ import absolute_import, division, print_function

import os
import numpy as np
import gzip
import pickle
from easydict import EasyDict as edict
import random
import collections
from models.PGPR.utils import get_movie_relationships, DATASET_DIR, \
    get_product_id_kgid_mapping, get_song_relationships, get_uid_to_kgid_mapping


class AmazonDataset(object):
    """This class is used to load data files and save in the instance."""

    def __init__(self, data_dir, set_name='train', word_sampling_rate=1e-4):
        self.dataset_name = data_dir.strip().split("/")[-1]
        self.data_dir = data_dir
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = set_name + '.txt.gz'
        self.load_entities()
        self.load_product_relations()
        self.load_reviews()

    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            # In Python 3, must use decode() to convert bytes to string!
            return [line.decode('utf-8').strip() for line in f]

    def load_entities(self):
        """Load 10 global entities from data files:
        'user','movie','actor','director','producer','production_company','category','editor','writter','cinematographer'
        Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """
        if self.dataset_name == "ml1m":
            entity_files = edict(
                    user='entities/user.txt.gz',
                    movie='entities/movie.txt.gz',
                    actor='entities/actor.txt.gz',
                    composer='entities/composer.txt.gz',
                    director='entities/director.txt.gz',
                    producer='entities/producer.txt.gz',
                    production_company='entities/production_company.txt.gz',
                    category='entities/category.txt.gz',
                    editor='entities/editor.txt.gz',
                    writter='entities/writter.txt.gz',
                    cinematographer='entities/cinematographer.txt.gz',
            )
        elif self.dataset_name == "lastfm":
            entity_files = edict(
                user='entities/user.txt.gz',
                song='entities/song.txt.gz',
                artist='entities/artist.txt.gz',
                engineer='entities/engineer.txt.gz',
                producer='entities/producer.txt.gz',
                category='entities/category.txt.gz',
                related_song='entities/related_song.txt.gz',
            )
        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)+1))
            print('Load', name, 'of size', len(vocab))

    def load_reviews(self):
        """Load user-product reviews from train/test data files.
        Create member variable `review` associated with following attributes:
        - `data`: list of tuples (user_idx, product_idx, [word_idx...]).
        - `size`: number of reviews.
        - `product_distrib`: product vocab frequency among all eviews.
        - `product_uniform_distrib`: product vocab frequency (all 1's)
        - `word_distrib`: word vocab frequency among all reviews.
        - `review_count`: number of words (including duplicates).
        - `review_distrib`: always 1.
        """
        review_data = []  # (user_idx, product_idx, rating out of 5, timestamp)
        product_distrib = np.zeros(self.movie.vocab_size) if self.dataset_name == "ml1m" else np.zeros(self.song.vocab_size)
        positive_reviews = 0
        negative_reviews = 0
        threshold = 3
        id2kgid = get_product_id_kgid_mapping(self.dataset_name)
        uid2kg_uid = get_uid_to_kgid_mapping(self.dataset_name)
        for line in self._load_file(self.review_file):
            arr = line.split(' ')
            user_idx = uid2kg_uid[int(arr[0])]
            if int(arr[1]) not in id2kgid: continue
            product_idx = id2kgid[int(arr[1])]
            rating = int(arr[2]) if self.dataset_name == "ml1m" else 0
            timestamp = int(arr[3]) if self.dataset_name == "ml1m" else int(arr[2])
            if rating >= threshold:
                positive_reviews+=1
            else:
                negative_reviews+=1
            review_data.append((user_idx, product_idx, rating, timestamp))
            product_distrib[product_idx] += 1

        self.review = edict(
                data=review_data,
                size=len(review_data),
                product_distrib=product_distrib,
                product_uniform_distrib=np.ones(self.movie.vocab_size if self.dataset_name == "ml1m" else self.song.vocab_size),
                review_count=len(review_data),
                review_distrib=np.ones(len(review_data)) #set to 1 now
        )
        print('Load review of size', self.review.size, 'with positive reviews=',
              positive_reviews, ' and negative reviews=',
              negative_reviews)#, ' considered as positive the ratings >= of ', threshold)

    def load_product_relations(self):
        """Load 8 product -> ? relations:
        - 'directed_by': movie -> director
        - 'produced_by_company': movie->production_company,
        - 'produced_by_producer': movie->producer,
        - 'starring': movie->actor,
        - 'belong_to': movie->category,
        - 'edited_by': movie->editor,
        - 'written_by': movie->writter,
        - 'cinematography': movie->cinematographer,

        Create member variable for each relation associated with following attributes:
        - `data`: list of list of entity_tail indices (can be empty).
        - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
        - `et_distrib`: frequency of entity_tail vocab.
        """
        if self.dataset_name == "ml1m":
            dataset_dir = DATASET_DIR[self.dataset_name]
            product_relations = edict(
                    directed_by=('relations/directed_by_m_d.txt.gz', self.director),
                    composed_by=('relations/composed_by_m_c.txt.gz', self.composer),
                    produced_by_company=('relations/produced_by_company_m_pc.txt.gz', self.production_company),
                    produced_by_producer=('relations/produced_by_producer_m_pr.txt.gz', self.producer),
                    starring=('relations/starring_m_a.txt.gz', self.actor),
                    belong_to=('relations/belong_to_m_ca.txt.gz', self.category),
                    edited_by=('relations/edited_by_m_ed.txt.gz', self.editor),
                    wrote_by=('relations/wrote_by_m_w.txt.gz', self.writter),
                    cinematography=('relations/cinematography_m_ci.txt.gz', self.cinematographer),
            )
        elif self.dataset_name == "lastfm":
            dataset_dir = DATASET_DIR[self.dataset_name]
            product_relations = edict(
                mixed_by=("/relations/mixed_by_s_e.txt.gz", self.engineer),
                featured_by=("/relations/featured_by_s_a.txt.gz",self.artist),
                sang_by=('/relations/sang_by_s_a.txt.gz', self.artist),
                related_to=('/relations/related_to_s_rs.txt.gz', self.related_song),
                alternative_version_of=("relations/alternative_version_of_s_rs.txt.gz", self.related_song),
                original_version_of=("relations/orginal_version_of_s_rs.txt.gz", self.related_song),
                belong_to=('relations/belong_to_s_ca.txt.gz', self.category),
                produced_by_producer=('relations/produced_by_producer_s_pr.txt.gz', self.producer),
            )

        for name in product_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            relation = edict(
                    data=[],
                    et_vocab=product_relations[name][1].vocab, #copy of brand, catgory ... 's vocab 
                    et_distrib= np.zeros(product_relations[name][1].vocab_size) #[1] means self.brand ..
            )
            size = 0
            for line in self._load_file(product_relations[name][0]): #[0] means brand_p_b.txt.gz ..
                knowledge = []
                for x in line.split(' '):  # some lines may be empty
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                        size += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print('Load', name, 'of size', size)

    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.vocab_size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.review.word_distrib) * sampling_threshold
        for i in range(self.word.vocab_size):
            if self.review.word_distrib[i] == 0:
                continue
            self.word_sampling_rate[i] = min((np.sqrt(float(self.review.word_distrib[i]) / threshold) + 1) * threshold / float(self.review.word_distrib[i]), 1.0)


class AmazonDataLoader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.review_size = self.dataset.review.size
        self.product_relations = get_movie_relationships() if dataset.dataset_name == "ml1m" else get_song_relationships()
        self.finished_review_num = 0
        self.reset()

    def reset(self):
        # Shuffle reviews order
        self.review_seq = np.random.permutation(self.review_size)
        self.cur_review_i = 0
        self.cur_word_i = 0
        self._has_next = True

    def get_batch(self):
        """Return a matrix of [batch_size x 8], where each row contains
        (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        batch = []
        review_idx = self.review_seq[self.cur_review_i]
        user_idx, product_idx, rating, _ = self.dataset.review.data[review_idx]
        product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations}

        while len(batch) < self.batch_size:
            data = [user_idx, product_idx]
            for pr in self.product_relations:
                if len(product_knowledge[pr]) <= 0:
                    data.append(-1)
                else:
                    data.append(random.choice(product_knowledge[pr]))
            batch.append(data)

            self.cur_review_i += 1
            self.finished_review_num += 1
            if self.cur_review_i >= self.review_size:
                self._has_next = False
                break
            review_idx = self.review_seq[self.cur_review_i]
            user_idx, product_idx, rating, _ = self.dataset.review.data[review_idx]
            product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations}
            '''
            # 1) Sample the word
            word_idx = rating_list[self.cur_word_i]
            if random.random() < self.dataset.word_sampling_rate[word_idx]:
                data = [user_idx, product_idx, word_idx]
                for pr in self.product_relations:
                    if len(product_knowledge[pr]) <= 0:
                        data.append(-1)
                    else:
                        data.append(random.choice(product_knowledge[pr]))
                batch.append(data)

            # 2) Move to next word/review
            self.cur_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i >= len(text_list):
                self.cur_review_i += 1
                if self.cur_review_i >= self.review_size:
                    self._has_next = False
                    break
                self.cur_word_i = 0
                review_idx = self.review_seq[self.cur_review_i]
                user_idx, product_idx, text_list = self.dataset.review.data[review_idx]
                product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations}
            '''
        return np.array(batch)

    def has_next(self):
        """Has next batch."""
        return self._has_next

