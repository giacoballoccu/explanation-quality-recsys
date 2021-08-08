from __future__ import absolute_import, division, print_function

from easydict import EasyDict as edict
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from data_utils import AmazonDataset


class KnowledgeEmbedding(nn.Module):
    def __init__(self, dataset, args):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda

        self.dataset_name = args.dataset
        # Initialize entity embeddings.
        if self.dataset_name == "ml1m":
            self.entities = edict(
                user=edict(vocab_size=dataset.user.vocab_size),
                movie=edict(vocab_size=dataset.movie.vocab_size),
                actor=edict(vocab_size=dataset.actor.vocab_size),
                director=edict(vocab_size=dataset.director.vocab_size),
                production_company=edict(vocab_size=dataset.production_company.vocab_size),
                producer=edict(vocab_size=dataset.producer.vocab_size),
                writter=edict(vocab_size=dataset.writter.vocab_size),
                editor=edict(vocab_size=dataset.editor.vocab_size),
                cinematographer=edict(vocab_size=dataset.cinematographer.vocab_size),
                category=edict(vocab_size=dataset.category.vocab_size),
            )
        elif self.dataset_name == "lastfm":
            self.entities = edict(
                user=edict(vocab_size=dataset.user.vocab_size),
                song=edict(vocab_size=dataset.song.vocab_size),
                artist=edict(vocab_size=dataset.artist.vocab_size),
                engineer=edict(vocab_size=dataset.engineer.vocab_size),
                related_song=edict(vocab_size=dataset.related_song.vocab_size),
                producer=edict(vocab_size=dataset.producer.vocab_size),
                category=edict(vocab_size=dataset.category.vocab_size),
            )
        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)

        # Initialize relation embeddings and relation biases.
        if self.dataset_name == "ml1m":
            self.relations = edict(
                watched=edict(
                    et='movie',
                    et_distrib=self._make_distrib(dataset.review.product_uniform_distrib)),
                produced_by_company=edict(
                    et='production_company',
                    et_distrib=self._make_distrib(dataset.produced_by_company.et_distrib)),
                produced_by_producer=edict(
                    et='producer',
                    et_distrib=self._make_distrib(dataset.produced_by_producer.et_distrib)),
                belong_to=edict(
                    et='category',
                    et_distrib=self._make_distrib(dataset.belong_to.et_distrib)),
                directed_by=edict(
                    et='director',
                    et_distrib=self._make_distrib(dataset.directed_by.et_distrib)),
                starring=edict(
                    et='actor',
                    et_distrib=self._make_distrib(dataset.starring.et_distrib)),
                wrote_by=edict(
                    et='writter',
                    et_distrib=self._make_distrib(dataset.wrote_by.et_distrib)),
                edited_by=edict(
                    et='editor',
                    et_distrib=self._make_distrib(dataset.edited_by.et_distrib)),
                cinematography=edict(
                    et='cinematographer',
                    et_distrib=self._make_distrib(dataset.cinematography.et_distrib)),
            )
        elif self.dataset_name == "lastfm":
            self.relations = edict(
                listened=edict(
                    et='song',
                    et_distrib=self._make_distrib(dataset.review.product_uniform_distrib)),
                sang_by=edict(
                    et='artist',
                    et_distrib=self._make_distrib(dataset.sang_by.et_distrib)),
                produced_by_producer=edict(
                    et='producer',
                    et_distrib=self._make_distrib(dataset.produced_by_producer.et_distrib)),
                belong_to=edict(
                    et='category',
                    et_distrib=self._make_distrib(dataset.belong_to.et_distrib)),
                featured_by=edict(
                    et='artist',
                    et_distrib=self._make_distrib(dataset.featured_by.et_distrib)),
                mixed_by=edict(
                    et='engineer',
                    et_distrib=self._make_distrib(dataset.mixed_by.et_distrib)),
                related_to=edict(
                    et='related_song',
                    et_distrib=self._make_distrib(dataset.related_to.et_distrib)),
                alternative_version_of=edict(
                    et='related_song',
                    et_distrib=self._make_distrib(dataset.alternative_version_of.et_distrib)),
                original_version_of=edict(
                    et='related_song',
                    et_distrib=self._make_distrib(dataset.original_version_of.et_distrib)),
            )

        for r in self.relations:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)

    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=np.float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        return loss

    def compute_loss(self, batch_idxs):
        """Compute knowledge graph negative sampling loss.
        batch_idxs: batch_size * 8 array, where each row is
                (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        regularizations = []
        if self.dataset_name == "ml1m":
            user_idxs = batch_idxs[:, 0]
            movie_idxs = batch_idxs[:, 1]
            production_company_idxs = batch_idxs[:, 2]
            producer_idxs = batch_idxs[:, 3]
            editor_idxs = batch_idxs[:, 4]
            writter_idxs = batch_idxs[:, 5]
            cinematographer_idxs = batch_idxs[:, 6]
            category_idxs = batch_idxs[:, 7]
            director_idxs = batch_idxs[:, 8]
            actor_idxs = batch_idxs[:, 9]

            # user + watched -> movie
            uw_loss, uw_embeds = self.neg_loss('user', 'watched', 'movie', user_idxs, movie_idxs)
            regularizations.extend(uw_embeds)
            loss = uw_loss

            # movie + produced_by_company -> production_company
            mpc_loss, mpc_embeds = self.neg_loss('movie', 'produced_by_company', 'production_company', movie_idxs,
                                                 production_company_idxs)
            if mpc_loss is not None:
                regularizations.extend(mpc_embeds)
                loss += mpc_loss

            # movie + produced_by_producer -> producer
            mpr_loss, mpr_embeds = self.neg_loss('movie', 'produced_by_producer', 'producer', movie_idxs,
                                                 producer_idxs)
            if mpr_loss is not None:
                regularizations.extend(mpr_embeds)
                loss += mpc_loss

            # product + belong_to -> category
            pc_loss, pc_embeds = self.neg_loss('movie', 'belong_to', 'category', movie_idxs, category_idxs)
            if pc_loss is not None:
                regularizations.extend(pc_embeds)
                loss += pc_loss

            # movie + starring -> actor
            pr2_loss, pr2_embeds = self.neg_loss('movie', 'starring', 'actor', movie_idxs, actor_idxs)
            if pr2_loss is not None:
                regularizations.extend(pr2_embeds)
                loss += pr2_loss

            # movie + directed_by -> director
            pr3_loss, pr3_embeds = self.neg_loss('movie', 'directed_by', 'director', movie_idxs, director_idxs)
            if pr3_loss is not None:
                regularizations.extend(pr3_embeds)
                loss += pr3_loss

            # movie + wrote_by -> writter
            mw_loss, mw_embeds = self.neg_loss('movie', 'wrote_by', 'writter', movie_idxs, writter_idxs)
            if mw_loss is not None:
                regularizations.extend(mw_embeds)
                loss += mw_loss

            # movie + edited_by -> editor
            med_loss, med_embeds = self.neg_loss('movie', 'edited_by', 'editor', movie_idxs, editor_idxs)
            if med_loss is not None:
                regularizations.extend(med_embeds)
                loss += med_loss

            # movie + cinematography -> cinematographer
            mc_loss, mc_embeds = self.neg_loss('movie', 'cinematography', 'cinematographer', movie_idxs,
                                               cinematographer_idxs)
            if mc_loss is not None:
                regularizations.extend(mc_embeds)
                loss += mc_loss

        elif self.dataset_name == "lastfm":
            user_idxs = batch_idxs[:, 0]
            song_idxs = batch_idxs[:, 1]
            producer_idxs = batch_idxs[:, 2]
            artist1_idxs = batch_idxs[:, 3]
            artist2_idxs = batch_idxs[:, 4]
            engineer_idxs = batch_idxs[:, 5]
            category_idxs = batch_idxs[:, 6]
            related_song1_idxs = batch_idxs[:, 7]
            related_song2_idxs = batch_idxs[:, 8]
            related_song3_idxs = batch_idxs[:, 9]

            # user + listened -> song
            ul_loss, ul_embeds = self.neg_loss('user', 'listened', 'song', user_idxs, song_idxs)
            regularizations.extend(ul_embeds)
            loss = ul_loss

            # movie + produced_by_producer -> producer
            spr_loss, spr_embeds = self.neg_loss('song', 'produced_by_producer', 'producer', song_idxs,
                                                 producer_idxs)
            if spr_loss is not None:
                regularizations.extend(spr_embeds)
                loss += spr_loss

            # song + sang_by -> artist
            sar1_loss, mpc1_embeds = self.neg_loss('song', 'sang_by', 'artist', song_idxs,
                                                 artist1_idxs)
            if sar1_loss is not None:
                regularizations.extend(mpc1_embeds)
                loss += sar1_loss

            # song + featured_by -> artist
            sar2_loss, mpc2_embeds = self.neg_loss('song', 'featured_by', 'artist', song_idxs,
                                                  artist2_idxs)
            if sar2_loss is not None:
                regularizations.extend(mpc2_embeds)
                loss += sar2_loss

            # song + belong_to -> category
            sc_loss, sc_embeds = self.neg_loss('song', 'belong_to', 'category', song_idxs, category_idxs)
            if sc_loss is not None:
                regularizations.extend(sc_embeds)
                loss += sc_loss

            # song + mixed_by -> engineer
            se_loss, se_embeds = self.neg_loss('song', 'mixed_by', 'engineer', song_idxs, engineer_idxs)
            if se_loss is not None:
                regularizations.extend(se_embeds)
                loss += se_loss

            # song + related_to -> related_song
            srs1_loss, srs1_embeds = self.neg_loss('song', 'related_to', 'related_song', song_idxs, related_song1_idxs)
            if srs1_loss is not None:
                regularizations.extend(srs1_embeds)
                loss += srs1_loss

            # song + original_version_of -> related_song
            srs2_loss, srs2_embeds = self.neg_loss('song', 'original_version_of', 'related_song', song_idxs, related_song2_idxs)
            if srs2_loss is not None:
                regularizations.extend(srs2_embeds)
                loss += srs2_loss

            # song + alternative_version_of -> related_song
            srs3_loss, srs3_embeds = self.neg_loss('song', 'alternative_version_of', 'related_song', song_idxs, related_song3_idxs)
            if srs3_loss is not None:
                regularizations.extend(srs3_embeds)
                loss += srs3_loss


        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        return loss

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs):
        # Entity tail indices can be -1. Remove these indices. Batch size may be changed!
        mask = entity_tail_idxs >= 0
        fixed_entity_head_idxs = entity_head_idxs[mask]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []

        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding
        entity_tail_embedding = getattr(self, entity_tail)  # nn.Embedding
        relation_vec = getattr(self, relation)  # [1, embed_size]
        relation_bias_embedding = getattr(self, relation + '_bias')  # nn.Embedding
        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size]

        return kg_neg_loss(entity_head_embedding, entity_tail_embedding,
                           fixed_entity_head_idxs, fixed_entity_tail_idxs,
                           relation_vec, relation_bias_embedding, self.num_neg_samples, entity_tail_distrib)


def kg_neg_loss(entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                relation_vec, relation_bias_embed, num_samples, distrib):
    """Compute negative sampling loss for triple (entity_head, relation, entity_tail).

    Args:
        entity_head_embed: Tensor of size [batch_size, embed_size].
        entity_tail_embed: Tensor of size [batch_size, embed_size].
        entity_head_idxs:
        entity_tail_idxs:
        relation_vec: Parameter of size [1, embed_size].
        relation_bias: Tensor of size [batch_size]
        num_samples: An integer.
        distrib: Tensor of size [vocab_size].

    Returns:
        A tensor of [1].
    """
    batch_size = entity_head_idxs.size(0)
    entity_head_vec = entity_head_embed(entity_head_idxs)  # [batch_size, embed_size]
    example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
    example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]

    entity_tail_vec = entity_tail_embed(entity_tail_idxs)  # [batch_size, embed_size]
    pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
    relation_bias = relation_bias_embed(entity_tail_idxs).squeeze(1)  # [batch_size]
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
    pos_loss = -pos_logits.sigmoid().log()  # [batch_size]

    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

    loss = (pos_loss + neg_loss).mean()
    return loss, [entity_head_vec, entity_tail_vec, neg_vec]

