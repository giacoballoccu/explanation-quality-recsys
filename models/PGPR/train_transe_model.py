from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from data_utils import AmazonDataset, AmazonDataLoader
from transe_model import KnowledgeEmbedding


logger = None


def train(args):
    dataset = load_dataset(args.dataset)
    dataloader = AmazonDataLoader(dataset, args.batch_size)
    review_to_train = len(dataset.review.data) * args.epochs

    model = KnowledgeEmbedding(dataset, args).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    steps = 0
    smooth_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            # Set learning rate.
            lr = args.lr * max(1e-4, 1.0 - dataloader.finished_review_num / float(review_to_train))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Get training batch.
            batch_idxs = dataloader.get_batch()
            batch_idxs = torch.from_numpy(batch_idxs).to(args.device)

            # Train model.
            optimizer.zero_grad()
            train_loss = model(batch_idxs)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            smooth_loss += train_loss.item() / args.steps_per_checkpoint

            steps += 1
            if steps % args.steps_per_checkpoint == 0:
                logger.info('Epoch: {:02d} | '.format(epoch) +
                            'Review: {:d}/{:d} | '.format(dataloader.finished_review_num, review_to_train) +
                            'Lr: {:.5f} | '.format(lr) +
                            'Smooth loss: {:.5f}'.format(smooth_loss))
                smooth_loss = 0.0

        torch.save(model.state_dict(), '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, epoch))


def extract_embeddings(args):
    """Note that last entity embedding is of size [vocab_size+1, d]."""
    model_file = '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, args.epochs)
    print('Load embeddings', model_file)
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
    if args.dataset == "ml1m":
        embeds = {
            USER: state_dict['user.weight'].cpu().data.numpy()[:-1],  # Must remove last dummy 'user' with 0 embed.
            MOVIE: state_dict['movie.weight'].cpu().data.numpy()[:-1],
            ACTOR: state_dict['actor.weight'].cpu().data.numpy()[:-1],
            DIRECTOR: state_dict['director.weight'].cpu().data.numpy()[:-1],
            PRODUCTION_COMPANY: state_dict['production_company.weight'].cpu().data.numpy()[:-1],
            CATEGORY: state_dict['category.weight'].cpu().data.numpy()[:-1],
            PRODUCER: state_dict['producer.weight'].cpu().data.numpy()[:-1],
            EDITOR: state_dict['editor.weight'].cpu().data.numpy()[:-1],
            WRITTER: state_dict['writter.weight'].cpu().data.numpy()[:-1],
            CINEMATOGRAPHER: state_dict['cinematographer.weight'].cpu().data.numpy()[:-1],
            COMPOSER: state_dict['composer.weight'].cpu().data.numpy()[:-1],
            WATCHED: (
                state_dict['watched'].cpu().data.numpy()[0],
                state_dict['watched_bias.weight'].cpu().data.numpy()
            ),
            DIRECTED_BY: (
                state_dict['directed_by'].cpu().data.numpy()[0],
                state_dict['directed_by_bias.weight'].cpu().data.numpy()
            ),
            PRODUCED_BY_PRODUCER: (
                state_dict['produced_by_producer'].cpu().data.numpy()[0],
                state_dict['produced_by_producer_bias.weight'].cpu().data.numpy()
            ),
            PRODUCED_BY_COMPANY: (
                state_dict['produced_by_company'].cpu().data.numpy()[0],
                state_dict['produced_by_company_bias.weight'].cpu().data.numpy()
            ),
            STARRING: (
                state_dict['starring'].cpu().data.numpy()[0],
                state_dict['starring_bias.weight'].cpu().data.numpy()
            ),
            BELONG_TO: (
                state_dict['belong_to'].cpu().data.numpy()[0],
                state_dict['belong_to_bias.weight'].cpu().data.numpy()
            ),
            WROTE_BY: (
                state_dict['wrote_by'].cpu().data.numpy()[0],
                state_dict['wrote_by_bias.weight'].cpu().data.numpy()
            ),
            EDITED_BY: (
                state_dict['edited_by'].cpu().data.numpy()[0],
                state_dict['edited_by_bias.weight'].cpu().data.numpy()
            ),
            CINEMATOGRAPHY: (
                state_dict['cinematography'].cpu().data.numpy()[0],
                state_dict['cinematography_bias.weight'].cpu().data.numpy()
            ),
            COMPOSED_BY: (
                state_dict['composed_by'].cpu().data.numpy()[0],
                state_dict['composed_by_bias.weight'].cpu().data.numpy()
            ),
    }
    elif args.dataset == "lastfm":
        embeds = {
            USER: state_dict['user.weight'].cpu().data.numpy()[:-1],  # Must remove last dummy 'user' with 0 embed.
            SONG: state_dict['song.weight'].cpu().data.numpy()[:-1],
            ARTIST: state_dict['artist.weight'].cpu().data.numpy()[:-1],
            ENGINEER: state_dict['engineer.weight'].cpu().data.numpy()[:-1],
            RELATED_SONG: state_dict['related_song.weight'].cpu().data.numpy()[:-1],
            CATEGORY: state_dict['category.weight'].cpu().data.numpy()[:-1],
            PRODUCER: state_dict['producer.weight'].cpu().data.numpy()[:-1],
            LISTENED: (
                state_dict['listened'].cpu().data.numpy()[0],
                state_dict['listened_bias.weight'].cpu().data.numpy()
            ),
            SANG_BY: (
                state_dict['sang_by'].cpu().data.numpy()[0],
                state_dict['sang_by_bias.weight'].cpu().data.numpy()
            ),
            FEATURED_BY: (
                state_dict['featured_by'].cpu().data.numpy()[0],
                state_dict['featured_by_bias.weight'].cpu().data.numpy()
            ),
            MIXED_BY: (
                state_dict['mixed_by'].cpu().data.numpy()[0],
                state_dict['mixed_by_bias.weight'].cpu().data.numpy()
            ),
            BELONG_TO: (
                state_dict['belong_to'].cpu().data.numpy()[0],
                state_dict['belong_to_bias.weight'].cpu().data.numpy()
            ),
            PRODUCED_BY_PRODUCER: (
                state_dict['produced_by_producer'].cpu().data.numpy()[0],
                state_dict['produced_by_producer_bias.weight'].cpu().data.numpy()
            ),
            RELATED_TO: (
                state_dict['related_to'].cpu().data.numpy()[0],
                state_dict['related_to_bias.weight'].cpu().data.numpy()
            ),
            ALTERNATIVE_VERSION_OF: (
                state_dict['alternative_version_of'].cpu().data.numpy()[0],
                state_dict['alternative_version_of_bias.weight'].cpu().data.numpy()
            ),
            ORIGINAL_VERSION_OF: (
                state_dict['original_version_of'].cpu().data.numpy()[0],
                state_dict['original_version_of_bias.weight'].cpu().data.numpy()
            ),
        }
    save_embed(args.dataset, embeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {beauty, cd, cell, clothing}.')
    parser.add_argument('--name', type=str, default='train_transe_model', help='model name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam.')
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Clipping gradient.')
    parser.add_argument('--embed_size', type=int, default=300, help='knowledge embedding size.')
    parser.add_argument('--num_neg_samples', type=int, default=5, help='number of negative samples.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=200, help='Number of steps for checkpoint.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    train(args)
    extract_embeddings(args)


if __name__ == '__main__':
    main()

