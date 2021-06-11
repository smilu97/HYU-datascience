#!/usr/bin/env python

# author: smilu97
# created at: 21-06-11 17:44

import tensorflow as tf
import numpy as np
import pandas as pd
import argparse

from svdtppatt import SVDtppAttRecommender

parser = argparse.ArgumentParser('recommender')
parser.add_argument('base', type=str, help='base filepath')
parser.add_argument('test', type=str, help='test filepath')
parser.add_argument('--epochs', '-e', type=int, default=0, help='epochs')
parser.add_argument('--batch-size', '-b', type=int, default=2000, help='batch size')
parser.add_argument('--dim', '-d', type=int, default=24, help='The size of hidden factor')
parser.add_argument('--reg-lambda', type=float, default=0.0000, help='regularization factor')
parser.add_argument('--lr', '-r', type=float, default=0.001, help='learning rate')
parser.add_argument('--sz-related', type=int, default=30, help='the maximum number of related objects')
parser.add_argument('--n-time-bins', type=int, default=30, help='the number of item time bins')
parser.add_argument('--augment', action='store_true', help='run data augmentation')
parser.add_argument('--seed', '-s', default=123, type=int, help='random seed')

def main():
    args = parser.parse_args()

    train_df = pd.read_csv(args.base, sep='\t', header=None)
    test_df  = pd.read_csv(args.test, sep='\t', header=None)
    observation = train_df.to_numpy()
    test  = test_df.to_numpy()

    tf.random.set_seed(args.seed)

    recommender = SVDtppAttRecommender(
        feature_dim=args.dim,
        lr=args.lr,
        reg_lambda=args.reg_lambda,
        sz_related=args.sz_related,
        n_time_bins=args.n_time_bins,
    )
    recommender.fit(observation, batch_size=args.batch_size, epochs=args.epochs, validation_data=test)
    test_y = recommender.predict(test[:, 0], test[:, 1], test[:, 3])
    test_y = np.minimum(5.0, np.maximum(0.0, test_y))

    pred = np.array(test[:,:3], dtype=np.object)
    pred[:, 2] = test_y

    pred_df = pd.DataFrame(pred)
    output_filepath = args.base + '_prediction.txt'
    pred_df.to_csv(output_filepath, sep='\t', header=None, index=False)
    print('Saved result on {}'.format(output_filepath))

    score = np.mean(np.square(test_y.flatten() - test[:, 2]))
    print('score:', score)

if __name__ == '__main__':
    main()
