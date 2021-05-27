#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import argparse

from preprocess import read_dataset
from svdpp import SVDPPModel

parser = argparse.ArgumentParser('Long term project: recommender system')
parser.add_argument('--epochs', '-e', type=int, default=40, help='epochs')
parser.add_argument('--batch-size', '-b', type=int, default=2000, help='batch size')
parser.add_argument('--dim', '-d', type=int, default=12, help='The size of hidden factor')
parser.add_argument('--validation-split', '-v', type=float, default=0.1, help='portion of validation split')
parser.add_argument('--reg-lambda', type=float, default=0.000000, help='regularization factor')
parser.add_argument('--lr', '-r', type=float, default=0.001, help='learning rate')
parser.add_argument('--output', '-o', type=str, default='output.txt', help='output filepath')
parser.add_argument('--sz-related', type=int, default=19, help='the maximum number of related objects')
parser.add_argument('--n-time-bins', type=int, default=30, help='the number of item time bins')
parser.add_argument('--augment', action='store_true', help='run data augmentation')

def split_data(data, nu, ni, sz_related=20):
    n = data.shape[0]

    users = data[:, 0].reshape((-1, 1))
    items = data[:, 1].reshape((-1, 1))
    ratings = data[:, 2]
    times = data[:, 3]

    user_related = [list() for _ in range(nu)]
    item_related = [list() for _ in range(ni)]
    user_times = [list() for _ in range(nu)]

    for idx in range(n):
        u, i, t = users[idx][0], items[idx][0], times[idx]
        user_related[u].append(i)
        item_related[i].append(u)
        user_times[u].append(t)

    ur = np.zeros((nu, sz_related))
    ir = np.zeros((ni, sz_related))
    user_time_means = np.array([sum(l)/len(l) if len(l) > 0 else 0 for l in user_times])

    for i in range(nu):
        rel = np.unique(user_related[i])
        ur[i,:len(rel)] = rel[:sz_related] + 1
    for i in range(ni):
        rel = np.unique(item_related[i])
        ir[i,:len(rel)] = rel[:sz_related] + 1
    
    min_item_times = np.ones(ni) * 10000000000
    max_item_times = np.zeros(ni)
    min_user_times = np.ones(nu) * 10000000000
    max_user_times = np.zeros(nu)

    for _ in range(n):
        u, i, t = users[i][0], items[i][0], times[i]
        min_item_times[i] = min(min_item_times[i], t)
        max_item_times[i] = max(max_item_times[i], t)
        min_user_times[u] = min(min_user_times[u], t)
        max_user_times[u] = max(max_user_times[u], t)
    
    user_time_means = (user_time_means - min_user_times) / (max_user_times - min_user_times + 1.0)

    return \
        users, \
        items, \
        ratings, \
        times, \
        ur, ir, \
        min_item_times, max_item_times, \
        min_user_times, max_user_times, \
        user_time_means

def main():
    '''
    Test model
    '''

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    validation_split = args.validation_split
    k = args.dim
    reg_lambda = args.reg_lambda
    lr = args.lr
    output = args.output
    sz_related = args.sz_related
    n_time_bins = args.n_time_bins
    augment = args.augment

    datasets = [read_dataset('./data/u{}'.format(i)) for i in range(1, 6)]

    train, test, user_mapper, item_mapper = read_dataset('./data/u1', augment=augment)
    print('shape train data:', train.shape)
    print('shape test  data:', test.shape)

    nu = len(user_mapper)
    ni = len(item_mapper)
    mu = np.mean(train[:,2])

    model = SVDPPModel(
        num_users=max(user_mapper.values()) + 1,
        num_items=max(item_mapper.values()) + 1,
        mu=mu,
        k=k,
        reg_lambda=reg_lambda,
        sz_related=sz_related,
        n_time_bins=n_time_bins,
        use_yp_attention=True,
        use_xq_attention=False,
        use_x=True,
        use_y=True,
        use_bti=True,
        use_btu=True
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[])

    test_users, test_items, test_ratings, test_times, _, _, _, _, _, _, _ = split_data(test, nu=nu, ni=ni, sz_related=sz_related)
    users, items, ratings, times, \
        ur, ir, \
        min_item_times, max_item_times, \
        min_user_times, max_user_times, \
        user_time_means = split_data(train, nu=nu, ni=ni, sz_related=sz_related)

    def get_item_time_bins(_times, _items):
        min_it = min_item_times[_items.flatten()]
        max_it = max_item_times[_items.flatten()]
        result = np.floor((_times - min_it) / (max_it - min_it + 1e-7) * n_time_bins)
        result = np.int64(result)
        result = np.maximum(np.minimum(result, n_time_bins - 1), 0)
        # if np.any(np.isnan(result)): throw Exception('Invalid division')
        return result
    
    def get_user_times(_times, _users):
        min_it = min_user_times[_users.flatten()]
        max_it = max_user_times[_users.flatten()]
        result = (_times - min_it) / (max_it - min_it + 1e-7)
        result = np.maximum(np.minimum(result, 1.0), 0.0)
        return result

    item_time_bins = get_item_time_bins(times, items)
    user_times = get_user_times(times, users)
    test_item_time_bins = get_item_time_bins(test_times, test_items)
    test_user_times = get_user_times(test_times, test_users)

    test_ur, test_ir = ur[test_users.flatten()], ir[test_items.flatten()]
    train_ur, train_ir = ur[users.flatten()], ir[items.flatten()]

    test_u_time_means = user_time_means[test_users.flatten()]
    u_time_means = user_time_means[users.flatten()]

    test_input = [test_users, test_items, test_user_times, test_ur, test_ir, test_item_time_bins, test_u_time_means]
    validation_data = (test_input, test_ratings)
    model.fit([users, items, user_times, train_ur, train_ir, item_time_bins, u_time_means], ratings,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data)

    preds = model.predict(test_input)
    s = '\n'.join(['{} {}'.format(test_ratings[i], preds[i]) for i in range(preds.shape[0])])
    open(output, 'w').write(s)

if __name__ == '__main__':
    main()
