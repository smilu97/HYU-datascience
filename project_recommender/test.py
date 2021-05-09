#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from preprocess import read_dataset
from svdpp import SVDPPModel

datasets = [read_dataset(i) for i in range(1, 6)]

def test_svdpp():
    train, test, user_mapper, item_mapper = datasets[0]
    mu = np.mean(train[:,2])
    epoch = 100
    batch_size = 500
    k = 12
    reg_lambda = 0.0001
    lr = 0.0005
    model = SVDPPModel(
        num_users=max(user_mapper.values()) + 1,
        num_items=max(item_mapper.values()) + 1,
        mu=mu,
        k=k,
        reg_lambda=reg_lambda
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[])

    test_users = test[:, 0].reshape((-1, 1))
    test_items = test[:, 1].reshape((-1, 1))
    test_ratings = test[:, 2]

    def get_test_mse():
        preds = model.predict([test_users, test_items])
        errors = preds.reshape((-1,)) - test_ratings
        return np.mean(np.abs(errors))

    print('mu:', mu)
    prev_mse = get_test_mse()
    min_mse = 1000.0
    min_iter = -1
    
    for i in range(epoch):
        indices = np.arange(train.shape[0])
        np.random.shuffle(indices)
        data = train[indices]
        users = data[:, 0].reshape((-1, 1))
        items = data[:, 1].reshape((-1, 1))
        ratings = data[:, 2]
        model.fit([users, items], ratings, batch_size=batch_size)
        curr_mse = get_test_mse()
        if curr_mse < min_mse:
            min_mse = curr_mse
            min_iter = i
        print(i, 'test:', curr_mse)

    
    print('prev_mse:', prev_mse)
    print('min_mse, min_iter:', min_mse, min_iter)
    print('test:', get_test_mse())

    preds = model.predict([test_users, test_items]).reshape((-1,))
    s = '\n'.join(['{} {}'.format(test_ratings[i], preds[i]) for i in range(preds.shape[0])])
    open('output.txt', 'w').write(s)

if __name__ == '__main__':
    test_svdpp()
