from collections import defaultdict

import numpy as np
import pandas as pd

def assign_numbers(arr):
    uniq_arr = np.unique(arr)
    fd_mapper = {}
    bw_mapper = {}
    for i in range(uniq_arr.shape[0]):
        fd_mapper[uniq_arr[i]] = i
        bw_mapper[i] = uniq_arr[i]
    for i in range(arr.shape[0]):
        arr[i] = fd_mapper[arr[i]]
    return bw_mapper, fd_mapper

def midpoint(a, b):
    if a == 0: return b
    if b == 0: return a
    return (a + b) // 2

def augment_data(tbl, n, m):
    dt = defaultdict(list)
    u_times = [list() for _ in range(n)]
    i_times = [list() for _ in range(m)]
    adj = (-1) * np.ones((n, m))
    for idx in range(tbl.shape[0]):
        u, i, r, t = tbl[idx, :4]
        adj[u, i] = r
        u_times[u].append(t)
        i_times[i].append(t)
    ut_means = [sum(l)/len(l) if len(l) > 0 else 0 for l in u_times]
    it_means = [sum(l)/len(l) if len(l) > 0 else 0 for l in i_times]
    augments = []
    for i in range(n):
        for j in range(m):
            if adj[i,j] != -1: continue
            t = midpoint(ut_means[i], it_means[j])
            augments.append([i, j, 1, t])
    return np.array(augments, dtype=np.int64)

def read_dataset(prefix, augment=True):
    df_base = pd.read_csv('{}.base'.format(prefix), sep='\t', header=None)
    df_test = pd.read_csv('{}.test'.format(prefix), sep='\t', header=None)
    np_base = df_base.to_numpy()
    np_test = df_test.to_numpy()
    np_all = np.concatenate([np_base, np_test])

    user_id_bw_mapper, user_id_fd_mapper = assign_numbers(np_all[:, 0])
    item_id_bw_mapper, item_id_fd_mapper = assign_numbers(np_all[:, 1])
    n, m = len(user_id_bw_mapper), len(item_id_bw_mapper)

    enc_base = np_all[:np_base.shape[0]]
    enc_test = np_all[np_base.shape[0]:]

    if augment:
        augments = augment_data(enc_base, n, m)
        augmented = np.concatenate([enc_base, augments], axis=0)
    else:
        augmented = enc_base

    return augmented, enc_test, user_id_bw_mapper, item_id_bw_mapper
