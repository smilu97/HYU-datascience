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

def read_dataset(index):
    df_base = pd.read_csv('./data/u{}.base'.format(index), sep='\t', header=None)
    df_test = pd.read_csv('./data/u{}.test'.format(index), sep='\t', header=None)
    np_base = df_base.to_numpy()
    np_test = df_test.to_numpy()
    np_all = np.concatenate([np_base, np_test])

    user_id_bw_mapper, user_id_fd_mapper = assign_numbers(np_all[:, 0])
    item_id_bw_mapper, item_id_fd_mapper = assign_numbers(np_all[:, 1])

    enc_base = np_all[:np_base.shape[0]]
    enc_test = np_all[np_base.shape[0]:]

    return enc_base, enc_test, user_id_bw_mapper, item_id_bw_mapper
