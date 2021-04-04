#!/usr/bin/env python

import numpy as np
import pandas as pd

from preprocess import encode_category, encode_category_by_tbs, build_decoder
from decision_tree import DecisionTree

datasets = {
  'first': (
    './data/dt_train.txt',
    './data/dt_test.txt',
    './test/dt_answer.txt',
  ),
  'second': (
    './data/dt_train1.txt',
    './data/dt_test1.txt',
    './test/dt_answer1.txt',
  ),
}

def test_with_dataset(dataset):
  path_train  = dataset[0]
  path_test   = dataset[1]
  path_answer = dataset[2]

  train_df = pd.read_csv(path_train, sep='\t')
  test_df  = pd.read_csv(path_test, sep='\t')
  ans_df   = pd.read_csv(path_answer, sep='\t').to_numpy()[:,-1]

  data = [
    train_df.to_numpy(),
    np.pad(test_df.to_numpy(), ((0, 0), (0, 1)), 'constant', constant_values=train_df[train_df.columns[-1]][0]),
  ]
  data = np.concatenate(data, axis=0)
  records, tbs = encode_category(data)
  sz_train = len(train_df)
  train_records = records[:sz_train]
  test_records = records[sz_train:]
  dec = build_decoder(tbs)

  dt = DecisionTree(train_records, tbs)
  ans = np.array([dec[-1][i] for i in dt.predict(test_records)])

  acc = np.mean(ans == ans_df)

  print('acc:', acc)

test_with_dataset(datasets['first'])
test_with_dataset(datasets['second'])
