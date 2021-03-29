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

  train_df = pd.read_csv(path_train, sep='\t').to_numpy()
  test_df  = pd.read_csv(path_test, sep='\t').to_numpy()
  ans_df   = pd.read_csv(path_answer, sep='\t').to_numpy()[:,-1]

  train_records, tbs = encode_category(train_df)
  test_records, tbs = encode_category_by_tbs(test_df, tbs)
  dec = build_decoder(tbs)

  dt = DecisionTree(train_records, tbs)
  ans = np.array([dec[-1][i] for i in dt.predict(test_records)])

  acc = np.mean(ans == ans_df)

  print('acc:', acc)

test_with_dataset(datasets['first'])
test_with_dataset(datasets['second'])
