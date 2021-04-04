#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd

from preprocess import encode_category, encode_category_by_tbs, build_decoder
from decision_tree import DecisionTree

def main():
  argc = len(sys.argv)
  if argc < 4:
    print('usage: {} train_file test_file output_file'.format(sys.argv[0]))
    exit(-1)
  
  fp_train = sys.argv[1]
  fp_test  = sys.argv[2] 
  fp_output = sys.argv[3]

  df_train = pd.read_csv(fp_train, sep='\t')
  df_test  = pd.read_csv(fp_test,  sep='\t')

  data = [
    df_train.to_numpy(),
    np.pad(df_test.to_numpy(), ((0, 0), (0, 1)), 'constant', constant_values=df_train[df_train.columns[-1]][0]),
  ]
  data = np.concatenate(data, axis=0)
  records, tbs = encode_category(data)
  sz_train = len(df_train)
  train_records = records[:sz_train]
  test_records = records[sz_train:]
  dec = build_decoder(tbs)

  dt = DecisionTree(train_records, tbs)
  ans = np.array([dec[-1][i] for i in dt.predict(test_records)])
  
  df_test[df_train.columns[-1]] = ans
  df_test.to_csv(fp_output, sep='\t')
    
if __name__ == '__main__':
  main()
