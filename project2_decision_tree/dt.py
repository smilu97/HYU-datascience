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

  df_train = pd.read_csv(fp_train, sep='\t').to_numpy()
  df_test  = pd.read_csv(fp_test,  sep='\t').to_numpy()

  train_records, tbs = encode_category(df_train)
  test_records, tbs = encode_category_by_tbs(test_df, tbs)
  dec = build_decoder(tbs)

  dt = DecisionTree(train_records, tbs)
  ans = np.array([dec[-1][i] for i in dt.predict(test_records)])
  
  df_test[df_train.columns[-1]] = ans
  df_test.to_csv(fp_output, sep='\t')
    

if __name__ == '__main__':
  main()
