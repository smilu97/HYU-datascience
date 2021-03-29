#!/usr/bin/env python

import numpy as np
import pandas as pd

from preprocess import encode_category

df = pd.read_csv('./data/dt_test1.txt', sep='\t')
df = df.to_numpy()
df, tbs = encode_category(df)

print(df)
