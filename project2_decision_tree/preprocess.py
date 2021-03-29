import numpy as np

def reverse_dict(d):
  res = {}
  for key in d:
    value = d[key]
    res[value] = key
  return res

def build_decoder(tbs):
  return [reverse_dict(d) for d in tbs]

def map_unique_numbers(arr):
  ul = np.unique(arr)
  res = {}
  for i in range(ul.shape[0]):
    res[ul[i]] = i
  return res

def encode_category_by_tbs(df, tbs):
  n = df.shape[-1]
  res = np.empty_like(df)
  src = df.reshape((-1, n))
  dst = res.reshape((-1, n))
  for i in range(src.shape[0]):
    for j in range(n):
      dst[i,j] = tbs[j][src[i,j]]
  return res.astype(np.int32), tbs

def encode_category(df):
  n = df.shape[-1]
  tbs = [map_unique_numbers(df[:,i]) for i in range(n)]
  return encode_category_by_tbs(df, tbs)
