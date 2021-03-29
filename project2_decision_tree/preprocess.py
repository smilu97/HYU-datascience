import numpy as np

def encode_category(df):
  n = df.shape[-1]
  tbs = [np.unique(df[:,i]) for i in range(n)]
  res = np.empty_like(df)
  src = df.reshape((-1, n))
  dst = res.reshape((-1, n))
  for i in range(src.shape[0]):
    for j in range(n):
      dst[i,j] = np.where(tbs[j] == src[i,j])[0][0]
  return res.astype(np.int32), tbs
