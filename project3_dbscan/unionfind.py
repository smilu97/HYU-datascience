# created at: 2021-05-09 21:36
# author: smilu97

import numpy as np

class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.par = np.arange(n)
    
    def root(self, x: int):
        if self.par[x] == x:
            return x
        pp = self.root(self.par[x])
        self.par[x] = pp
        return pp
    
    def merge(self, a: int, b: int):
        pa = self.root(a)
        pb = self.root(b)
        if pa == pb: return False
        self.par[pb] = pa
        return True
