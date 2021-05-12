# created at: 2021-05-12 12:24
# author: smilu97

from collections import defaultdict

import numpy as np
import hashlib

class Geocoding:
    def __init__(self, data: np.array, box_size: float):
        self.data = data
        assert(len(data.shape) == 2)
        self.n = data.shape[0]
        self.dim = data.shape[-1]
        self.box_size = box_size
    
        self._organize()
    
    def neighbors(self, point: np.array, rg: int = 0):
        base_from = self._base(point) - rg
        diam = (2 * rg) + 1
        iterator = np.array(list(np.ndindex((diam,) * self.dim)))
        bs = iterator + base_from
        hs = [self._hash(x) for x in bs]
        ds = [self.mapper[x] for x in hs]
        return np.int64(np.concatenate(ds))
    
    def _base(self, point: np.array):
        return np.int64(np.floor(point / self.box_size))

    def _hash(self, obj):
        s = bytes(self._stringify(obj), 'utf8')
        enc = hashlib.md5()
        enc.update(s)
        d = enc.digest()
        return int.from_bytes(d, 'little')
    
    def _stringify(self, obj):
        return '-'.join([str(x) for x in obj])

    def _organize(self):
        base = self._base(self.data)
        hs = [self._hash(x) for x in base]
        mapper = defaultdict(list)
        for index, h in enumerate(hs):
            mapper[h].append(index)
        self.mapper = mapper
        self.hs = hs
