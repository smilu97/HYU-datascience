# created at: 2021-05-12 12:24
# author: smilu97

from collections import defaultdict
from typing import Iterable

import numpy as np
import hashlib

class Geocoding:
    '''
    Assume N-dim euclid space is quantimized by certain length (`box_size`),
    and each points in space are mapped into certain quantumized hyper-cube
    many-to-one. Points in same hyper-cube satisfy locality, and we can easily
    find neighbor hyper-cube.
    '''
    def __init__(self, data: np.array, box_size: float):
        '''
        Calculate mapper (`geocode unitbox index` => `item indices`)
        '''
        self.data = data
        assert(len(data.shape) == 2)
        self.n = data.shape[0]
        self.dim = data.shape[-1]
        self.box_size = box_size
    
        base = Geocoding._base(self.data, self.box_size)
        hs = [Geocoding._hash(x) for x in base]
        mapper = defaultdict(list)
        for index, h in enumerate(hs):
            mapper[h].append(index)
        self.mapper = mapper
        self.hs = hs
    
    def neighbors(self, point: np.array, rg: int = 0):
        '''Find geocode neighbors'''
        base_from = Geocoding._base(point, self.box_size) - rg
        diam = (2 * rg) + 1
        iterator = np.array(list(np.ndindex((diam,) * self.dim)))
        bs = iterator + base_from
        hs = [Geocoding._hash(x) for x in bs]
        ds = [self.mapper[x] for x in hs]
        return np.int64(np.concatenate(ds))
        
    @staticmethod
    def _base(point: np.array, box_size: float):
        '''Find base index of geocode unitbox'''
        return np.int64(np.floor(point / box_size))
    
    @staticmethod
    def _stringify(obj: Iterable[object]):
        '''Stringify array'''
        return '-'.join([str(x) for x in obj])

    @staticmethod
    def _hash(obj: Iterable[object]):
        '''Hash iterable, and convert into int'''
        s = bytes(Geocoding._stringify(obj), 'utf8')
        enc = hashlib.md5()
        enc.update(s)
        d = enc.digest()
        return int.from_bytes(d, 'little')
