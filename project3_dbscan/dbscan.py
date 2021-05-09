# created at: 2021-05-09 20:34
# author: smilu97

from typing import List, Callable, Tuple, DefaultDict
from unionfind import UnionFind
from collections import defaultdict

import numpy as np

class DBScanItem:
    def __init__(self, id: int, data: List[float]):
        self.id = id
        self.data = np.array(data)

def l2_distance(diff: np.array) -> float:
    """Calculate L2 Distance"""
    return np.sqrt(np.sum(np.square(diff), axis=-1))

def dbscan(
    items: List[DBScanItem],
    eps: float,
    min_pts: int,
    fn_distance: Callable[[np.array, np.array], float] = l2_distance,
    **kwargs,
) -> List[List[DBScanItem]]:
    '''
    Run DBScan algorithm
    
    Parameters:
        items (list[DBScanItem]): the items in same space
        eps (float): maximum radius of the neighborhood
        min_pts (int): minimum number of points in an Eps-neighborhood
            of a given point
    
    Returns:
        clusters (list[list[DBScanItem]]):clusters which are consist of
            the items from parameter
    '''
    
    sz = len(items)
    dim = items[0].data.shape[-1]
    arr = np.array([x.data for x in items], dtype=np.float64)
    adj = l2_distance(arr.reshape((sz, 1, dim)) - arr) <= eps # adjacent matrix
    neighbors = [[i for i in range(sz) if nbs[i]] for nbs in adj]
    is_core = np.array([len(i) for i in neighbors]) >= min_pts
    is_noise = np.zeros(sz)
    uf = UnionFind(sz)
    for p in range(sz):
        if is_core[p]:
            for q in neighbors[p]:
                if is_core[q]:
                    uf.merge(p, q)
        else:
            found = False
            for q in neighbors[p]:
                if is_core[q]:
                    found = True
                    uf.merge(p, q)
                    break
            if not found: is_noise[p] = 1

    clusters: DefaultDict[int, List[int]] = defaultdict(list)
    for i in range(sz):
        if is_noise[i]: continue
        clusters[uf.root(i)].append(i)
    clusters = [[items[i] for i in c] for c in clusters.values()]
    clusters = sorted(clusters, key=lambda x: -len(x))
    
    return clusters