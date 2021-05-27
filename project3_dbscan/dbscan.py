# created at: 2021-05-09 20:34
# author: smilu97

from typing import List, Callable, Tuple, DefaultDict
from unionfind import UnionFind
from collections import defaultdict
from geocoding import Geocoding

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
    arr = np.array([x.data for x in items], dtype=np.float64)

    geo = Geocoding(arr, eps)
    neighbors = [
        [x for x in geo.neighbors(arr[i], rg=1)
            if fn_distance(arr[i] - arr[x]) <= eps
        ]
        for i in range(sz)
    ]
    is_core = np.array([len(i) for i in neighbors]) >= min_pts
    is_noise = np.zeros(sz)
    uf = UnionFind(sz)
    for p in range(sz):
        if is_core[p]:
            for q in neighbors[p]:
                if is_core[q]:
                    uf.merge(p, q)
        else:
            core_neighbors = [q for q in neighbors[p] if is_core[q]]
            if len(core_neighbors) == 0:
                is_noise[p] = 1
            else:
                uf.merge(p, min(core_neighbors))

    clusters: DefaultDict[int, List[int]] = defaultdict(list)
    for i in range(sz):
        if is_noise[i]: continue
        clusters[uf.root(i)].append(i)
    clusters = [[items[i] for i in c] for c in clusters.values()]
    clusters = sorted(clusters, key=lambda x: -len(x))
    
    return clusters