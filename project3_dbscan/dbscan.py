# created at: 2021-05-09 20:34
# author: smilu97

from typing import List, Callable

import numpy as np

class DBScanItem:
    def __init__(self, data: List[float]):
        self.data = np.array(data)

def l2_distance(a: DBScanItem, b: DBScanItem) -> float:
    """Calculate L2 Distance"""
    return np.sqrt(np.sum(np.square(a.data - b.data)))

def dbscan(
    items: List[DBScanItem],
    fn_distance: Callable[[DBScanItem, DBScanItem], float] = l2_distance
) -> List[List[DBScanItem]]:
    '''
    Run DBScan algorithm
    
    Parameters:
        items (list[DBScanItem]): the items in same space
    
    Returns:
        clusters (list[list[DBScanItem]]):clusters which are consist of
            the items from parameter
    '''
    

    