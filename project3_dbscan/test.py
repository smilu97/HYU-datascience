#!/usr/bin/env python

# created at: 2021-05-09 21:03
# author: smilu97

from argparse import ArgumentParser
from dbscan import DBScanItem, dbscan
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--i', default=1)

input_filepaths = (
    'input1.txt', 'input2.txt', 'input3.txt',
)

ideal_clusters_filepaths = (
    ('input1_cluster_{}_ideal.txt'.format(i) for i in range(8)),
    ('input2_cluster_{}_ideal.txt'.format(i) for i in range(5)),
    ('input3_cluster_{}_ideal.txt'.format(i) for i in range(4))
)

hyper_parameters = (
    (8, 15.0, 22),
    (5, 2.0, 7),
    (4, 5.0, 5),
)

colors = (
    (1, 0, 0, 0.5),
    (0, 1, 0, 0.5),
    (0, 0, 1, 0.5),
    (1, 1, 0, 0.5),
    (1, 0, 1, 0.5),
    (0, 1, 1, 0.5),
    (0, 0, 0, 0.5),
    (0.6, 0, 0, 0.5),
    (0, 0.6, 0, 0.5),
    (0, 0, 0.6, 0.5),
    (0.6, 0.6, 0, 0.5),
    (0.6, 0, 0.6, 0.5),
    (0, 0.6, 0.6, 0.5),
)

def clusters_into_dict(cs: List[List[int]]):
    """Convert clusters into dictionary"""
    res: Dict[int, int] = dict()
    for index, c in enumerate(cs):
        for i in c:
            res[i] = index
    return res

def evaluate(sz: int, ideal: List[List[int]], real: List[List[int]]):
    """evaluate score of dbscan result"""
    ideal_dict = clusters_into_dict(ideal)
    real_dict = clusters_into_dict(real)
    cnt = 0
    correct = 0

    for i in range(sz):
        for j in range(i+1,sz):
            # if ideal_dict.get(i) is None: continue
            # if ideal_dict.get(j) is None: continue
            # if real_dict.get(i) is None: continue
            # if real_dict.get(j) is None: continue
            a = ideal_dict.get(i, -1) == ideal_dict.get(j, -1)
            b = real_dict.get(i, -1) == real_dict.get(j, -1)
            cnt += 1
            correct += a == b
    
    return correct / cnt

def show_clusters(clusters: List[List[DBScanItem]]):
    x = np.concatenate([[i.data[0] for i in c] for c in clusters])
    y = np.concatenate([[i.data[1] for i in c] for c in clusters])
    c = np.concatenate([
        [colors[ci % len(colors)] for i in c]
        for ci, c in enumerate(clusters)
    ])
    plt.scatter(x, y, c=c, s=1.0)
    plt.show()

def evaluate_hyper_param(
    items: List[DBScanItem],
    ideal_clusters: List[List[int]],
    n: int,
    sz: int,
    eps: float,
    min_pts: int):
    """Evaluate hyper paramete"""

    clusters = dbscan(items, eps=eps, min_pts=min_pts)[:n]
    clusters_ids = [[x.id for x in c] for c in clusters]
    ev = evaluate(sz, ideal_clusters, clusters_ids)
    print('eps: {}, ev: {}'.format(eps, ev))
    return ev

def main():
    args = parser.parse_args()
    index = int(args.i) - 1

    n, eps, min_pts = hyper_parameters[index]

    with open(input_filepaths[index], 'r') as fd:
        text_input = fd.read()
    input_data = [
            [float(x) for x in l.split()]
        for l in text_input.split('\n') if len(l) > 0
    ]
    items = [DBScanItem(int(d[0]), d[1:]) for d in input_data]
    # sz = len(items)
    
    text_clusters = [
        open(fp, 'r').read() for fp in ideal_clusters_filepaths[index]
    ]
    _ideal_clusters = [
        [int(x) for x in s.split('\n') if len(x) > 0] for s in text_clusters
    ]

    clusters = dbscan(items, eps=eps, min_pts=min_pts)[:n]
    show_clusters(clusters)

if __name__ == '__main__':
    main()
