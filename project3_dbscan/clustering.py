#!/usr/bin/env python

from argparse import ArgumentParser
from dbscan import DBScanItem, dbscan
from typing import List, Dict

import numpy as np

parser = ArgumentParser('clustering')
parser.add_argument('input', type=str, help='input filename')
parser.add_argument('n', type=int, help='the maximum number of clusters to emit')
parser.add_argument('eps', type=float, help='maximum distance of neighbor')
parser.add_argument('min_pts', type=int, help='the minimum number of neighbors to be core')

def main():
    args = parser.parse_args()
    
    n = args.n
    eps = args.eps
    min_pts = args.min_pts
    i_filepath = args.input

    with open(i_filepath, 'r') as fd:
        raw = fd.read()

    input_data = [
        [float(x) for x in l.split()]
        for l in raw.split('\n') if len(l) > 0
    ]
    items = [DBScanItem(int(d[0]), d[1:]) for d in input_data]

    clusters = dbscan(items, eps=eps, min_pts=min_pts)[:n]

    prefix_filepath = i_filepath[:-4] + '_cluster_'
    for ci, c in enumerate(clusters):
        o_filepath = prefix_filepath + str(ci) + '.txt'
        content = '\n'.join([str(item.id) for item in c])
        with open(o_filepath, 'w') as fd:
            fd.write(content)

if __name__ == '__main__':
    main()
