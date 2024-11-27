#! /usr/bin/env python
from collections import deque
import heapq
import inspect
from typing import Dict, List, Set, Tuple, Optional, Union, Hashable,\
        Generator, Any, Callable

from Graph_classes.utils import UnionFind

def kruskalIndex(self) -> Tuple[Union[float, int, Set[int]]]:
        
    uf = UnionFind(self.n)
    res = set()
    cost = 0
    edge_heap = []
    adj = getattr(self, self.adj_name)
    for i1 in range(self.n):
        for i2, weights in adj[i1].items():
            edge_heap.append((i1, i2, weights[0]))
    while edge_heap:
        i1, i2, w = heapq.heappop(edge_heap)
        if uf.connected(i1, i2): continue
        uf.union(i1, i2)
        res.add((i1, i2, w))
        cost += w
        if len(res) == self.n - 1: break
    return (cost, res)

def kruskal(self) -> Tuple[Union[float, int, Set[Hashable]]]:
    
    cost, edges = self.kruskalIndex()
    return (cost, {(self.index2Vertex(e[0]),\
            self.index2Vertex(e[1]), e[2]) for e in edges})
