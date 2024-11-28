#! /usr/bin/env python
from collections import deque
import heapq
import inspect
from typing import Dict, List, Set, Tuple, Optional, Union, Hashable,\
        Generator, Any, Callable

from Graph_classes.utils import UnionFind

def kruskalIndex(self) -> Tuple[Union[float, int], Set[Tuple[int, int, int]]]:
    """
    Implementation of Kruskal's algorithm to find a minimum spanning
    tree or forest of a weighted graph, returning the included edges in
    terms of the indices of the vertices they connect.
    For a connected graph a minimum spanning tree is the tree
    connecting all of the vertices of the graph (where the edges are
    considered to be undirected) such that the sum of weights of the
    edges is no larger than that of any other such tree.
    For a non-connected graph, a minimum spanning forest is a
    union of trees over the connected components of the graph, such
    that each such tree is a minimum spanning tree of the corresponding
    connected component.

    Args:
        None

    Returns:
    A 2-tuple whose index 0 contains the sum of the weights of edges
    of any minimum spanning tree or forest over the graph, and whose
    index 1 contains a set representing each edge in one of the
    minimum spanning tress or forests, in the form of a 3-tuple where
    indices 0 and 1 contain the indices of the vertices the edge
    connects and index 2 contains the weight of that edge.
    """
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

def kruskal(self) -> Tuple[Union[float, int], Set[Tuple[Hashable, Hashable, int]]]:
    """
    Implementation of Kruskal's algorithm to find a minimum spanning
    tree or forest of a weighted graph, returning the included edges in
    terms of the defined labels of the vertices they connect.
    For a connected graph a minimum spanning tree is the tree
    connecting all of the vertices of the graph (where the edges are
    considered to be undirected) such that the sum of weights of the
    edges is no larger than that of any other such tree.
    For a non-connected graph, a minimum spanning forest is a
    union of trees over the connected components of the graph, such
    that each such tree is a minimum spanning tree of the corresponding
    connected component.

    Args:
        None

    Returns:
    A 2-tuple whose index 0 contains the sum of the weights of edges
    of any minimum spanning tree or forest over the graph, and whose
    index 1 contains a set representing each edge in one of the
    minimum spanning tress or forests, in the form of a 3-tuple where
    indices 0 and 1 contain the defined labels of the vertices the edge
    connects and index 2 contains the weight of that edge.
    """
    cost, edges = self.kruskalIndex()
    return (cost, {(self.index2Vertex(e[0]),\
            self.index2Vertex(e[1]), e[2]) for e in edges})
