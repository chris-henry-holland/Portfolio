#!/usr/bin/env python

from collections import deque
import math
import random
from sortedcontainers import SortedList
import time
from typing import Generator, Dict, List, Set, Tuple, Optional, Union

def determinant(mat: List[List[Union[int, float]]]) -> Union[int, float]:
    """
    Calculates the determinant of a square matrix mat.
    """
    n = len(mat)
    cols = SortedList(range(n))
    def recur(start_row_idx: int) -> Union[int, float]:
        if len(cols) == 1:
            return mat[start_row_idx][cols[0]]
        mult = 1
        res = 0
        for i in range(len(cols)):
            col_idx = cols.pop(i)
            mult2 = ((not col_idx & 1) << 1) - 1
            res += mult * mat[start_row_idx][col_idx] *\
                    recur(start_row_idx + 1)
            cols.add(col_idx)
            mult *= -1
        return res
    return recur(0)

def circumcircle(points: List[Tuple[Union[int, float]]])\
        -> Tuple[Union[Tuple[Union[int, float]], Union[int, float]]]:
    """
    For a set of between 1 and 3 points (inclusive) in the 2d plane,
    finds the centre and radius squared of the smallest circle passing
    through every one of those points.
    
    If three points are given, these should not be colinear.
    """
    if not (1 <= len(points) <= 3):
        raise ValueError("Function circumcircle() may only be applied to "
                "between 1 and 3 points inclusive.")
    if len(points) == 1:
        return (points[0], 0)
    elif len(points) == 2:
        diam_sq = sum((x - y) ** 2 for x, y in zip(*points))
        rad_sq = (diam_sq >> 2) if isinstance(diam_sq, int) and not diam_sq & 3 else (diam_sq / 4)
        return (tuple((x + y) / 2 for x, y in zip(*points)), rad_sq)
    # Based on https://en.wikipedia.org/wiki/Circumcircle
    abs_sq_points = [sum(x ** 2 for x in pt) for pt in points]
    a = determinant([[*y, 1] for y in points])
    if not a:
        raise ValueError("The three points given to the function circumcircle() "
                "were colinear, in which case there is no finite circumcircle")
    S = (determinant([[x, y[1], 1] for x, y in zip(abs_sq_points, points)]),\
            -determinant([[x, y[0], 1] for x, y in zip(abs_sq_points, points)]))
    
    centre = tuple(x // (2 * a) if isinstance(x, int) and isinstance(a, int) and\
            not x % (2 * a) else x / (2 * a) for x in S)
    rad_sq = sum((x - y) ** 2 for x, y in zip(points[0], centre))
    return (centre, rad_sq)

def welzl(points: List[Tuple[Union[int, float]]], eps: float=10 ** -5)\
        -> Tuple[Union[Tuple[Union[int, float]], Union[int, float]]]:
    """
    Uses the Welzl algorithm to find the centre and radius squared of
    the smallest circle that encloses every one of a set of points
    in the 2d plane (where points on the circle itself are considered
    to be enclosed by the circle).
    """

    # Based on https://en.wikipedia.org/wiki/Smallest-circle_problem
    points = list(set(tuple(x) for x in points))
    n = len(points)
    random.shuffle(points)
    boundary_points = []

    def recur(idx: int) -> Tuple[Union[Tuple[Union[int, float]],\
            Union[int, float]]]:
        if idx == n or len(boundary_points) == 3:
            #print(boundary_points)
            if not boundary_points:
                return ((0, 0), -1)
            return circumcircle(boundary_points)
        pt = points[idx]
        centre, rad_sq = recur(idx + 1)
        if sum((x - y) ** 2 for x, y in zip(centre, pt)) <=\
                rad_sq + eps:
            return centre, rad_sq
        boundary_points.append(pt)
        centre, rad_sq = recur(idx + 1)
        boundary_points.pop()
        return centre, rad_sq
    return recur(0)

def outerTrees(self, trees: List[List[int]]) -> List[float]:
    """
    
    Solution to Leetcode #1924 (Premium)
    
    Original problem description:
    
    You are given a 2D integer array trees where trees[i] = [xi, yi]
    represents the location of the ith tree in the garden.

    You are asked to fence the entire garden using the minimum length
    of rope possible. The garden is well-fenced only if all the trees
    are enclosed and the rope used forms a perfect circle. A tree is
    considered enclosed if it is inside or on the border of the circle.

    More formally, you must form a circle using the rope with a center
    (x, y) and radius r where all trees lie inside or on the circle and
    r is minimum.

    Return the center and radius of the circle as a length 3 array
    [x, y, r]. Answers within 10-5 of the actual answer will be
    accepted.
    """
    eps = 10 ** -5
    centre, rad_sq = welzl(trees, eps=eps)
    return [*centre, math.sqrt(rad_sq)]
