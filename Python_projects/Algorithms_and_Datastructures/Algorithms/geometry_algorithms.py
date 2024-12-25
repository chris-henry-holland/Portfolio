#!/usr/bin/env python

from collections import deque
import math
import random
from sortedcontainers import SortedList
import time
from typing import Dict, List, Set, Tuple, Optional, Union

def determinant(mat: List[List[Union[int, float]]]) -> Union[int, float]:
    """
    Calculates the determinant of a square matrix mat.

    Args:
        Required positional:
        mat (list of lists of real numeric values): The matrix
                whose determinant is to be calculated, expressed
                as a list of the rows of the matrix. This is
                required be a square matrix (i.e. all of the
                lists within mat are the same length as each
                other and mat itself).
    
    Returns:
    Real numeric value (int or float) giving the determinant of
    the matrix mat.
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
    through every one of those points (the so-called circumcircle),
    specifying the circle by the Cartesian coordinates of its centre
    and its radius squared.
    
    If three points are given, these should not be colinear (i.e. there
    should not exist a single straight line that passes exactly through
    all three points).

    Args:
        Required positional:
        points (list of 2-tuples of real numeric values): A list of no
                more than 3 points, expressed in Cartesian coordinates
                for which the circumcircle is to be calculated. For
                lists of three points, these should not be colinear.
    
    Returns:
    2-tuple whose index 0 contains a 2-tuple of real numeric values
    specifying the location of the centre of the circumcircle
    identified in Cartesian coordinates and whose index 1 contains
    a non-negative numeric value specifying the radius squared of
    the circumcircle.
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

def welzl(points: List[Tuple[Union[int, float]]])\
        -> Tuple[Union[Tuple[Union[int, float]], Union[int, float]]]:
    """
    Uses the Welzl algorithm to find the centre and radius squared of
    the smallest circle that encloses every one of a set of points
    in the 2D plane (where points on the edge of circle itself are
    considered to be enclosed by the circle).

    Args:
        Required positional:
        points (list of 2-tuples with numeric values): The 2D Cartesian
                coordinates of the points to be enclosed.
    
    Returns:
    2-tuple whose index 0 contains a 2-tuple of real numeric values
    specifying the location of the centre of the enclosing circle
    identified in Cartesian coordinates and whose index 1 contains
    a non-negative numeric value specifying the radius squared of
    that enclosing circle.
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
        if sum((x - y) ** 2 for x, y in zip(centre, pt)) <= rad_sq:
            return centre, rad_sq
        boundary_points.append(pt)
        centre, rad_sq = recur(idx + 1)
        boundary_points.pop()
        return centre, rad_sq
    return recur(0)

def smallestCircularEnclosure(trees: List[List[int]]) -> List[float]:
    """
    Finds the circular enclosure with the smallest circumference that
    encloses every 2D point in the list trees. Points that fall on
    the edge of the enclosure are considered to be inside the enclosure.

    Args:
        Required positional:
        trees (list of list of ints): The 2D points to be enclosed,
                expressed in Cartesian coordinates (such that each
                element of the list is a length 2 list containing
                integers).
    
    Returns:
    A list containing exactly three real numeric values, with the
    first two giving the x and y coordinates of the centre of the
    circular enclosure respectively and the final value giving its
    radius.

    Solution to Leetcode #1924 (Premium): Erect the Fence II
    
    Original problem description of Leetcode #1924:
    
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

def grahamScan(points: List[Tuple[Union[int, float]]], include_border_points: bool=False) -> List[Tuple[int]]:
    """
    Implementation of the Graham scan to find the convex hull of a set
    of points in 2 dimensional space expressed in Cartesian coordinates.

    Args:
        Required positional:
        points (list of 2-tuples of real numeric values): The points
                in two dimensional space for which the convex hull
                is to be found.
        
        Optional named:
        include_border_points (bool): Whether to include elements of
                points that fall on an edge of the convex hull but are
                not vertices of the convex hull (i.e. points that are
                on the line directly between two consecutive vertices
                of the convex hull).
            Default: False

    Returns:
    The points in the convex hull expressed in Cartesian coordinates,
    ordered such that they cycle the convex hull in an anticlockwise
    direction, with the point with the smallest x-coordinate (and if
    there are several of these, the one of those with the smallest
    y-coordinate) as the first element.

    Examples:
        >>> grahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=False)
        [(1, 1), (2, 0), (4, 2), (2, 4)]

        >>> grahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=True)
        [(1, 1), (2, 0), (4, 2), (3, 3), (2, 4)]

        Note that this includes the point (3, 3) which was omitted by
        the previous example (which consists of the same points but
        gives the parameter include_border_points as False) as it
        falls directly on the line between the points (4, 2) and (2, 4),
        which are consecutive vertices of the convex hull.
    """
    points = [tuple(x) for x in points]
    if len(points) < 3: return sorted(set(points))

    comp = (lambda x, y: x <= y) if include_border_points else (lambda x, y: x < y)
    
    ref_pt = min(points)
    sorted_pts = []
    min_x_pts = []
    for pos in points:
        if pos[0] == ref_pt[0]:
            min_x_pts.append(pos)
            continue
        diff = tuple(x - y for x, y in zip(pos, ref_pt))
        slope = diff[1] / diff[0]
        sorted_pts.append((slope, diff, pos))
    sorted_pts.sort()
    
    if len(min_x_pts) > 1:
        if include_border_points:
            tail = sorted(min_x_pts)
            pos = tail.pop()
            sorted_pts.append((None, tuple(x - y for x, y in zip(pos, ref_pt)), pos))
            tail = tail[::-1]
        else:
            pos = max(min_x_pts)
            sorted_pts.append((None, tuple(x - y for x, y in zip(pos, ref_pt)), pos))
            tail = [ref_pt]
    else:
        tail = []
        tup0 = sorted_pts.pop()
        diff0 = tup0[1]
        while sorted_pts and diff0[0] * sorted_pts[-1][1][1] == diff0[1] * sorted_pts[-1][1][0]:
            tail.append(sorted_pts.pop()[2])
        tail = tail[::-1] if include_border_points else []
        tail.append(ref_pt)
        sorted_pts.append(tup0)
    stk = [(sorted_pts[0][2], tuple(x - y for x, y in zip(sorted_pts[0][2], ref_pt)))]
    order = [x[0] for x in stk]
    for i in range(1, len(sorted_pts)):
        pos = sorted_pts[i][2]
        while stk:
            diff = tuple(x - y for x, y in zip(pos, stk[-1][0]))
            cross_prod = stk[-1][1][0] * diff[1] -\
                    stk[-1][1][1] * diff[0]
            if comp(0, cross_prod): break
            stk.pop()
        
        stk.append((pos, tuple(x - y for x, y in zip(pos, (stk[-1][0] if stk else ref_pt)))))
    res = [x[0] for x in stk] + tail

    return [res[-1], *res[:-1]]

def outerTrees(trees: List[List[int]]) -> List[List[int]]:
    """
    Given the two dimensional Cartesian coordinates of multiple trees,
    consider the fence of minimal length that encloses all of the
    trees (with trees in contact with the fence considered to be
    enclosed). This function finds the coordinates of the trees that
    are in contact with the fence.

    Args:
        Required positional:
        trees (list of lists of ints): List of 2-dimensional integer
                Cartesian coordinates (each given as lists of length 2
                with integer values) representing the positions of the
                trees to be enclosed.

    Returns:
    List of lists, where each element of the outer list has length 2
    and contains integers (int), giving the Cartesian coordinates of
    all the trees that are in contact with the minimal length fence
    that encloses all of the trees given.
    
    Examples:
        >>> outerTrees([[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]])
        [[1, 1], [2, 0], [4, 2], [3, 3], [2, 4]]

        >>> outerTrees([[1, 2], [2, 2], [4, 2]])
        [[1, 2], [4, 2], [2, 2]]
    
    Solution to Leetcode #587: Erect the Fence

    Original problem description for Leetcode #587:

    You are given an array trees where trees[i] = [xi, yi] represents the
    location of a tree in the garden.

    Fence the entire garden using the minimum length of rope, as it is
    expensive. The garden is well-fenced only if all the trees are
    enclosed.

    Return the coordinates of trees that are exactly located on the fence
    perimeter. You may return the answer in any order.
    """
    return [list(y) for y in grahamScan([tuple(x) for x in trees], include_border_points=True)]

if __name__ == "__main__":

    res = grahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=False)
    print(f"\ngrahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=False) = {res}")

    res = grahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=True)
    print(f"\ngrahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=True) = {res}")

    res = outerTrees([[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]])
    print(f"\nouterTrees([[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]]) = {res}")

    res = outerTrees([[1, 2], [2, 2], [4, 2]])
    print(f"\nouterTrees([[1, 2], [2, 2], [4, 2]]) = {res}")