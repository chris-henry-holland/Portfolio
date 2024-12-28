#!/usr/bin/env python

import heapq
import itertools
import math
import os
import sys
import time

import numpy as np
import scipy.special as sp
import sympy as sym

from collections import deque
from sortedcontainers import SortedDict, SortedList
from typing import Dict, List, Tuple, Set, Union, Generator, Callable, Optional, Any, Hashable

sys.path.append(os.path.join(os.path.dirname(__file__), "../Algorithms_and_Datastructures/Data_structures"))
from prime_sieves import PrimeSPFsieve

def gcd(a: int, b: int) -> int:
    return a if not b else gcd(b, a % b)
    
def lcm(a: int, b: int) -> int:
    return a * (b // gcd(a, b))

def isqrt(n: int) -> int:
    """
    For a non-negative integer n, finds the largest integer m
    such that m ** 2 <= n (or equivalently, the floor of the
    positive square root of n).
    Uses Newton's method.
    
    Args:
        Required positional:
        n (int): The number for which the above process is
                performed.
    
    Returns:
    Integer (int) giving the largest integer m such that
    m ** 2 <= n.
    
    Examples:
    >>> isqrt(4)
    2
    >>> isqrt(15)
    3
    """
    x2 = n
    x1 = (n + 1) >> 1
    while x1 < x2:
        x2 = x1
        x1 = (x2 + n // x2) >> 1
    return x2

# Problem 101- Look into Lagrange polynomial interpolation
def polynomialFit(seq: List[int], n0=0) -> Tuple[Tuple[int], int]:
    """
    For an integer sequence seq such that the first value in seq
    corresponds to n = n0, finds the coefficients of the lowest order
    polynomial P(n) such that P(n) = seq[n - n0] for each integer n
    between n0 and n0 + len(seq) - 1 inclusive.
    
    Args:
        Required positional:
        seq (List of ints): The integer sequence in question
        
        Optional named:
        n0 (int): The value of n to which the first element of seq
                corresponds
            Default: 0
    
    Returns:
    Tuple whose 0th index contains a tuple of integers representing
    the numerators of the coefficients of the polynomial, where the
    entry at index i represents the coefficient of the n ** i term,
    and whose 1st index contains an integer (int) representing the
    denominator of all those coefficients.
    """
    m = len(seq)
    #mat = np.zeros((m, m), dtype=np.uint64)
    mat = [[0] * m for _ in range(m)]
    for i1 in range(n0, n0 + m):
        mat[i1 - n0][0] = 1
        if not i1: continue
        for i2 in range(1, m):
            mat[i1 - n0][i2] = i1 ** i2
    mat = sym.matrices.Matrix(mat)
    vec = sym.matrices.Matrix(m, 1, seq)
    #res = np.linalg.solve(mat, np.array(seq, dtype=np.int64))
    res = mat.LUsolve(vec)
    res = list(res)
    while len(res) > 1 and not res[-1]:
        res.pop()
    denom = 1
    for frac in res:
        if hasattr(frac, "denominator"):
            denom = lcm(denom, frac.denominator())
    return (tuple(int(x * denom) for x in res), denom)

def optimumPolynomial(coeffs: Tuple[int]=(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1))\
        -> Union[int, float]:
    """
    Solution to Project Euler #101
    Consider a polynomial P(n) (where n is the variable of the
    polynomial) whose coefficients are given by the tuple of ints
    coeffs, where the value at the ith index of coeffs is the
    coefficient of the n ** i term in the polynomial.
    
    For a given sequence of finite length, the optimum polynomial Q(n)
    is the polynomial with minimum degree such that for Q(i) is equal
    to the ith element of the sequence for all integers i between 1
    and the length of the sequence. It can be shown that this uniquely
    defines such a polynomial.
    
    For the polynomial P(n) define OP(k, m) to be the mth term of the
    optimum polynomial for the sequence of length k such that the
    ith term in the sequence is equal to P(i).
    
    This function calculates the sum of OP(k, m(k)) over all
    strictly positive integers k for which m(k) is defined, where for
    each k, m(k) is the smallest positive integer m for which
    OP(k, m) is not equal to P(m), if any, and is undefined if there
    is no such m.
    
    Note that for k > (degree of P(n)), OP(k, m) = P(m) for all m,
    given that for any sequence defined to be a polynomial with more
    terms than the degree of the polynomial, a consequence of the
    fundamental theorem of algebra is that the optimal polynomial
    must be equal to the polynomial that defined the sequence.
    Therefore, only the terms where k is between 1 and the degree of
    P(n) contribute to the sum.
    
    Args:
        Optional named:
        coeffs (tuple of ints): The coefficients of the polynomial P(n)
                where the value at the ith index of the tuple is the
                coefficient of the n ** i term in the polynomial.
            Default: (1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1)
    
    Returns:
    Number (int/float) giving the described sum.
    """
    since = time.time()
    n = len(coeffs)
    res = [0, 1]
    ans_real = sum(coeffs)
    seq = [ans_real]
    skip_count = 0
    i = 2
    while i <= n or skip_count:
        if not skip_count:
            poly, denom = polynomialFit(seq, n0=1)
        #print(seq)
        #print(poly)
        ans_poly = poly[0]
        for j in range(1, len(poly)):
            ans_poly += poly[j] * i ** j
        ans_real = coeffs[0]
        for j in range(1, n):
            ans_real += coeffs[j] * i ** j
        seq.append(ans_real)
        #print(i, ans_poly, ans_real, skip_count)
        if ans_poly * res[1] != ans_real * denom:
            #print("hi")
            denom2 = lcm(denom, res[1])
            res = [res[0] * (denom2 // res[1]) +\
                    (skip_count + 1) * ans_poly * (denom2 // denom), denom2]
            #print(res)
            g = gcd(*res)
            res = [x // g for x in res]
            skip_count = 0
        else: skip_count += 1
        i += 1
    #print(seq)
    res2 = res[0] if res[1] == 1 else res[0] / res[1]
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res2

# Problem 102
def loadTriangles(doc: str) -> List[Tuple[Tuple[int]]]:
    """
    Loads the coordinates in the plane of the vertices of a
    sequence of triangles from the .txt file at relative or
    absolute location doc.
    In this .txt file, the set of coordinates representing each
    triangle are separated by a line break ('\\n') and each set
    of coordinates is a list of 6 numbers, with each consecutive
    pair representing the 2-dimensional Cartesian coordinates of
    each vertex of the triangle.
    
    Args:
        Required positional:
        doc (str): The relative or absolute path to the .txt
                file containing the coordinates of the vertices
                of the triangles.
    
    Returns:
    A list of 3-tuples of 2-tuples of ints. Each element of the list
    represents one of the triangles in the .txt file at location
    doc in the same order as they appear in that file. Each entry
    of the list is a 3-tuple, with the entries of this 3-tuple being
    the 2-dimensional Cartesian coordinates (as a 2-tuple of ints)
    of the vertices of the triangle.
    """
    with open(doc) as f:
        txt = f.read()
    res = []
    for s in txt.split("\n"):
        if not s: continue
        nums = [int(x.strip()) for x in s.split(",")]
        triangle = []
        for i in range(0, len(nums), 2):
            triangle.append((nums[i], nums[i + 1]))
        res.append(tuple(triangle))
    return res

def lineEquation(v1: Tuple[int], v2: Tuple[int]) -> Tuple[int]:
    """
    Given Cartesian coordinates of two distinct points on the plane v1
    and v2 with integer coordinates, finds the equation for the unique
    straight line that passes through both of these points in the form:
        a * x + b * y = c
    where x and y are the Cartesian coordinates and a, b and c are
    constants such that gcd(a, b) = 1 and a > 0 or a = 0 and b > 0.
    
    Args:
        Required positional:
        v1 (2-tuple of ints): The Cartesian coordinates in the form
                (x, y) of one of the points the line must pass through
        v2 (2-tuple of ints): The Cartesian coordinates in the form
                (x, y) the other points the line must pass through.
                Note that v2 must be different to v1 in at least one
                of its entries.
    
    Returns:
    3-tuple of ints representing the equation found for the straight
    line found, in the form:
        (a, b, c)
    where the equation for the line is given by:
        a * x + b * y = c
    """
    dx, dy = [x1 - x2 for x1, x2 in zip(v1, v2)]
    if not dx and not dy:
        raise ValueError("v1 and v2 must be different")
    if (dy, dx) < (0, 0): dx, dy = -dx, -dy
    return (dy, -dx, dy * v1[0] - dx * v1[1])

def lineAxisIntersectionSign(line_eqn: Tuple[int], axis: int=0)\
        -> Optional[int]:
    """
    For a straight line represented by line_eqn, such that if
    line_eqn = (a, b, c) then the line has the equation:
        a * x + b * y = c
    where (x, y) are Cartesian coordinates of the plane, finds
    whether the point of intersection with the given axis (where
    axis=0 represents the x-axis and axis=1 the y-axis) exists,
    and if so whether the value of that axis at the point of
    intersection is positive, negative or zero.
    
    Args:
         Required positional:
         line_eqn (3-tuple of ints): The representation of the
                equation of the line in the plane to be considered,
                where for line_eqn = (a, b, c) the line it
                represents has the equation:
                    a * x + b * y = c
                where (x, y) are Cartesian coordinates of the plane
         
         Optional named:
         axis (int): Either 0 or 1, with 0 representing that the
                intersection with the x-axis is to be considered
                and 1 representing that the intersection with the
                y-axis is to be considered.
    
    Returns:
    Integer (int) between -1 and 1 inclusive or None. If the line
    is parallel to the chosen axis (meaning either the line never
    meets the axis or the line coincides with the axis along its
    whole length) then returns None. Otherwise, returns -1 if the
    point of intersection is on the negative portion of the chosen
    axis, 1 if the point of intersection is on the positive portion
    of the chosen axis and 0 if it passes through exactly 0 on the
    chosen axis (i.e. the line passes through the origin of the
    Cartesian coordinate system used).
    """
    if not line_eqn[axis]: return None
    if not line_eqn[-1]: return 0
    return -1 if (line_eqn[axis] < 0) ^ (line_eqn[-1] < 0) else 1
    
def crossProduct2D(vec1: Tuple[int], vec2: Tuple[int]) -> int:
    """
    Finds the value of the cross product of two vectors in two
    dimensions, where the vectors are represented in terms of a
    right-handed orthonormal basis (e.g. Cartesian coordinates).
    The cross product of two vectors v1 and v2 in two dimensions is
    the scalar value:
        vec1 x vec2 = (length of v1) * (length of v2) *\
                            sin(angle between v1 and v2)
    where the angle from v1 to v2 (i.e. the angle vector v1 needs to be
    turned by in order to be made parallel to vector v2) is positive
    if it is an anti-clockwise turn and negative if it is a clockwise
    turn. Note that this is antisymmetric so:
       vec2 x vec1 = -vec1 x vec2.
    
    Args:
        vec1 (2-tuple of ints): The representation of the vector
                appearing first in the cross product in terms of the
                basis vectors (i.e. for basis vectors i and j, the
                vector represented is vec1[0] * i + vec1[1] * j.
                In this case, an right-handed orthonormal basis,
                i and j are unit vectors orthogonal to each other,
                and if i is turned pi/2 radians in an anticlockwise
                direction it will be parallel to j. In terms of
                Cartesian coordinates, here i is a unit vector parallel
                with the x-axis and pointing in the direction of
                increasing x, while j is a unit vector parallel with
                the y-axis and pointing in the direction of increasing
                y).
        vec2 (2-tuple of ints): The representation of the vector
                appearing second in the cross product in terms of the
                basis vectors, similarly to vec1.
    
    Returns:
    Integer (int) giving the value of the cross product of the vector
    represented by vec1 with the vector represented by vec2.
    """
    return vec1[0] * vec2[1] - vec1[1] * vec2[0]

def triangleContainsPoint(p: Tuple[int],\
        triangle_vertices: Tuple[Tuple[int]],\
        include_surface: bool=False) -> bool:
    """
    Using the 2-dimensional cross product, finds whether the point
    with 2-dimensional Cartesian coordinates p falls inside a triangle
    whose vertices are at Cartesian coordinates given by
    triangle_vertices.
    Points falling exactly on the edges and vertices of the triangle
    are considered as being inside the triangle if and only if
    include_surface is given as True.
    
    Args:
        Required positional:
        p (2-tuple of ints): The 2-dimensional Cartesian coordinate
                of the point in the plane under consideration.
        triangle_vertices (3-tuple of 2-tuples of ints): The
                2-dimensional Cartesian coordinates of the vertices
                of the triangle being considered.
        
        Optional named:
        include_surface (bool): Whether of not points falling
                exactly on the edges or vertices of the triangle
                are considered to be inside the triangle.
    
    Returns:
    Boolean (bool) giving whether the point with 2-dimensional
    Cartesian coordinates p is inside the triangle whose vertices
    have the 2-dimensional Cartesian coordinates triangle_vertices
    (where points falling exactly on the edges and vertices of the
    triangle are considered as being inside the triangle if and only
    if include_surface was given as True).
    
    Outline of rationale:
    If we arbitrarily label the vertices v1, v2 and v3. Consider the
    set of values given by the 2d cross product of the vector from
    the chosen point to v1 with the vector from v1 to v2, the 2d cross
    product of the vector from the chosen point to v2 with the vector
    from v2 to v3 and the 2d cross product of the vector from the
    chosen point to v3 with the vector from v3 to v1.
    
    It can be shown that if all values in this set are either all
    positive or all negative, then the point is strictly inside
    the triangle, while if there are both positive and negative
    values in this set then the point is strictly outside the
    triangle. For the remaining possible cases, the point is
    exactly on a vertex or edges- if two of the values are 0
    then the point is on a vertex, while if one value is 0 while
    the other two values are either both positive or both negative
    then the point is exactly on one of the edges.
    """
    res = set()
    v2 = triangle_vertices[2]
    for i in range(3):
        v1 = v2
        v2 = triangle_vertices[i]
        ans = crossProduct2D(tuple(x - y for x, y in zip(v1, p)),\
                tuple(x - y for x, y in zip(v2, v1)))
        if not ans:
            if not include_surface: return False
            continue
        res.add(ans > 0)
        if len(res) > 1: return False
    return True

"""
    Detail:
    Consider the vector from the point at p to some point on one of the
    surface of the triangle (i.e. on one of the edges or vertices of
    the triangle), and consider how the direction of this vector
    changes as we move p around one complete circuit of the surface of
    the triangle. If the point represented by p is strictly inside the
    triangle (i.e. within the triangle and not on the surface), then
    during this process the vector is always turning in the same
    direction, either clockwise or anticlockwise and turns through
    exactly 2 * pi radians. On the other hand if the point represented
    by p is strictly outside the triangle then the vector first turns
    in one direction, then the other, and the net angle it turns
    through is exactly 0 radians. We consider the case where the point
    is on the surface separately.
    
    We now consider how the vector turns when traversing an edge, from
    one vertex of the triangle to another. Either the vector turns
    only in one direction, or (as is the case when the edge and the
    vector are parallel, and recalling that we are not yet considering
    the case when the point is on one of the edges) not at all. In
    either case, the direction of turn does not change. Consequently,
    the direction the vector turns can only change at one of the
    vertices, when transitioning from one edge to the next.
    
    With this in mind, as long as the point is not on the surface of
    the triangle, if we consider turning the vector to point from
    one vertex to the next moving round the triangle in one direction
    and if we always choose to turn the vector in the direction that
    results in the vector turning the least (i.e. choosing the
    angle that is strictly less than pi / 2 noting that given that
    the point is not on an edge, a turn of pi / 2 cannot occur),
    if the point is inside the triangle then the angle will always
    turn in the same direction, while if the point is outside the
    triangle the direction will sometimes be in one direction and
    sometimes in the opposite direction (and in some cases may
    not turn at all).
    
    This can be quantified by using the 2-dimensional cross product.
    If we take the cross product of the vector from the point to
    the current vertex with the vector from that vertex to the
    next vertex in the traversal, then a positive result signifies
    that between these two vertices, the vector turns anti-clockwise,
    a negative result that the vector turns clockwise and a zero
    result that the vector does not turn.
    
    Thus, if two of the cross products between the vector from
    the point to a vertex and the vector from that vertex to
    the next vertex in that traversal have a different sign,
    this signifies that the point is outside the triangle.
    
    Conversely, if these cross products are all positive or
    all negative then the point is inside the triangle.
    
    We now consider what happens when the point is on the
    surface of the triangle. If it is at a vertex, then since
    the vector from the point to that vertex is length 0, the
    cross product of that vector with any other vector is also
    zero. Additionally, the cross product of the vector to
    the preceding vertex in the traversal is exactly the
    negative of the vector from the preceding vertex to this
    vertex, so the cross product for the preceding vertex
    is also 0. This leaves only one potentially non-zero
    cross product (between the other two vertices). In fact,
    since the corresponding edge cannot be parallel to
    either of the other two in order for the vertices to
    be considered a triangle, this cross product must be non-zero.
    
    TODO- revise
    
    
    If we label the vertices v1, v2 and v3 and consider the vectors
    vec1 from v2 to v1, vec2 from v2 to v3 and vec3 from v3 to v1,
    
"""

def triangleContainsOrigin(v1: Tuple[int], v2: Tuple[int], v3: Tuple[int]) -> bool:
    """
    Finds whether the point at the origin (0, 0) of a given Cartesian
    coordinate system falls inside a triangle whose vertices are at
    Cartesian coordinates given by v1, v2 and v3.
    Points falling exactly on the edges and vertices of the triangle
    are considered as being inside the triangle.
    
    Args:
        Required positional:
        v1 (2-tuple of ints): The 2-dimensional Cartesian coordinates
                of one of the vertices of the triangle being
                considered.
        v2 (2-tuple of ints): The 2-dimensional Cartesian coordinates
                of another of the vertices of the triangle being
                considered.
        v3 (2-tuple of ints): The 2-dimensional Cartesian coordinates
                of the final vertex of the triangle being considered.
    
    Returns:
    Boolean (bool) giving whether the origin of the 2-dimensional
    Cartesian coordinate system used is inside the triangle whose
    vertices have the 2-dimensional Cartesian coordinates v1, v2
    and v3 (where points falling exactly on the edges and vertices of
    the triangle are considered as being inside the triangle).
    
    Not currently used- superceded by triangleContainsPoint()
    """
    if v1 == (0, 0) or v2 == (0, 0) or v3 == (0, 0):
        return True
    if (v1[0] > 0 and v2[0] > 0 and v3[0] > 0) or\
            (v1[0] < 0 and v2[0] < 0 and v3[0] < 0) or\
            (v1[1] > 0 and v2[1] > 0 and v3[1] > 0) or\
            (v1[1] < 0 and v2[1] < 0 and v3[1] < 0):
        return False
    
    intercept_sgns = [set(), set()]
    
    for pair in ((v1, v2), (v2, v3), (v3, v1)):
        u1, u2 = pair
        #print(pair)
        #print([x * y > 0 for x, y in zip(*pair)])
        if all(x * y > 0 for x, y in zip(*pair)):
            continue
        eqn = lineEquation(*pair)
        #print(pair)
        #print(eqn)
        for i in range(2):
            if pair[0][~i] * pair[1][~i] > 0: continue
            ans = lineAxisIntersectionSign(eqn, axis=i)
            #print(pair, i, ans)
            if ans is not None:
                intercept_sgns[i].add(ans)
    #print(intercept_sgns)
    return all(len(x) > 1 for x in intercept_sgns)

def triangleContainment(p: Tuple[int]=(0, 0), doc: str="0102_triangles.txt",\
        include_surface: bool=True):
    """
    Solution to Project Euler #102
    Given the list of triangles represented by the 2-dimensional
    Cartesian coordinates of their vertices in the .txt file at
    location doc (see loadTriangles()), counts how many of these
    triangles contain the point with 2-dimensional Cartesian
    coordinates p.
    
    Args:
        Optional named:
        p (2-tuple of ints): The 2-dimensional Cartesian coordinates
                of the point in the plane of interest.
            Default: (0, 0)
        doc (str): The relative or absolute location of the .txt file
                containing the coordinates of the vertices of the
                triangles.
            Default: "0102_triangles.txt"
        include_surface (bool): If True, considers points that are
                exactly on the edge or on a vertex of a given triangle
                as being inside that triangle, otherwise considers
                these points to be outside the triangle.
    
    Returns:
    Integer (int) giving the number of triangles from the .txt file
    at location doc contain the point with Cartesian coordinates
    p, subject to the specified classification of points falling
    exactly on an edge or vertex of a triangle.
    """
    since = time.time()
    triangles = loadTriangles(doc)
    res = sum(triangleContainsPoint(p, x,\
            include_surface=include_surface) for x in triangles)
    #res = sum(triangleContainsOrigin(*x) for x in triangles)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 103
def isSpecialSumSet(nums: Tuple[int], nums_sorted: bool=False) -> bool:
    if not nums_sorted: nums = sorted(nums)
    n = len(nums)
    curr = [0, 0]
    for i in range(-((-n) >> 1)):
        curr[0] += nums[i]
        if curr[0] <= curr[1]: return False
        curr[1] += nums[~i]
    seen = set()
    for bm in range(1, 1 << n):
        cnt = 0
        sm = 0
        for i in range(n):
            if bm & 1:
                cnt += 1
                sm += nums[i]
            bm >>= 1
        if sm in seen:
            return False
        seen.add(sm)
    return True

def findOptimalSpecialSumSet(n: int=7) -> Tuple[int]:
    """
    
    
    Outline of rationale:
    We can simplify the requirements by making the following
    observations:
     1) Two non-empty disjoint subsets of A exist with equal sum
        if and only if two unequal subsets of A exist with equal sum
     2) Two non-empty disjoint subsets of A exist such that one of
        the subsets has more elements than the other but a sum not
        exceeding that of the other, if and only if two unequal
        subsets of A exist such that one of the subsets has more
        elements than the other but a sum not exceeding that of the
        other.
     3) Two unequal subsets of A exist such that one of the
        subsets has more elements than the other but a sum not
        exceeding that of the other if and only if there exists an
        integer m such that 1 <= m <= n / 2 (where n is the
        number of elements in A) and:
            (sum of smallest m + 1 elements of A) <=
                                (sum of largest m elements of A)
    TODO
    
    
    We use a backtracking algorithm. We prune the search space
    by noting that the sum of the two smallest elements of the
    set must be strictly greater than the largest element. For
    a given sum of the two smallest elements, this makes the number
    of cases to check finite.
    
    We search for increasing sum of the two smallest elements until
    we have found a candidate. This allows us to further reduce the
    search space until we reach a sum of two smallest elements such
    that the smallest possible sum of a set with this sum of the
    smallest two elements is at least as large as the current best sum,
    at which point we can conclude that the current best candidate
    is an optimal special sum set, and so return this.
    """
    if n == 1: return (1,)
    elif n == 2: return (1, 2)
    since = time.time()
    curr_best = float("inf")
    curr = [0] * n
    curr_sums = [0, 0]
    
    def recur(sum_set: Set[int], i1: int=2, i2: int=n - 2)\
            -> Generator[Tuple[int], None, None]:
        #if i1 == 2:
        #    print(curr, curr_sums)
        #print(i1, i2, curr, sum_set, i1 + n - 1 - i2, len(sum_set))
        tot_sum = sum(curr_sums)
        if tot_sum >= curr_best: return
        if i1 > i2:
            yield tuple(curr)
            return
        elif i1 == i2:
            for num in range(curr[i2 - 1] + 1, min(curr[i1 + 1], curr_best - tot_sum)):
                if num in sum_set: continue
                for x in sum_set:
                    if x + num in sum_set:
                        break
                else:
                    curr[i1] = num
                    curr_sums[0] += num
                    yield tuple(curr)
                    curr_sums[0] -= num
                #if isSpecialSumSet(curr):
                #    curr_sums[0] += num
                #    yield tuple(curr)
                #    curr_sums[0] -= num
            return
        n_remain = i2 - i1 + 1
        lb = ((curr[i1 - 1] + n_remain) * (curr[i1 - 1] + n_remain + 1) -\
                curr[i1 - 1] * (curr[i1 - 1] + 1)) >> 1
        
        lb2 = lb - curr[i1 - 1] - n_remain
        #if i1 == 2 and curr[0] == 11 and curr[1] == 18:
        #    print(i1, i2, curr, curr_sums, curr_best)
        #    print(lb, lb2, max(curr[i1 - 1], curr_sums[1] - curr_sums[0]) + 1,\
        #        min(curr[i1 - 1] + (curr_best - tot_sum - lb) // n_remain + 2,\
        #        curr[i2 + 1] - n_remain + 1))
        #print(curr[i1 - 1] + (curr_best - tot_sum - lb) // n_remain + 2,\
        #        curr[i2 + 1] - n_remain + 1)
        rng_mx = curr[i2 + 1] - n_remain + 1
        if isinstance(curr_best, int):
            rng_mx = min(rng_mx,\
                    curr[i1 - 1] + (curr_best - tot_sum - lb) // n_remain + 2)
        for num1 in range(max(curr[i1 - 1], curr_sums[1] - curr_sums[0]) + 1,\
                rng_mx):
            #if (num - 2 - curr[i1 - 1]) * n_remain > curr_best - tot_sum - lb:
            if num1 in sum_set: continue
            sum_set2 = set(sum_set)
            sum_set2.add(num1)
            for x in sum_set:
                x2 = x + num1
                if x2 in sum_set2: break
                sum_set2.add(x2)
            else:
                curr[i1] = num1
                curr_sums[0] += num1
                lb2 += n_remain - 1
                for num2 in range(num1 + n_remain - 1,\
                        min(curr_sums[0] - curr_sums[1],\
                        curr[i2 + 1] + 1,\
                        curr_best - tot_sum - lb2 + 1)):
                    if num2 in sum_set2: continue
                    sum_set3 = set(sum_set2)
                    sum_set3.add(num2)
                    for x in sum_set2:
                        x2 = x + num2
                        if x2 in sum_set3: break
                        sum_set3.add(x2)
                    else:
                        curr[i2] = num2
                        curr_sums[1] += num2
                        yield from recur(sum_set=sum_set3, i1=i1 + 1, i2=i2 - 1)
                        curr_sums[1] -= num2
                curr_sums[0] -= num1
        return
        """
        for num in range(num2 + i - 1, curr[i + 1]):
            curr[i] = num
            curr_sm[0] += num
            if curr_sm[0] + (((curr[1] + i - 3) * (curr[1] + i - 2)\
                    - curr[1] * (curr[1] + 1)) >> 1) >= curr_best:
                curr_sm[0] -= num
                break
            yield from recur(i=i - 1)
            curr_sm[0] -= num
        
        return
        """
    
    res = ()
    pair1_sum = ((n - 1) << 1) + 1
    while True:
        print(f"pair 1 sum = {pair1_sum}")
        looped = False
        curr_sums = [pair1_sum, 0]
        for num1 in reversed(range(n - 1, -((-pair1_sum) >> 1))):
            num2 = pair1_sum - num1
            #print(pair1_sum, num1, num2)
            lb = num1 + (((num2 + n - 1) * (num2 + n - 2) - num2 * (num2 - 1)) >> 1)
            if lb >= curr_best: break
            looped = True
            curr[0], curr[1] = num1, num2
            for num_mx in range(num2 + n - 2, pair1_sum):
                curr[-1] = num_mx
                curr_sums[1] = num_mx
                sum_set = {num1, num2, num_mx, num1 + num2,\
                        num1 + num_mx, num2 + num_mx, num1 + num2 + num_mx}
                for res in recur(sum_set=sum_set,i1=2, i2=n - 2):
                    curr_best = sum(curr_sums)
                    print(res, curr_best)
        if not looped: break
        pair1_sum += 1
    print(f"Time taken = {time.time() - since:.4f} seconds")
    print(res, sum(res))
    return res

# Problem 104
def isPandigital(num: int, base: int=10, chk_rng: bool=True) -> bool:
    if chk_rng and not base ** (base - 2) <= num < base ** (base - 1):
        return False
    dig_set = set()
    while num:
        num, r = divmod(num, base)
        if not r or r in dig_set: return False
        dig_set.add(r)
    return True

def startAndEndPandigital(num: int, base: int=10, target: Optional[int]=None,\
        md: Optional[int]=None) -> bool:
    if target is None: target = base ** (base - 2)
    if num < target: return False
    if md is None: md = target * base
    if not isPandigital(num % md): return False
    while num > md: num //= base
    return isPandigital(num)

def FibonacciFirstKDigits(i: int, k: int, base: int=10) -> int:
    """
    
    Using Binet's formula
    """
    phi = (1 + math.sqrt(5)) / 2 # The golden ratio
    lg_rt5 = math.log(math.sqrt(5), base)
    lg_num = i * math.log(phi, base) - lg_rt5
    if lg_num > k + 1:
        # Ensuring no rounding error
        lg_num2 = (lg_num % 1) + (k - 1)
        diff = round(lg_num - lg_num2)
        cnt = 0
        div = 1
        while cnt < diff:
            res = math.floor(base ** lg_num2)
            if res % base != base - 1:
                return res // div
            #print("hi")
            #print(res)
            div *= base
            cnt += 1
            lg_num2 += 1
    # Calculating exactly
    psi = (1 - math.sqrt(5)) / 2
    res = (phi ** i - psi ** i) / math.sqrt(5)
    #print(res)
    res = round(res)
    mx = base ** k
    while res > mx:
        res //= base
    return res
    

def pandigitalFibonacciStart(i: int, base: int=10) -> bool:
    """
    
    Using Binet's formula
    """
    return isPandigital(FibonacciFirstKDigits(i, k=base - 1, base=10))
    
    
    if target is None: target = base ** (base - 2)
    lg_num = i * math.log((1 + math.sqrt(5)) / 2, base)
    #if lg_num < base - 1: continue
    #num = 10 ** 
    num = round(((1 + math.sqrt(5)) / 2) ** i / math.sqrt(5))
    if num < target: return False
    if md is None: md = target * base
    while num > md: num //= base
    print(i, num)
    res = isPandigital(num, chk_rng=False)
    print(res)
    return res

def pandigitalFibonacciEnds(base: int=10) -> int:
    since = time.time()
    if base == 2: return 1
    curr = [1, 1]
    i = 2
    target = base ** (base - 2)
    md = base ** (base - 1)
    while curr[1] < target:
        curr = [curr[1], sum(curr)]
        i += 1
        #if not i % 1000:
        #    print(i)
    curr[1] %= md
    #not startAndEndPandigital(curr[1]):
    while not isPandigital(curr[1], chk_rng=False) or\
            not pandigitalFibonacciStart(i, base=base):
        #if isPandigital(curr[1]) and pandigitalFibonacciStart(i, base=base):
        #    return i
        curr = [curr[1], sum(curr) % md]
        i += 1
        #if not i % 1000:
        #    print(i)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return i

# Problem 105
def loadSets(doc: str) -> List[Tuple[int]]:
    with open(doc) as f:
        txt = f.read()
    return [tuple(int(y.strip()) for y in x.split(",")) for x in txt.split("\n")]

def specialSubsetSumsTesting(doc: str="0105_sets.txt") -> int:
    """
    Solution to Project Euler #105
    """
    since = time.time()
    sp_sets = loadSets(doc)
    #print(sp_sets)
    res = sum(sum(x) for x in sp_sets if isSpecialSumSet(x, nums_sorted=False))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 106
def specialSubsetSumsComparisons(n: int=12) -> int:
    """
    Solution to Project Euler #106
    """
    res = 0
    for i in range(2, (n >> 1) + 1):
        # Using Catalan numbers
        res += (sp.comb(n, 2 * i, exact=True) * sp.comb(2 * i, i, exact=True) *\
                (i - 1)) // (2 * (i + 1))
    return res

# Problem 107
class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.root = list(range(n))
        self.rank = [1] * n
    
    def find(self, v: int) -> int:
        r = self.root[v]
        if r == v: return v
        res = self.find(r)
        self.root[v] = res
        return res
    
    def union(self, v1: int, v2: int) -> None:
        r1, r2 = list(map(self.find, (v1, v2)))
        if r1 == r2: return
        d = self.rank[r1] - self.rank[r2]
        if d < 0: r1, r2 = r2, r1
        elif not d: self.rank[r1] += 1
        self.root[r2] = r1
        return
    
    def connected(self, v1: int, v2: int) -> bool:
        return self.find(v1) == self.find(v2)

def loadNetwork(doc: str) -> Tuple[Union[int, List[Tuple[int]]]]:
    with open(doc) as f:
        txt = f.read()
    res = []
    arr = txt.split("\n")
    n = len(arr)
    for i1, row in enumerate(arr):
        if not row: continue
        row2 = row.split(",")
        for i2 in range(i1):
            v = row2[i2].strip()
            #print(v)
            if v == "-": continue
            res.append((i1, i2, int(v)))
    return n, res

def KruskallAlgorithm(n: int, edges: List[Tuple[int]]):
    edges = sorted(edges, key=lambda x: x[2])
    res = []
    uf = UnionFind(n)
    for e in edges:
        if uf.connected(e[0], e[1]):
            continue
        uf.union(e[0], e[1])
        res.append(e)
    return res

def minimalNetwork(doc: str="0107_network.txt"):
    """
    Solution to Project Euler #107
    """
    n, edges = loadNetwork(doc)
    mst_edges = KruskallAlgorithm(n, edges)
    return sum(x[2] for x in edges) - sum(x[2] for x in mst_edges)

# Problem 108 & 110
def diophantineReciprocals(min_n_solutions: int=1001) -> int:
    """
    Solution to Project Euler #108 and Project Euler #110 (with
    min_n_solutions=4 * 10 ** 6 + 1)
    """
    since = time.time()
    n_p = 0
    num = 1
    while num < min_n_solutions:
        num *= 3
        n_p += 1
    p_lst = []
    ps = PrimeSPFsieve()
    p_gen = ps.endlessPrimeGenerator()
    for i in range(n_p):
        p_lst.append(next(p_gen))
    curr_best = [float("inf")]
    target = (min_n_solutions << 1) - 1
    #print(p_lst)
    def recur(i: int=0, curr_num: int=1, curr_n_solutions: int=1,\
            mx_count: Union[int, float]=float("inf")) -> None:
        #print(i, curr_num, curr_n_solutions)
        if curr_num >= curr_best[0] or i >= len(p_lst): return
        if curr_n_solutions >= target:
            curr_best[0] = curr_num
            #print(i, curr_num, curr_n_solutions)
            #print(f"curr_best = {curr_best[0]}")
            return
        for j in range(1, min(mx_count, (-((-target) // curr_n_solutions) - 1) >> 1) + 1):
            curr_num *= p_lst[i]
            if curr_num >= curr_best[0]: break
            recur(i + 1, curr_num=curr_num,\
                    curr_n_solutions=curr_n_solutions * ((j << 1) + 1),\
                    mx_count=j)
        return
    recur()
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return curr_best[0]
            

# Problem 109
def dartCheckouts(mx_score: int=99) -> int:
    """
    Solution to Project Euler #109
    For a standard dart board, calculates the sum of the number
    of ways to check out without missing over all scores no greater
    than mx_score where two ways of checking out are distinct
    if and only if the sets of regions hit in the three darts are
    different or the final double is different.
    
    Args:
        Optional named:
        mx_score (int): The largest checkout score included in
                the sum.
            Default: 99
    
    Returns:
    Integer (int) giving the sum of the number of ways to check
    out without missing over all scores no greater than mx_score
    subject to the definition of distinct checkouts given above.
    """
    since = time.time()
    double_dict = SortedDict()
    for num in range(1, min(20, mx_score >> 1) + 1):
        double_dict[num << 1] = 1
    if 50 <= mx_score: double_dict[50] = 1
    
    score_dict = SortedDict({0: 1})
    for num in range(1, min(20, mx_score) + 1):
        score_dict[num] = score_dict.get(num, 0) + 1
        dbl = num << 1
        if dbl > mx_score: continue
        score_dict[dbl] = score_dict.get(dbl, 0) + 1
        trpl = num * 3
        if trpl > mx_score: continue
        score_dict[trpl] = score_dict.get(trpl, 0) + 1
    if 25 <= mx_score:
        score_dict[25] = score_dict.get(25, 0) + 1
        if 50 <= mx_score:
            score_dict[50] = score_dict.get(50, 0) + 1
    
    prev = SortedDict(score_dict)
    curr = SortedDict()
    for num1, f1 in prev.items():
        for num2, f2 in score_dict.items():
            sm = num1 + num2
            if sm > mx_score: break
            elif num2 == num1:
                curr[sm] = curr.get(sm, 0) + ((f1 * (f1 + 1)) >> 1)
                break
            curr[sm] = curr.get(sm, 0) + f1 * f2
    prev = curr
    curr = SortedDict()
    for num1, f1 in prev.items():
        for num2, f2 in double_dict.items():
            sm = num1 + num2
            if sm > mx_score: break
            curr[sm] = curr.get(sm, 0) + f1 * f2
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return sum(curr.values())
    
# Problem 110- see Problem 108


# Problem 111
def permutationsWithRepeats(objs: List[Any], freqs: List[int])\
        -> Generator[Tuple[Any], None, None]:
    n = len(objs)
    m = sum(freqs)
    remain = set(range(n))
    curr = [None] * m
    def recur(i: int=0) -> Generator[Tuple[Any], None, None]:
        if i == m:
            yield tuple(curr)
            return
        for j in list(remain):
            freqs[j] -= 1
            if not freqs[j]: remain.remove(j)
            curr[i] = objs[j]
            yield from recur(i + 1)
            if not freqs[j]: remain.add(j)
            freqs[j] += 1
        return
    
    yield from recur(i=0)
    return

def digitCountIntegerGenerator(n_dig: int, rpt_dig: int, n_rpt: int,\
        base: int=10) -> Generator[int, None, None]:
    if n_dig < n_rpt: return
    digs = [x for x in range(base) if x != rpt_dig]
    objs = [rpt_dig]
    freqs = [n_rpt]
    
    def recur(i: int, n_remain: int) -> Generator[int, None, None]:
        if not n_remain or i == base - 2:
            if n_remain:
                objs.append(digs[-1])
                freqs.append(n_remain)
            #print(objs, freqs)
            for dig_tup in permutationsWithRepeats(objs, freqs):
                #print(dig_tup)
                if not dig_tup[0]: continue
                ans = 0
                for d in dig_tup:
                    ans = ans * base + d
                yield ans
            if n_remain:
                objs.pop()
                freqs.pop()
            return
        yield from recur(i + 1, n_remain)
        objs.append(digs[i])
        freqs.append(0)
        for j in range(1, n_remain + 1):
            freqs[-1] += 1
            yield from recur(i + 1, n_remain - j)
        freqs.pop()
        objs.pop()
        return
    
    yield from recur(0, n_dig - n_rpt)
    return

def mostRepeatDigitPrimes(n_dig: int, rpt_dig: int, base: int=10,\
        ps: Optional[PrimeSPFsieve]=None)\
        -> Tuple[Union[List[int], int]]:
    
    if ps is None: ps = PrimeSPFsieve()
    ps.extendSieve(isqrt(base ** n_dig))
    for n_rpt in reversed(range(n_dig + 1)):
        p_lst = []
        for num in digitCountIntegerGenerator(n_dig, rpt_dig,\
                n_rpt, base=10):
            if ps.isPrime(num):
                p_lst.append(num)
        if p_lst: break
    return (p_lst, n_rpt)

def primesWithRuns(n_dig: int=10, base: int=10,\
        ps: Optional[PrimeSPFsieve]=None) -> int:
    since = time.time()
    if ps is None: ps = PrimeSPFsieve()
    res = sum(sum(mostRepeatDigitPrimes(n_dig, d, base=10, ps=ps)[0])\
            for d in range(base))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
        

# Problem 112
def isBouncy(num: int, base: int=10) -> bool:
    """
    Assesses whether positive integer num when expressed in the
    given base is bouncy.
    A positive integer is bouncy when expressed in the given
    base if and only if the sequence of digits of the expression
    of that integer in the given base is not weakly increasing
    or weakly decreasing.
    """
    incr = True
    decr = True
    num, curr = divmod(num, base)
    while num:
        prev = curr
        num, curr = divmod(num, base)
        if curr < prev:
            if not decr: return True
            incr = False
        elif curr > prev:
            if not incr: return True
            decr = False
    return False

def bouncyProportions(prop_numer: int=99, prop_denom: int=100) -> int:
    since = time.time()
    bouncy_cnt = 0
    g = gcd(prop_numer, prop_denom)
    prop_numer //= g
    prop_denom //= g
    rng = (1, prop_denom + 1)
    while True:
        bouncy_cnt += sum(isBouncy(x) for x in range(*rng))
        if bouncy_cnt * prop_denom == (rng[1] - 1) * prop_numer:
            break
        rng = (rng[1], rng[1] + prop_denom)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return rng[1] - 1

# Problem 113
class NonBouncyCounter:
    def __init__(self, n_dig: int=1, base: int=10):
        self.base = base
        self.memo = [[[0, 0] for _ in range(base)],\
                [[1, 1] for _ in range(base)]]
        self.extendMemo(n_dig)
    
    def extendMemo(self, n_dig: int) -> None:
        for i in range(len(self.memo), n_dig + 1):
            self.memo.append([[0, 0] for _ in range(self.base)])
            curr = 0
            for j, pair in enumerate(self.memo[i - 1]):
                curr += pair[0]
                self.memo[i][j][0] = curr
            curr = 0
            for j in reversed(range(len(self.memo[i - 1]))):
                curr += self.memo[i - 1][j][1]
                self.memo[i][j][1] = curr
        return
    
    def __call__(self, n_dig: int) -> int:
        if not n_dig: return 0
        self.extendMemo(n_dig)
        # Subtract self.base - 1 since numbers with all one digit
        # are double counted
        return sum(sum(x) for x in self.memo[n_dig][1:]) - (self.base - 1)

def nonBouncyNumbers(mx_n_dig: int=100, base: int=10) -> int:
    """
    Solution to Project Euler #113
    """
    since = time.time()
    nbc = NonBouncyCounter(n_dig=mx_n_dig, base=base)
    res = sum(nbc(i) for i in range(1, mx_n_dig + 1))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 114
def countingBlockCombinations(tot_len: int=50, min_large_len: int=3) -> int:
    """
    Solution to Project Euler #114
    """
    since = time.time()
    qu = deque([1] * (min_large_len + 1))
    tot = 0
    for _ in range(min_large_len + 1, tot_len + 2):
        tot += qu.popleft()
        qu.append(qu[-1] + tot)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return qu[-1]

# Problem 115
def countingBlockCombinationsII(min_large_len: int=50, target_count: int=10 ** 6 + 1) -> int:
    """
    Solution to Project Euler #114
    """
    since = time.time()
    if target_count <= 1: return 0
    qu = deque([1] * (min_large_len + 1))
    n = min_large_len - 1
    tot = 0
    while qu[-1] < target_count:
        tot += qu.popleft()
        qu.append(qu[-1] + tot)
        n += 1
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return n

# Problem 116
def redGreenOrBlueTiles(tot_len: int=50, min_large_len: int=2,\
        max_large_len: int=4) -> int:
    """
    Solution to Project Euler #116
    """
    since = time.time()
    res = 0
    for large_len in range(min_large_len, max_large_len + 1):
        qu = deque([1] * (large_len))
        for _ in range(large_len, tot_len + 1):
            qu.append(qu[-1] + qu.popleft())
        res += qu[-1]
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res - (max_large_len - min_large_len + 1)

# Problem 117
def redGreenAndBlueTiles(tot_len: int=50, min_large_len: int=2,\
        max_large_len: int=4) -> int:
    """
    Solution to Project Euler #117
    """
    since = time.time()
    qu1 = deque([1] * (min_large_len))
    tot = 0
    qu2 = deque()
    for _ in range(min_large_len, max_large_len + 1):
        qu2.append(qu1.popleft())
        tot += qu2[-1]
        qu1.append(qu1[-1] + tot)
    for _ in range(max_large_len + 1, tot_len + 1):
        qu2.append(qu1.popleft())
        tot += qu2[-1]
        tot -= qu2.popleft()
        qu1.append(qu1[-1] + tot)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return qu1[-1]

# Problem 118- try to make faster
def pandigitalPrimeSets(base: int=10) -> int:
    
    since = time.time()
    ps = PrimeSPFsieve(isqrt(base ** base))
    res = [0]
    def recur(nums: Tuple[int], i: int=0, prev: int=0, prev_n_dig: int=0) -> None:
        n = len(nums)
        #print(i, prev, prev_n_dig)
        num = 0
        for i2 in range(i, i + prev_n_dig):
            num = num * base + nums[i2]
        i2 = i + prev_n_dig
        if num <= prev:
            if i2 == n: return
            num = num * base + nums[i2]
            i2 += 1
        if n - i2 >= i2 - i:
            #if i2 > i + 1 and nums[i2 - 1] not in disallowed_last and\
            #        ps.isPrime(num):
            if i2 > i and ps.isPrime(num):
                #print(num)
                recur(nums, i=i2, prev=num, prev_n_dig=i2 - i)
            i3 = i + ((n - i) >> 1)
            for i2 in range(i2, i3):
                num = num * base + nums[i2]
                #if nums[i2] not in disallowed_last and ps.isPrime(num):
                if ps.isPrime(num):
                    #print(num)
                    recur(nums, i=i2 + 1, prev=num,\
                            prev_n_dig=i2 - i + 1)
        else: i3 = i2
        for i3 in range(i3, n):
            num = num * base + nums[i3]
        #if num > prev and ps.isPrime(num): print(num)
        res[0] += (num > prev) and ps.isPrime(num)
        return
    
    disallowed_last = {x for x in range(1, base) if gcd(base, x) != 1}
    for perm in itertools.permutations(range(1, base)):
        if perm[-1] in disallowed_last: continue
        recur(perm)
        #print(perm, res)
    res[0] += all(ps.isPrime(x) for x in range(1, base))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res[0]

# Problem 119- TODO
def isPower(num: int, base: int) -> int:
    num2 = base
    res = 1
    while num2 < num:
        num2 *= base
        res += 1
    return res if num2 == num else -1

def digitCountAndDigitSum(num: int, base: int=10) -> int:
    res = [0, 0]
    while num:
        num, r = divmod(num, base)
        res[0] += 1
        res[1] += r
    return res

def powerDigitSumEqualNumDigitCountUpperBound(exp: int, base: int=10) -> int:
    
    def check(n_dig: int) -> bool:
        num = ((base - 1) * n_dig) ** exp
        return num >= base ** (n_dig - 1)

    mult = 10
    prev = 0
    curr = 1
    while check(curr):
        prev = curr
        curr *= mult
    #print(prev, curr)
    lft, rgt = prev, curr - 1
    while lft < rgt:
        mid = lft - ((lft - rgt) >> 1)
        if check(mid): lft = mid
        else: rgt = mid - 1
    return lft
    
    """
    n_dig = 0
    curr = 0
    comp = 1
    while True:
        curr += base - 1
        if curr ** exp < comp: break
        n_dig += 1
        comp *= base
    #print(n_dig)
    return n_dig
    """

def digitPowerSumSequence(n_terms: int, base: int=10) -> List[Tuple[int]]:
    
    res = []
    heap = [(2 ** 2, 2, 2)]
    n_dig_limit_heap = [(powerDigitSumEqualNumDigitCountUpperBound(2, base=base), 2)]
    mx_b = -float("inf")
    mx_a = 2
    curr_n_dig = 0
    while True:
        num, a, b = heapq.heappop(heap)
        n_dig, dig_sum = digitCountAndDigitSum(num, base=base)
        if dig_sum == a:
            res.append((num, a, b))
            #print(len(res), num, a, b)
            #print(n_dig_limit_heap[0])
            if len(res) == n_terms: break
        if n_dig > curr_n_dig:
            curr_n_dig = n_dig
            #print(n_dig_limit_heap)
            while n_dig_limit_heap and n_dig_limit_heap[0][0] < n_dig:
                heapq.heappop(n_dig_limit_heap)
                #print(f"new min = {n_dig_limit_heap[0]}")
        heapq.heappush(heap, (num * a, a, b + 1))
        if b + 1 > mx_b:
            mx_b = b + 1
            heapq.heappush(n_dig_limit_heap,\
                    (powerDigitSumEqualNumDigitCountUpperBound(mx_b, base=base), mx_b))
            #print(n_dig_limit_heap)
        if a == mx_a:
            mx_a += 1
            b2 = n_dig_limit_heap[0][1]
            heapq.heappush(heap, (mx_a ** b2, mx_a, b2))
    return res

def digitPowerSum(n: int, base: int=10) -> int:
    """
    Solution to Project Euler #119
    """
    since = time.time()
    res = digitPowerSumSequence(n, base=base)[-1][0]
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 120
def squareRemainders(a_min: int=3, a_max: int=1000) -> int:
    """
    Solution to Project Euler #120
    """
    res = 0
    for a in range(a_min, a_max + 1):
        md = a ** 2
        res += max(2 % md, (2 * ((a - 1) >> 1) * a) % md)
    return res





