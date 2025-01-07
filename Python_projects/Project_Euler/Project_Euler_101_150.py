#!/usr/bin/env python

import bisect
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

sys.path.append(os.path.join(os.path.dirname(__file__), "../Algorithms_and_Datastructures/Algorithms"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../Algorithms_and_Datastructures/Data_structures"))
from prime_sieves import PrimeSPFsieve
from addition_chains import AdditionChainCalculator

def gcd(a: int, b: int) -> int:
    """
    For non-negative integers a and b (not both zero),
    calculates the greatest common divisor of the two, i.e.
    the largest positive integer that is an exact divisor
    of both a and b.

    Args:
        Required positional:
        a (int): Non-negative integer which is the first
                which the greatest common divisor must
                divide.
        b (int): Non-negative integer which is the second
                which the greatest common divisor must
                divide. Must be non-zero if a is zero.
    
    Returns:
    Strictly positive integer giving the greatest common
    divisor of a and b.
    """
    return a if not b else gcd(b, a % b)
    
def lcm(a: int, b: int) -> int:
    """
    For non-negative integers a and b (not both zero),
    calculates the lowest common multiple of the two, i.e.
    the smallest positive integer that is a multiple
    of both a and b.

    Args:
        Required positional:
        a (int): Non-negative integer which is the first
                which must divide the lowest common multiple.
        b (int): Non-negative integer which is the second
                which must divide the lowest common multiple.
                Must be non-zero if a is zero.
    
    Returns:
    Strictly positive integer giving the lowest common
    multiple of a and b.
    """

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

def addFractions(frac1: Tuple[int, int], frac2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Finds the sum of two fractions in lowest terms (i.e. such that
    the numerator and denominator are coprime)

    Args:
        frac1 (2-tuple of ints): The first of the fractions to sum,
                in the form (numerator, denominator)
        frac2 (2-tuple of ints): The second of the fractions to sum,
                in the form (numerator, denominator)
    
    Returns:
    2-tuple of ints giving the sum of frac1 and frac2 in the form (numerator,
    denominator). If the result is negative then the numerator is negative
    and the denominator positive.
    """
    denom = lcm(abs(frac1[1]), abs(frac2[1]))
    numer = (frac1[0] * denom // frac1[1]) + (frac2[0] * denom // frac2[1])
    g = gcd(numer, denom)
    return (numer // g, denom // g)

def multiplyFractions(frac1: Tuple[int, int], frac2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Finds the product of two fractions in lowest terms (i.e. such that
    the numerator and denominator are coprime)

    Args:
        frac1 (2-tuple of ints): The first of the fractions to multiply,
                in the form (numerator, denominator)
        frac2 (2-tuple of ints): The second of the fractions to multiply,
                in the form (numerator, denominator)
    
    Returns:
    2-tuple of ints giving the product of frac1 and frac2 in the form (numerator,
    denominator). If the result is negative then the numerator is negative
    and the denominator positive.
    """
    neg = (frac1[1] < 0) ^ (frac1[1] < 0) ^ (frac2[0] < 0) ^ (frac2[1] < 0)
    frac_prov = (abs(frac1[0] * frac2[0]), abs(frac1[1] * frac2[1]))
    g = gcd(frac_prov[0], frac_prov[1])
    return (-(frac_prov[0] // g) if neg else (frac_prov[0] // g), frac_prov[1] // g)

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
            denom = lcm(denom, frac.denominator)
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
def loadTriangles(doc: str, relative_to_program_file_directory: bool=False) -> List[Tuple[Tuple[int]]]:
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
        
        Optional named:
        relative_to_program_file_directory (bool): If True then
                if doc is specified as a relative path, that
                path is relative to the directory containing
                the program file, otherwise relative to the
                current working directory.
            Default: False
    
    Returns:
    A list of 3-tuples of 2-tuples of ints. Each element of the list
    represents one of the triangles in the .txt file at location
    doc in the same order as they appear in that file. Each entry
    of the list is a 3-tuple, with the entries of this 3-tuple being
    the 2-dimensional Cartesian coordinates (as a 2-tuple of ints)
    of the vertices of the triangle.
    """
    if relative_to_program_file_directory and not doc.startswith("/"):
        doc = os.path.join(os.path.dirname(__file__), doc)
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
        relative_to_program_file_directory: bool=True, include_surface: bool=True):
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
        relative_to_program_file_directory (bool): If True then
                if doc is specified as a relative path, that
                path is relative to the directory containing
                the program file, otherwise relative to the
                current working directory.
            Default: True
        include_surface (bool): If True, considers points that are
                exactly on the edge or on a vertex of a given triangle
                as being inside that triangle, otherwise considers
                these points to be outside the triangle.
            Default: True
    
    Returns:
    Integer (int) giving the number of triangles from the .txt file
    at location doc contain the point with Cartesian coordinates
    p, subject to the specified classification of points falling
    exactly on an edge or vertex of a triangle.
    """
    since = time.time()
    triangles = loadTriangles(doc, relative_to_program_file_directory=relative_to_program_file_directory)
    res = sum(triangleContainsPoint(p, x,\
            include_surface=include_surface) for x in triangles)
    #res = sum(triangleContainsOrigin(*x) for x in triangles)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 103
def isSpecialSumSet(nums: Tuple[int], nums_sorted: bool=False) -> bool:
    """
    For a given list of integers nums, identifies whether it is
    a special sum set.

    A special sum set is a set of distinct positive integers for
    which:
     1) For any two disjoint non-empty subsets (i.e. subsets that
        contain at least one element and have no common element)
        the sums over all elements is different for the two subsets
     2) For any subset, the sum over all elements in that subset
        is strictly greater than that of any other subset that is
        disjoint with the chosen set that contains fewer elements.
    
    
    Args:
        Required positional:
        nums (tuple of ints): The set of integers to be assessed
                for whether it is a special sum set.
        
        Optional named:
        nums_sorted (bool): Whether the contents of nums has
                already been sorted.
            Default: False
    
    Returns:
    Boolean (bool) giving True if nums represents a special sum
    set and False if not.
    
    Note- the two conditions for a special sum set are equivalent
    to the same conditions with the disjoint requirement being
    replaced by a distinct requirement. This is because the
    distinct requirement encompasses the disjoint requirement, and
    in both of the conditions, if there exist two distinct non-empty
    subsets that violate that condition, then by removing the common
    elements of the two sets, we can construct disjoint sets that
    violate the condition, at least one of which must be non-empty.
    If the constructed sets are both non-empty then there exist
    disjoint non-empty sets that violate one of the conditions. On
    the other hand it is actually impossible for either of the
    constructed sets to be empty if the original sets violate one
    of the conditions, as for sets containing strictly positive
    integers an empty set always has a strictly smaller sum (0)
    than a non-empty set. Thus, this replacement of distinct for
    disjoint gives rise equivalent conditions for the special sum
    set. As this condition is easier to work with, it is used
    instead in the calculation.
    """
    n = len(nums)
    # Sorting and ensuring no repeated elements
    if not nums_sorted:
        if len(set(nums)) != n: return False
        nums = sorted(nums)
    else:
        for i in range(n - 1):
            if nums[i] == nums[i + 1]: return False
    # Checking that all elements are strictly positive
    if nums[0] < 1: return False
    # Checking that all subsets have sums strictly greater
    # than any subsets with fewer elements
    curr = [0, 0]
    for i in range(-((-n) >> 1)):
        curr[0] += nums[i]
        if curr[0] <= curr[1]: return False
        curr[1] += nums[~i]
    # Checking that there are no repeated sums
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

def findOptimalSpecialSumSets(n: int) -> List[Tuple[int]]:
    """
    Identifies every optimal special sum set with n elements.

    A special sum set is a set of distinct positive integers for
    which:
     1) For any two disjoint non-empty subsets (i.e. subsets that
        contain at least one element and have no common element)
        the sums over all elements is different for the two subsets
     2) For any subset, the sum over all elements in that subset
        is strictly greater than that of any other subset that is
        disjoint with the chosen set that contains fewer elements.
    
    An optimal special sum set for a given number of elements is
    a special sum set such that the sum of its elements is no
    greater than than of any other special sum set with the same
    number of elements.

    Args:
        Required positional:
        n (int): The number of elements for which an optimal special
                sum set is sought.
    
    Returns:
    List of n-tuples containing strictly positive integers (int),
    representing every optimal special sum set with n elements,
    each sorted in strictly increasing order. The optimal special
    sum sets are sorted in lexicographically increasing order
    over the elements from left to right.
    
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
    
    res = []
    curr_best = float("inf")
    pair1_sum = ((n - 1) << 1) + 1
    while True:
        #print(f"pair 1 sum = {pair1_sum}")
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
                for seq in recur(sum_set=sum_set,i1=2, i2=n - 2):
                    sm = sum(seq)
                    if sm > curr_best: continue
                    if sm < curr_best:
                        curr_best = sm
                        res = []
                    res.append(seq)
        if not looped: break
        pair1_sum += 1
    #print(res)
    return sorted(res)

def specialSubsetSumsOptimum(n: int=7) -> str:
    """
    Solution to Project Euler #103

    Identifies the lexicographically smallest optimum special
    subset sum for n elements (where the lexicographic sorting is
    over the elements in the set from smallest to largest). The
    result is given as the string concatenation of the numbers
    in the identified optimum special subset sum from smallest to
    largest, each expressed in base 10.

    A special sum set is a set of distinct positive integers for
    which:
     1) For any two disjoint non-empty subsets (i.e. subsets that
        contain at least one element and have no common element)
        the sums over all elements is different for the two subsets
     2) For any subset, the sum over all elements in that subset
        is strictly greater than that of any other subset that is
        disjoint with the chosen set that contains fewer elements.
    
    An optimal special sum set for a given number of elements is
    a special sum set such that the sum of its elements is no
    greater than than of any other special sum set with the same
    number of elements.

    Args:
        Optional named:
        n (int): The number of elements for which an optimal special
                sum set is sought.
            Default: 7
    
    Returns:
    String (str) containing the concatenation of the numbers in the
    lexicographically smallest optimum special subset sum for n
    elements (where the lexicographic sorting is over the elements
    in the set from smallest to largest), where the numbers are
    concatenated in order from smallest to largest.
    """
    since = time.time()
    res = findOptimalSpecialSumSets(n)[0]
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return "".join([str(x) for x in res])

# Problem 104
def isPandigital(num: int, base: int=10, chk_rng: bool=True) -> bool:
    """
    Function assessing whether an integer num is pandigital in a given
    base (i.e. which num is expressed in the chosen base each digit from
    0 to (base - 1)) appears as one of the digits in this expression and
    0 is not the first digit).

    Args:
        Required positional:
        num (int): The number whose status as pandigital in the chosen
                base is being assessed.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base in
                which num is to be expressed for its pandigital status.
            Default: 10
        chk_rng (bool): Whether to check that the number is in the
                value range that is a necessary condition for it to
                be pandigital in the chosen base. If this is given
                as False, it is assumed that this has already been
                tested and the test was passed.
            Default: True
    
    Returns:
    Boolean (bool) which is True if num is pandigital in the chosen base,
    False otherwise.
    """
    if chk_rng and not base ** (base - 2) <= num < base ** (base - 1):
        return False
    dig_set = set()
    while num:
        num, r = divmod(num, base)
        if not r or r in dig_set: return False
        dig_set.add(r)
    return True

"""
def startAndEndPandigital(num: int, base: int=10, target: Optional[int]=None,\
        md: Optional[int]=None) -> bool:
    if target is None: target = base ** (base - 2)
    if num < target: return False
    if md is None: md = target * base
    if not isPandigital(num % md): return False
    while num > md: num //= base
    return isPandigital(num)
"""
def FibonacciFirstKDigits(i: int, k: int, base: int=10) -> int:
    """
    Finds the first k digits in the i:th Fibonacci number (where
    the 0th term is 0 and the 1st term is 1) when expressed in
    the chosen base.

    Calculated using Binet's formula.

    Args:
    Required positional:
        i (int): Non-negative integer giving the term in the Fibonacci
                sequence for which the first k digits when expressed
                in the chosen base are to be calculated.
        k (int): Strictly positive integer giving the number of digits
                to be calculated.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the i:th Fibonacci number is to be expressed
                when finding the first k digits.
            Default: 10
    
    Returns:
    Integer (int) giving the value of the first k digits of the i:th
    Fibonacci number when expressed in the chosen base, when interpreted
    as a number in the chosen base.
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
    Finds whether the first base digits in the i:th Fibonacci number 
    where the 0th term is 0 and the 1st term is 1) when expressed in
    the chosen base are pandigital in that base. Leading zeroes are
    not allowed.

    Calculated using Binet's formula (via FibonacciFirstKDigits()).

    Args:
    Required positional:
        i (int): Non-negative integer giving the term in the Fibonacci
                sequence for which the first base digits when expressed
                in the chosen base are to be assessed for their
                pandigital status in that base.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the i:th Fibonacci number is to be expressed
                when assessing whether its first base digits are
                pandigital.
            Default: 10
    
    Returns:
    Boolean (bool) giving True if the first base digits in the i:th
    Fibonacci number when expressed in that base without leading zeroes
    are pandigital in the chosen base.
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
    """
    Solution to Project Euler #104

    Finds the smallest Fibonacci number such that when expressed in
    the chosen base, the first base digits and the last base digits
    are both pandigital in that base. Leading zeroes are not allowed
    for the first base digits.

    Args:
    Required positional:
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the i:th Fibonacci number is to be expressed
                when assessing whether its first base digits and
                last base digits are pandigital.
            Default: 10
    
    Returns:
    Integer (int) giving the value of the smallest Fibonacci number that
    fulfills the stated requirement.
    """
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
def loadSets(doc: str, relative_to_program_file_directory: bool=False) -> List[Tuple[int]]:
    """
        Optional named:
        relative_to_program_file_directory (bool): If True then
                if doc is specified as a relative path, that
                path is relative to the directory containing
                the program file, otherwise relative to the
                current working directory.
            Default: False
    """
    if relative_to_program_file_directory and not doc.startswith("/"):
        doc = os.path.join(os.path.dirname(__file__), doc)
    with open(doc) as f:
        txt = f.read()
    return [tuple(int(y.strip()) for y in x.split(",")) for x in txt.split("\n")]

def specialSubsetSumsTesting(doc: str="0105_sets.txt", relative_to_program_file_directory: bool=True) -> int:
    """
    Solution to Project Euler #105

        Optional named:
        relative_to_program_file_directory (bool): If True then
                if doc is specified as a relative path, that
                path is relative to the directory containing
                the program file, otherwise relative to the
                current working directory.
            Default: True
    """
    since = time.time()
    sp_sets = loadSets(doc, relative_to_program_file_directory=relative_to_program_file_directory)
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

def loadNetwork(doc: str, relative_to_program_file_directory: bool=False) -> Tuple[Union[int, List[Tuple[int]]]]:
    """
        Optional named:
        relative_to_program_file_directory (bool): If True then
                if doc is specified as a relative path, that
                path is relative to the directory containing
                the program file, otherwise relative to the
                current working directory.
            Default: False
    """
    if relative_to_program_file_directory and not doc.startswith("/"):
        doc = os.path.join(os.path.dirname(__file__), doc)
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

def minimalNetwork(doc: str="0107_network.txt", relative_to_program_file_directory: bool=True):
    """
    Solution to Project Euler #107

        Optional named:
        relative_to_program_file_directory (bool): If True then
                if doc is specified as a relative path, that
                path is relative to the directory containing
                the program file, otherwise relative to the
                current working directory.
            Default: True
    """
    n, edges = loadNetwork(doc, relative_to_program_file_directory=relative_to_program_file_directory)
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
    """
    Solution for Project Euler #111
    """
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
    """
    Solution to Project Euler #112
    """
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
    Solution to Project Euler #115
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
    """
    Solution to Project Euler #118
    """
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

# Problem 119
def isPower(num: int, base: int) -> int:
    """
    Identifies whether a strictly positive integer num is
    a power of the strictly positive integer base (i.e.
    num = base^n for some non-negative integer n), and if
    so what the exponent is (i.e. the integer n in the
    previous formula).

    Args:
        Required positional:
        num (int): Strictly positive integer whose status
                as an integer power of the integer base.
        base (int): The number for which the status of num
                as an integer power of this number is being
                assessed.
    
    Returns:
    Integer (int) which is the integer power to which base is
    to be taken to get num if such a number exists, and if
    not -1.
    """
    num2 = base
    res = 1
    while num2 < num:
        num2 *= base
        res += 1
    return res if num2 == num else -1

def digitCountAndDigitSum(num: int, base: int=10) -> Tuple[int, int]:
    """
    Calculates the number and sum of digits of a strictly
    positive integer when expressed terms of a given base.

    Args:
        Required positional:
        num (int): The strictly positive integer whose number
                and sum of digits when expressed in the
                chosen base is to be calculated.

        Optional named:
        base (int): The integer strictly exceeding 1 giving
                the base in which num is to be expressed when
                assessing the digit number and sum.
            Default: 10

    Returns:
    2-tuple whose index 0 contains the number of digits (without
    leading 0s) of num, and whose index 1 contains the sum of
    digits of num, both when num is expressed in the chosen base.

    Examples:
        >>> digitCountAndDigitSum(5496, base=10)
        (4, 24)

        This signifies that 5496 when expressed in base 10 (i.e.
        the standard base) has 4 digits which sum to 24
        (5 + 4 + 9 + 6).

        >>> digitCountAndDigitSum(6, base=2)
        (3, 2)

        This signifies that 6 when expressed in base 2 (binary,
        in which 6 is expressed as 101) has 3 digits which sum
        to 2 (1 + 0 + 1).
    """
    res = [0, 0]
    while num:
        num, r = divmod(num, base)
        res[0] += 1
        res[1] += r
    return tuple(res)

def powerDigitSumEqualNumDigitCountUpperBound(exp: int, base: int=10) -> int:
    """
    For a given exponent and base, finds an upper bound for the
    number of digits a strictly positive integer can have in
    that base and it be possible for the sum over the chosen exponent
    of each of its digits in the chosen base to be equal to the
    integer itself.

    Args:
        Required positional:
        exp (int): Non-negative positive integer giving the exponent
                to which each of the digits in the chosen base is
                taken in the described sum.

        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are expressed when taking
                the exponentiated digit sums as described.
    
    Returns:
    Strictly positive integer (int) giving an upper bound on the
    number of digits in the chosen base an integer may have, and
    for the described exponentiated digit sum of that integer
    to be equal to the integer itself. That is to say, there
    may exist integers with this property with this number of
    digits or fewer in the chosen base, but there cannot be
    any with more.
    """
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
    """
    TODO
    """
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

def digitPowerSum(n: int=30, base: int=10) -> int:
    """
    Solution to Project Euler #119

    Finds the n:th smallest integer no less than base, such
    that when expressed in that base, there exists a non-negative
    integer exponent for which the sum over the digits taken
    to the power of that exponent is equal to the integer
    itself.

    Args:
        Optional named:
        n (int): Strictly positive integer specifying the term
                in the sequence of the integers with the described
                property in ascending order, starting at 1 is to
                be found.
            Default: 30
        base (int): Strictly positive integer specifying the
                base in which the integers should be expressed
                when assessing the described property.
            Default: 10
    
    Returns:
    Integer (int), giving the n:th smallest integer no less than
    base such that when expressed in that base, there exists a
    non-negative integer exponent for which the sum over the
    digits taken to the power of that exponent is equal to the
    integer itself.
    """
    since = time.time()
    seq = digitPowerSumSequence(n, base=base)
    res = seq[-1][0]
    #print(seq)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 120
def squareRemainders(a_min: int=3, a_max: int=1000) -> int:
    """
    Solution to Project Euler #120

    For given non-negative integers a and n, consider the maximum
    value of the remainder of (a - 1)^n + (a + 1)^n when divided
    by a^2.
    Keeping a the fixed, consider the value of this remainder for
    all possible values of n and choose the largest. This function
    calculates the sum of these largest values for all values of
    a between a_min and a_max inclusive.
    
    Args:
        Optional named:
        a_min (int): Non-negative integer giving the smallest value
                of a considered in the sum.
            Default: 3
        a_max (int): Non-negative integer giving the largest value
                of a considered in the sum.
    
    Returns:
    Non-negative integer giving the calculated value of the sum
    described above.

    Outline of rationale:
    
    In the binomial expansion of (a + 1)^n and (a - 1)^n, all
    terms except the constant term and linear term are divisible
    by a^2. As such,
        (a + 1)^n = a * n + 1 (mod a^2)
        (a - 1)^n = (-1)^n * (1 - a * n) (mod a^2)
    Therefore, for even n:
        (a - 1)^n + (a + 1)^n = 2 (mod a^2)
    and for odd n:
        (a - 1)^n + (a + 1)^n = 2 * a * n (mod a^2)
    Thus, for even n the maximum value modulo a^2 is 2 mod a^2,
    and for odd n the maximum value modulo a^2 is when n
    is the largest odd number strictly smaller than an integer
    multiple of half of a modulo a^2. By considering the
    different remainders of a modulo 4, it can be found that
    the maximising value of n modulo a^2 is:
        floor((a - 1) / 2)
    Thus, for given a, the maximum value of (a - 1)^n + (a + 1)^n
    modulo a^2 for non-negative integers n is:
        max(2 (mod a^2), 2 * a * floor((a - 1) / 2))
    """
    res = 0
    for a in range(a_min, a_max + 1):
        md = a ** 2
        res += max(2 % md, (2 * ((a - 1) >> 1) * a) % md)
    return res

# Problem 121
def diskGameBlueDiskProbability(n_turns: int, min_n_blue_disks: int) -> Tuple[int, int]:
    """
    Consider a game consisting of a bag and red and blue disks.
    Initially, the bag contains one red and one blue disk. A
    turn of the game consists of randomly choosing a disk from
    the bag such that the probability fo drawing each individual
    disk is the same. After each turn the drawn disk is replaced
    and an additional red disk is placed in the bag.

    This function calculates the probability that after n_turns
    turns of this game the total number of times the blue disk
    is drawn is at least min_n_blue_disks as a fraction.

    Args:
        Required positional:
        n_turns (int): The number of turns in the game considered.
        min_n_blue_disks (int): The number of blue disks for which
                the probability of drawing at least this many in
                the number of turns is to be calculated.
    
    Returns:
    2-tuple giving the probability that in a run of the described
    game with n_turns turns a total of at least min_n_blue_disks are
    drawn over the course of the game, expressed as a fraction
    (numerator, denominator).
    """
    if min_n_blue_disks > n_turns: return (0, 1)
    row = [(1, 1)]
    n_red = 1
    n_tot = 2
    for i in range(n_turns):
        prev = row
        row = [(1, 1)]
        for i in range(1, min(len(prev), min_n_blue_disks) + 1):
            row.append(addFractions(multiplyFractions(prev[i - 1], (n_tot - n_red, n_tot)), multiplyFractions(prev[i], (n_red, n_tot)) if i < len(prev) else (0, 1)))
        n_red += 1
        n_tot += 1
    #print(row[-1])
    return row[-1]

def diskGameMaximumNonLossPayout(n_turns: int=15) -> int:
    """
    Solution to Project Euler #121

    Consider a game consisting of a bag and red and blue disks.
    Initially, the bag contains one red and one blue disk. A
    turn of the game consists of randomly choosing a disk from
    the bag such that the probability fo drawing each individual
    disk is the same. After each turn the drawn disk is replaced
    and an additional red disk is placed in the bag.

    The player wins if the number of blue disks drawn over the
    course of the game strictly exceeds the number of red disks
    drawn.

    Given a wager of £1 that the player will win a game consisting
    of n_turns turns, this function calculates the whole pound
    maximum payout such that as the number of attempts approaches
    infinity, the organisation running the game should not expect
    to make a net loss (with the payout including the player's
    initial wager).

    Args:
        Required positional:
        n_turns (int): The number of turns in the game considered.

    Returns:
    Integer (int) giving the whole pound maximum payout such that
    the organisation should not expect to make a net loss in the
    long term repeated running of the described game with n_turns
    turns.
    """
    player_win_n_blue_disks = (n_turns >> 1) + 1
    p_player_win = diskGameBlueDiskProbability(n_turns, player_win_n_blue_disks)
    return math.floor(p_player_win[1] / p_player_win[0])

# Problem 122
def efficientExponentiation(sum_min: int=1, sum_max: int=200, method: Optional[str]="exact") -> float:
    """
    Solution to Project Euler #122

    Calculates the sum over the least number of multiplications
    required to achieve each of the powers individually from
    sum_min to sum_max using a specified method.

    Args:
        Optional named:
        sum_min (int): Strictly positive integer giving the smallest
                exponent considered
            Default: 1
        sum_max (int): Strictly positive integer giving the largest
                exponent considered
            Default: 200
        method (string or None): Specifies the method:
                "exact": calculates exactly, in a way that is
                        guaranteed to give the correct answer
                        for any sum_min and sum_max. This is an
                        exponential time algorithm, so can be very
                        slow for larger values of sum_max. Specifying
                        the method as None defaults to this method
                "Brauer": Uses the Brauer method, which restricts
                        the search space, giving faster evaluation but
                        not guaranteeing that the result found is
                        optimum. Gives the optimum number for all
                        exponents less than 12509
                "approx": A method that further restricts the search
                        space, giving still faster evaluation but
                        again not guaranteeing that the result found
                        is optimum. Gives the optimum number for all
                        exponents less than 77.
                "binary": Uses the binary method, where the path is
                        constructed on exponents that are powers of
                        2. This is the fastest but least accurate
                        method (as it can be calculated directly from
                        the binary expression for the exponent). Gives
                        the optimum number for all exponents less than
                        14.
            Default: "exact"

    Returns:
    Integer giving the sum over the least number of multiplications
    required to achieve each of the powers individually from
    sum_min to sum_max for the chosen method, with this being guranteed
    to be the optimum if the method "exact" is chosen.
    """
    since = time.time()
    if method is None:
        method = "exact"
    
    addition_chain_calculator = AdditionChainCalculator()
    if method == "approx":
        func = addition_chain_calculator.shortestAddPathApprox
    elif method == "Brauer":
        func = addition_chain_calculator.shortestAddPathBrauer
    elif method == "exact":
        func = addition_chain_calculator.shortestAddPathExact
    elif method == "binary":
        func = addition_chain_calculator.shortestAddPathBinary
    
    res = sum(len(func(i)) - 1 for i in range(sum_min, sum_max + 1))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 123
def calculateSquareRemainder(num: int, exp: int) -> int:
    """
    For the integer num and the non-negative integer exp,
    calculates the remainder when:
        (num - 1)^exp + (num + 1)^exp
    is divided by num^2.

    Args:
        Required positional:
        num (int): The integer num used in the above expression.
        exp (int): The non-negative integer exp in the above
                expression.
    
    Returns:
    Integer between 0 and (num^2 - 1) inclusive giving the
    remainder when:
        (num - 1)^exp + (num + 1)^exp
    is divided by num^2.
    """
    md = num ** 2
    return (pow(num - 1, exp, md) + pow(num + 1, exp, md)) % md

def primeSquareRemainders(target_remainder: int=10 ** 10 + 1):
    """
    Solution to Project Euler #123

    Finds the smallest number n such that if p_n is the n:th prime
    (where p_1 = 2, p_2 = 3, ...) then the remainder when:
        (p_n - 1)^n + (p_n - 1)^n
    is divided by p_n^2, the result is at least target_remainder.

    Args:
        Optional named:
        target_remainder (int): The minimum target result of the
                calculation given above.
            Default: 10 ** 10 + 1
    
    Returns:
    Strictly positive integer (int) giving the smallest number
    n such that the remainder when:
        (p_n - 1)^n + (p_n - 1)^n
    is divided by p_n^2, the result is at least target_remainder.
    """
    # Review- prove that for even n the remainder is always 2 and
    # try to find further rules that restricts the search space,
    # or enables direct calculation of the answer- see solution to
    # Project Euler #120
    since = time.time()
    ps = PrimeSPFsieve()
    # p_n^2 must be strictly greater than the square root of target_remainder,
    # as the remainder on dividing by p_n^2 is strictly smaller than p_n^2.
    start = isqrt(target_remainder) + 1
    mx = start * 10
    ps.extendSieve(mx)
    
    i = bisect.bisect_left(ps.p_lst, start)
    if i & 1: i += 1
    # For even i, the result is always 2
    while True:
        while i >= len(ps.p_lst):
            mx *= 10
            ps.extendSieve(mx)
        #print(i + 1, ps.p_lst[i], calculateSquareRemainder(ps.p_lst[i], i + 1))
        if calculateSquareRemainder(ps.p_lst[i], i + 1) >= target_remainder:
            print(f"Time taken = {time.time() - since:.4f} seconds")
            return i + 1
        i += 2
    return -1

# Problem 124
def radicalCount(p_facts: Set[int], mx: int) -> int:
    """
    For a given set of distinct primes p_facts, finds the number of positive
    integers up to and including mx whose radical is the product of those
    primes.

    The radical of a positive integer is the product of its distinct prime
    factors (note that as 1 has no prime factoris, it has a radical of the
    multiplicative identity, 1).

    Args:
        Required positional:
        p_facts (set of ints): List of distinct prime numbers for which
                the number of integers not exceeding mx whose radicals
                are equal to the product of these primes is to be
                calculated.
                It is assumed that these are indeed primes, and this
                property is not checked.
        mx (int): The largest number considered.
    
    Returns:
    Integer (int) equal to the number of positive integers not exceeding
    mx whose radicals are equal to the product of p_facts.
    """
    n_p = len(p_facts)
    p_facts_lst = list(p_facts)
    mn = 1
    for p in p_facts_lst: mn *= p
    mx2 = mx // mn
    #if mn < 1: return 0
    #elif mn == 1: return 1
    #res = [0]

    memo = {}
    def recur(curr: int, p_i: int=0) -> int:
        if not curr: return 0
        args = (curr, p_i)
        if args in memo.keys():
            return memo[args]
        res = 1
        for i in range(p_i, n_p):
            p = p_facts_lst[i]
            res += recur(curr // p, p_i=i)
        memo[args] = res
        return res
    
    return recur(mx2, p_i=0)

def orderedRadicals(n: int=100000, k: int=10000) -> int:
    """
    Solution to Project Euler #124

    Consider all the integers between 1 and n inclusive. Sort
    these into a list based on:

    1) The radical of the integer from smallest to largest
    2) For numbers with the same radical, the size of the integer
       from smallest to largest.

    The radical of a positive integer is the product of its distinct prime
    factors (note that as 1 has no prime factoris, it has a radical of the
    multiplicative identity, 1).

    This function calculates the k:th item on that list (1-indexed)

    Args:
        Named positional:
        n (int): The largest number considered
        k (int): Which item on the list to be returned (with k = 1
                corresponding to the first item on the list).
        
    Returns:
    Integer (int) giving the k:th number on the list constructed as
    described above.

    Examples:
        >>> orderedRadicals(n=10, k=4)
        8

        >>> orderedRadicals(n=10, k=6)
        9

        Of the numbers between 1 and 10, the number 1 has radical
        1, the numbers 2, 4 and 8 have radical 2, the numbers
        3 and 9 have radical 3, and the numbers 5, 6, 7 and 10
        do not have any repeated prime factors and so they are
        each their own radical. As per the instructions for sorting
        as given above, the list for n = 10 becomes:
            [1, 2, 4, 8, 3, 9, 5, 6, 7, 10]
        Thus, for n = 10, k = 4 gives the 4th item on this list
        (8) while k = 6 gives the 6th item in this list (9).
    """
    since = time.time()
    if k == 1:
        print(f"Time taken = {time.time() - since:.4f} seconds")
        return 1
    ps = PrimeSPFsieve(n_max=k)
    """
    chk = 1
    for i in range(2, n + 1):
        pf = ps.primeFactorisation(i)
        if max(pf.values()) > 1: continue
        p_lst = sorted(pf.keys())
        rad_cnt = radicalCount(p_lst, n)
        chk += rad_cnt
    print(f"Total = {chk}")
    """

    k2 = k - 1 # Minus one to account for 1
    for i in range(2, k + 1):
        pf = ps.primeFactorisation(i)
        if max(pf.values()) > 1: continue
        #p_lst = sorted(pf.keys())
        rad_cnt = radicalCount(pf.keys(), n)
        #print(i, rad_cnt)
        if rad_cnt >= k2: break
        k2 -= rad_cnt
    #print(p_lst)
    #print(k2)
    rad = 1
    mx = n
    p_lst = sorted(pf.keys())
    n_p = len(p_lst)
    for p in p_lst:
        mx //= p
        rad *= p
    heap = [-1]
    def recur(curr: int, p_i: int=0) -> None:
        for i in range(p_i, n_p):
            p = p_lst[i]
            nxt = curr * p
            if nxt > mx: break
            if len(heap) < k2:
                heapq.heappush(heap, -nxt)
            else:
                heapq.heappushpop(heap, -nxt)
            recur(nxt, p_i=i)
        return

    recur(1, p_i=0)
    res = -heap[0] * rad
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 125
def isPalindromic(num: int, base: int=10) -> bool:
    """
    For a given non-negative integer, assesses whether it is
    palindromic when expressed in the chosen base (i.e. the
    digits in the expression read the same forwards and
    backwards).

    Args:
        Required positional:
        num (int): The non-negative integer to be assessed
                for its status as palindromic when expressed
                in the chosen base.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving
                the base in which num is to be expressed
                when assessing whether or not it is
                palindromic.
            Default: 10
    
    Returns:
    Boolean (bool) giving True if num is palindromic when
    expressed in the chosen base and False otherwise.
    """
    digs = []
    num2 = num
    while num2:
        num2, r = divmod(num2, base)
        digs.append(r)
    for i in range(len(digs) >> 1):
        if digs[i] != digs[~i]: return False
    return True

def palindromicConsecutiveSquareSumStart(start: int, mx: int, base: int=10) -> List[int]:
    """
    For a given integer start, finds all of the integers with
    value no greater than mx that can be expressed as the
    sum of at least two consecutive integer squares, starting
    with start^2, and are palindromic in the chosen base (i.e.
    the digits in the expression read the same forwards and
    backwards).

    Args:
        Required positional:
        start (int): The integer whose square is the first
                in the consecutive integer square sums
                considered.
        mx (int): The maximum allowed returned value.

        Optional named:
        base (int): Integer strictly greater than 1 giving the
                base in which integers are to be expressed
                when assessing whether or not they are
                palindromic.
            Default: 10
    
    Returns:
    List of integers (ints) giving all the sums of at least 2
    consecutive squares starting at start^2 that are 
    palindromic and no greater than mx, in strictly increasing
    order.
    """
    curr = start
    tot = curr * curr
    res = []
    while True:
        curr += 1
        tot += curr * curr
        if tot > mx: break
        if isPalindromic(tot): res.append(tot)
    return res

def palindromicConsecutiveSquareSums(mx: int=100000000 - 1, base: int=10) -> int:
    """
    Solution to Project Euler #125

    Finds the sum of all of the integers with value no greater than mx
    that are palindromic in the chosen base (i.e. the digits in the
    expression of the integer in the chosen base read the same forwards
    and backwards) and can be expressed as the sum of at least two
    consecutive integer squares.
    Note that if a palindromic integer can be expressed as the
    sum of consecutive squares in more than one way, it is still
    only included once in the sum.

    Args:
        Optional named:
        mx (int): The maximum value allowed to be included in the
                sum.
            Default: 99999999
        base (int): Integer strictly greater than 1 giving the
                base in which integers are to be expressed
                when assessing whether or not they are
                palindromic.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of all the integers with value no
    greater than mx that are palindromic and can be expressed as the
    sum of at least two consecutive integer squares.
    """
    since = time.time()
    end = isqrt(mx >> 1)
    res = 0
    palindromic_set = set()
    #count = 0
    for start in range(1, end + 1):
        lst = palindromicConsecutiveSquareSumStart(start, mx, base=10)
        palindromic_set |= set(lst)
        #print(start, lst)
        #count += len(lst)
        res += sum(lst)
    #print(count)
    res = sum(palindromic_set)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 126
def cuboidLayerSizes(dims: Tuple[int, int, int], max_layer_size: int, min_layer_size: int=1) -> List[int]:
    n_faces = (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2]) << 1
    n_edges = sum(dims) << 2
    if n_faces > max_layer_size: return []
    #n_internal_edges = 0
    #n_internal_corners = 0
    i = 0
    a = 4
    b = n_edges - 4
    c = n_faces - (min_layer_size - 1)
    rad = b ** 2 - 4 * a * c
    if rad >= 0:
        rad_sqrt = isqrt(rad)
        i = max(0, ((rad_sqrt - b) // (2 * a)) + 1)
    
    func = lambda x : a * x ** 2 + b * x + n_faces
    if func(i) < min_layer_size: print(f"i too small")
    #if i > 0 and func(i - 1) >= min_layer_size:
    #    print(f"i too large")
    #    print(f"i = {i}, func(i - 1) = {func(i - 1)}, min_layer_size = {min_layer_size}")
    #print(i)
    res = []
    #n_internal_edges = n_edges
    #n_internal_corners = 6
    while True:
        nxt = func(i)
        #print(nxt, max_layer_size)
        if nxt > max_layer_size: break
        if nxt >= min_layer_size:
            res.append(nxt)
        i += 1
    return res

def cuboidHasLayerSize(dims: Tuple[int, int, int], target_layer_size: int) -> bool:
    n_faces = (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2]) << 1
    n_edges = sum(dims) << 2
    a = 4
    b = n_edges - 4
    c = n_faces - target_layer_size
    rad = b ** 2 - 4 * a * c
    if rad < 0: return False
    rad_sqrt = isqrt(rad)
    if rad_sqrt ** 2 != rad: return False
    return rad_sqrt >= b

def cuboidLayers(target_layer_size_count: int=1000, step_size: int=10000) -> int:
    since = time.time()

    #step_size = 20000
    sz_rng = [1, step_size]
    print(sz_rng)
    #tot = 0
    #tot2 = 0
    while True:
        counts = {}
        candidates = SortedList()
        a_mx = (sz_rng[1] - 2) // 4
        for a in range(1, a_mx + 1):
            #print(f"a = {a}")
            b_mx = (sz_rng[1] - 2 * a) // (2 * (a + 1))
            for b in range(1, min(a, b_mx) + 1):
                #print(f"b = {b}")
                c_mx = (sz_rng[1] - 2 * a * b) // (2 * (a + b))
                for c in range(1, min(b, c_mx) + 1):
                    #print(f"c = {c}")
                    #print(a, b, c)
                    lst = cuboidLayerSizes((a, b, c), sz_rng[1], min_layer_size=sz_rng[0])
                    #print(a, b, c)
                    #print(lst)
                    for sz in set(lst):
                        #tot += 1
                        counts[sz] = counts.get(sz, 0) + 1
                        if counts[sz] == target_layer_size_count:
                            candidates.add(sz)
                        elif counts[sz] == target_layer_size_count + 1:
                            candidates.remove(sz)
        #print(sz_rng)
        #print(counts)
        #if 154 in counts.keys():
        #    print(f"C(154) = {counts[154]}")
        #tot2 += sum(counts.values())
        if candidates: break
        sz_rng = [sz_rng[1] + 1, sz_rng[1] + step_size]
        print(sz_rng)
    #print(f"tot = {tot}")
    #print(f"tot2 = {tot2}")
    res = candidates[0]
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 127
def abcHits(c_max: int=199999) -> int:
    """
    
    Note that if a + b = c and gcd(a, b) = 1 then gcd(a, c) = 1
    and gcd(b, c) = 1 and rad(abc) = rad(a) * rad(b) * rad(c).
    """
    since = time.time()

    

    def radical(p_facts: List[int]) -> int:
        res = 1
        for p in p_facts: res *= p
        return res

    ps = PrimeSPFsieve(n_max=c_max, use_p_lst=True)
    radicals = [1] * (c_max + 1)
    for p in ps.p_lst:
        for i in range(p, c_max + 1, p):
            radicals[i] *= p
    
    b_radicals = SortedList()

    res = 0
    for c in range(5, c_max + 1):
        if not c % 1000: print(f"c = {c}")
        b_radicals.add((radicals[c - 2], c - 2))
        if not c & 1:
            b_radicals.remove((radicals[c >> 1], c >> 1))
        #c_facts = ps.primeFactors(c)
        rad_c = radicals[c]
        if rad_c == c: continue
        rad_ab_mx = (c - 1) // rad_c
        rad_b_mx = rad_ab_mx >> 1
        i_mx = b_radicals.bisect_right((rad_b_mx, float("inf")))
        for i in range(i_mx):
            rad_b, b = b_radicals[i]
            if gcd(rad_b, rad_c) != 1: continue
            a = c - b
            if radicals[a] * rad_b <= rad_ab_mx:
                res += c
        b = c - 1
        #b_facts = ps.primeFactors(b)
        if radicals[b] <= rad_ab_mx:
            res += c
        """
        #b_sieve = [True] * c
        #for p in c_facts:
        #    start = ((((c - 1) >> 1) // p) + 1) * p
        #    for i in range(start, c, p):
        #        b_sieve[i] = False
        
        # If c is even then both a and b must be odd
        if c & 1:
            rng = ((c + 1) >> 1, c - 1)
        else:
            start = (c + 1) >> 1
            if not start & 1: start += 1
            rng = (start, c - 1, 2)
        for b in range(*rng):
            #if not b_sieve[b]: continue
            rad_b = radicals[b]
            if rad_b > rad_b_mx or gcd(rad_b, rad_c) != 1: continue
            #b_facts = ps.primeFactors(b)
            #rad_b = radical(b_facts)
            a = c - b
            #a_facts = ps.primeFactors(a)
            if radicals[a] * rad_b <= rad_ab_mx:
                res += c
        b = c - 1
        #b_facts = ps.primeFactors(b)
        if radicals[b] <= rad_ab_mx:
            res += c
        """
        """
        a_sieve = [True] * b
        for p in b_facts:
            for i in range(p, b, p):
                a_sieve[i] = False
        for a in range(1, min(b, c_max - b + 1)):
            if not a_sieve[a]: continue
            a_facts = ps.primeFactors(a)
            c = a + b
            c_facts = ps.primeFactors(c)
            res += radical(a_facts) * rad_b * radical(c_facts) < c
        """
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    
# Problem 128
def hexagonalLayerPrimeDifferenceCountIs3(layer: int, ps: PrimeSPFsieve) -> List[int]:
    """
    Consider a tessellating tiling of numbered regular hexagons
    of equal size constructed in the following manner. First,
    place hexagon 1. This is layer 0. For each subsequent layer
    (labelled with the number one greater than that of the
    previous layer), place the next unused positive integer
    hexagon so that it shares an edge (i.e. neighbours) only the
    first placed hexagon of the previous layer. Then, place the
    next unused positive integer hexagon so that it neighbours
    the first hexagon of that layer and a hexagon in the previous
    layer on the anticlockwise side of the first hexagon of that
    layer (where rotation is around hexagon 1, the very first
    hexagon placed). Then place repeatedly place the hexagon with
    the next unused positive integer in the unique position
    neighbouring the immediately previously placed hexagon and
    a hexagon in the previous layer until there are no such
    positions available. The last hexagon placed in the layer
    will be neighbouring the first hexagon placed in the layer.
    Then repeat the process with the next layer.

    For a given layer, numbered as described and with number
    no less than 2, this identifies the numbers of the hexagons
    in that layer for which three of the neighbouring hexagons
    in the tiling (i.e. hexagons that share an edge with the
    chosen hexagon) have numbers that differ from the number of
    the chosen hexagon (either above or below) by a prime number. 

    Args:
        Required positional:
        layer (int): Non-negative integer giving the layer number
                to be considered, where layer 0 consists of hexagon
                1 only, layer 1 consists of the hexagons 2 to 7,
                layer 2 consists of the hexagons 8 to 19 etc.
        ps (PrimeSPFsieve object): Object representing a prime
                sieve, enabling the rapid assessment of whether
                a number is prime in the case of repeated testing
                of relatively small number (<= 10 ** 6).
    
    Returns:
    List of integers (int) giving the numbers of hexagons in the
    chosen layer for which three of the neighbouring hexagons
    in the tiling (i.e. hexagons that share an edge with the
    chosen hexagon) have numbers that differ from the number of
    the chosen hexagon (either above or below) by a prime number.
    These are sorted in strictly increasing order.
    
    Note that this uses the fact that the only hexagons in a
    layer with number no less than 2 that can possibly have the
    described property are the first and last hexagons placed
    in that layer. For an outline of a proof of this, see
    documentation for hexagonalTileDifferences().
    """
    # layer >= 2
    #if idx not in {0, 1, 3, 5}: return False
    #num = findHexagonalCorner(layer, idx)
    if not ps.isPrime(layer * 6 - 1, extend_sieve=False, extend_sieve_sqrt=True):
        return []
    diffs = [
        (6 * layer + 1, 12 * layer + 5),
        (12 * layer - 7, 6 * layer + 5)
    ]
    res = []
    if ps.isPrime(6 * layer + 1, extend_sieve=False, extend_sieve_sqrt=True) and ps.isPrime(12 * layer + 5, extend_sieve=False, extend_sieve_sqrt=True):
        #print(layer, 0)
        res.append(3 * layer * (layer - 1) + 2)
    
    if ps.isPrime(6 * layer + 5, extend_sieve=False, extend_sieve_sqrt=True) and ps.isPrime(12 * layer - 7, extend_sieve=False, extend_sieve_sqrt=True):
        #print(layer, 1)
        res.append(3 * layer * (layer + 1) + 1)
    
    return res

def hexagonalTileDifferences(sequence_number: int=2000) -> int:
    """
    Consider a tessellating tiling of numbered regular hexagons
    of equal size constructed in the following manner. First,
    place hexagon 1. This is layer 0. For each subsequent layer
    (labelled with the number one greater than that of the
    previous layer), place the next unused positive integer
    hexagon so that it shares an edge (i.e. neighbours) only the
    first placed hexagon of the previous layer. Then, place the
    next unused positive integer hexagon so that it neighbours
    the first hexagon of that layer and a hexagon in the previous
    layer on the anticlockwise side of the first hexagon of that
    layer (where rotation is around hexagon 1, the very first
    hexagon placed). Then place repeatedly place the hexagon with
    the next unused positive integer in the unique position
    neighbouring the immediately previously placed hexagon and
    a hexagon in the previous layer until there are no such
    positions available. The last hexagon placed in the layer
    will be neighbouring the first hexagon placed in the layer.
    Then repeat the process with the next layer.

    Now consider all the hexagons in this tiling for which three
    of the neighbouring hexagons in the tiling (i.e. hexagons that
    share an edge with the chosen hexagon) have numbers that differ
    from the number of the chosen hexagon (either above or below)
    by a prime number. Let the numbers of all such hexagons,
    organised in strictly increasing order form a sequence. This
    function identifies term sequence_number in that sequence
    (where the first term is term 1).

    Args:
        Optional named:
        sequence_number (int): Strictly positive integer
                specifying the term in the sequence described to be
                returned, with the sequence starting with term
                1.
    
    Returns:
    Integer (int) giving term sequence_number in the sequence
    described above.

    Solution to Project Euler #128

    Outline of rationale:

    In the first two layers (up to hexagon 7) the hexagons
    with 3 neighbours with prime differences are hexagons
    1 and 2

    Counting the layers from 0, from layer 2 onwards (the layer
    starting with number 8) the only possible hexagons for which
    three adjacent hexagons have prime difference are the
    first and last hexagon in that layer. This can be proved as
    follows.
    
    First note that after the first two layers, the difference
    between any two neighbours is either 1 or strictly greater
    than 2 (so any neighbour difference divisible by 2 for these
    layers means that the difference is not prime).
    
    Now, consider the hexagons for which the preceding and
    succeeding values are opposite. We refer to these as
    edge hexagons. The differences between this tile and
    the preceding and suceeding tiles are both one, which
    is not prime. Considering the two neighbouring hexagons
    on the next layer in. These are two consecutive numbers
    and so the difference with the chosen hexagon must be
    even for one of them and so (since as established the
    difference cannot be 2) not prime. Thus, the difference
    can only be prime for at most one of these hexagons.
    Using identical reasoning, we can also conclude that
    for the two neighbouring hexagons on the next layer
    out, at most one of the differences with the chosen
    hexagon can be prime. Thus, for edge hexagons, the
    largest number of prime differences with neighbouring
    hexagons is 2, so none of these hexagons will be
    counted.

    Consider the hexagons for which the preceding and
    succeeding values are neighbouring but not opposite.
    We refer to these as corner hexagons. As for the edge
    hexagons, the differences between this tile and the
    preceding and succeeding tiles are both one, which
    is not prime. Considering the three neighbouring
    hexagons on the next layer out, we first note that
    the middle of these is a corner hexagon of the next
    layer out, which we refer to as the corresponding
    corner hexagon of the next layer out. These three
    hexagons contain three consecutive numbers. At most
    two of these can have prime difference with the
    chosen hexagon, and when that is the case they must
    be the two hexagons other than the corresponding
    corner hexagon of the next layer out. The remaining
    neighbouring hexagon is on the next layer in and
    is also a corner hexagon, which we refer to as the
    corresponding corner hexagon of the next layer in.
    As such, in order for three of the differences to
    be prime, the difference with the corresponding
    corner hexagon of the next layer in must be prime
    and the difference with the preceding and
    succeeding hexagons of the corresponding corner
    hexagon of the next layer out must both be
    prime. It can be shown that the corresponding corner
    hexagons of the next layer out and in must either
    both be odd or both be even, and so the corresponding
    corner hexagon on the next layer in must have
    different parity from the preceding and succeeding
    hexagons of the corresponding corner hexagon of the
    next layer out. This implies that the differences
    of these three hexagons and the chosen hexagon
    cannot all be odd and so (since as established the
    differences are all strictly greater than 2) cannot
    all be prime. Thus, like for edge hexagons, for
    corner hexagons, the largest number of prime
    differences with neighbouring hexagons is 2, so none
    of these hexagons will be counted.

    The only cases that remain for layers 2 and out
    (i.e. the only hexagons on these layers that are
    not classified as either an edge hexagon or as
    a corner hexagon) are when the hexagon does not
    neighbour to both its preceding and succeeding
    hexagon, which is the case if and only if the
    hexagon is the first in its layer or the last in
    its layer. Therefore, we restrict our search to
    those two cases. In both cases there are only 3
    neighbouring hexagons that may have prime difference,
    and a formula can be derived based on the layer
    number to calculate those candidate differences.
    TODO
    """
    since = time.time()
    ps = PrimeSPFsieve(12 * sequence_number)

    if sequence_number <= 2: return sequence_number
    count = 2
    layer = 2
    while True:
        layer_candidates = hexagonalLayerPrimeDifferenceCountIs3(layer, ps)
        count += len(layer_candidates)
        #if layer_candidates: print(layer_candidates)
        if count >= sequence_number:
            print(f"Time taken = {time.time() - since:.4f} seconds")
            return layer_candidates[~(count - sequence_number)]

        layer += 1
    return -1

# Problem 129
def findSmallestRepunitDivisibleByK(k: int, base: int=10) -> int:
    """
    For a given base, finds the smallest repunit in that base
    that is divisible by k. If no such repunit exists, then
    returns -1.

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Required positional:
        k (int): Strictly positive integer giving the quantity
                for which the returned repunit must be divisible.
        
        Optional named:
        base (int): The base in which the repunits are expressed.
            Default: 10

    Returns:
    Integer (int) giving the value of the smallest repunit in
    the chosen base that is divisible by k if any such
    repunit exists, otherwise -1.

    Note that we have used the property that if k and
    base are not coprime then no repunit in this base can
    be divisible by k. To see this, suppose k and base share
    a prime divisor p and there exists a repunit r in the
    chosen base that is divisible by k. Then r = 0 (mod p)
    and base = 0 (mod p). Now, r - 1 ends in a 0 when
    expressed in the chosen base and so is divisible by base,
    and therefore p. Consequently:
        r - 1 = 0 (mod p)
        r = p - 1 (mod p)
        0 = p - 1 (mod p)
    Given that no prime is less than 2, this cannot occur,
    so we have a contradiction. Therefore, if k and base
    share a prime divisor then there cannot exist a repunit
    in that base that is divisible by k.

    We have also used the property that if there exists a
    repunit divisible by k then the shortest such repunit
    will be at most length k (see documentation of
    repunitDivisibility() for an outline of the proof of
    this).
    """
    if gcd(k, base) != 1: return -1
    if k == 1: return 1
    base_mod_k = base % k
    res = 1
    curr = 1
    while curr:
        curr = (curr * base_mod_k + 1) % k
        res += 1
        if res > k: return -1
    return res
    
def repunitDivisibility(target_repunit_length: int=1000000, base: int=10) -> int:
    """
    Solution to Project Euler #129

    For a given base, finds the smallest integer k such that
    the smallest repunit in that base divisible by k exists and
    is no smaller than target_repunit_length.

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Optional named:
        target_repunit_length (int): Strictly positive integer
                giving the target size of the smallest repunit
                divisible by the returned value k.
            Default: 1000000
        base (int): The base in which the repunits are expressed.
            Default: 10

    Returns:
    Integer (int) giving the value k such that the smallest
    repunit in that base divisible by k exists and is no smaller
    than target_repunit_length.
    
    Outline of rationale:

    We observe that if there exists a repunit that is divisible
    by an integer k, then the smallest such repunit must have
    a length that does not exceed k. This can be seen by finding
    the remainder on division by k as we build up the repunit
    adding one 1 at a time. At each stage, we can calculate
    the value by multiplying the value for the previous repunit
    by 10 and adding 1, then taking the modulus. Thus, the
    value of a repunit can be calculated solely from the repunit
    with one fewer digit. Suppose the value calculated has been
    seen before. Then this will give rise to an infinite cycle.
    Thus, if the value 0 occurs in any of the repunits, then
    there can be no repeated values for the remainder among all
    of the repunits smaller than it. As there are only (k - 1)
    other possible remainders, this implies that if any of the
    repunits have remainder 0 on division by k (and thus the
    repunit is divisible by k) then the first such occurrence
    must be for a repunit of length no greater than k.
    """
    since = time.time()
    num = target_repunit_length
    while True:
        if findSmallestRepunitDivisibleByK(num, base=base) >= target_repunit_length:
            break
        num += 1
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return num

def compositesWithPrimeRepunitProperty(n_smallest: int, base: int=10) -> List[int]:
    """
    For a given base, finds the n_smallest smallest composite
    integers such that for each such integer, the smallest repunit
    in that base divisible by the integer exists and the number of
    digits it contains in the chosen base exactly divides one less
    than the integer.

    A composite integer is a strictly positive integer such that
    there exists a prime number that exactly divides it that is not
    equal to that integer. 

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Required positional:
        n_smallest (int): The number of integers with the described
                property to be found.
        
        Optional named:
        base (int): The base in which the repunits are expressed.
            Default: 10
    
    Returns:
    List of integers (int) giving the smallest n_smallest composite
    integers with the described property in strictly increasing
    order.
    """

    ps = PrimeSPFsieve()
    p_gen = ps.endlessPrimeGenerator()
    p0 = 2
    res = []
    for p in p_gen:
        #print(p0, p)
        for i in range(p0 + 1, p):
            val = findSmallestRepunitDivisibleByK(i, base=base)
            if val > 0 and not (i - 1) % val:
                res.append(i)
                if len(res) == n_smallest: break
        else:
            p0 = p
            continue
        break
    print(res)
    return res


def sumCompositesWithPrimeRepunitProperty(n_to_sum=25, base: int=10) -> List[int]:
    """
    Solution to Project Euler #130
    
    For a given base, finds sum of the n_to_sum smallest composite
    integers such that for each such integer, the smallest repunit
    in that base divisible by the integer exists and the number of
    digits it contains in the chosen base exactly divides one less
    than the integer.

    A composite integer is a strictly positive integer such that
    there exists a prime number that exactly divides it that is not
    equal to that integer. 

    In a given base, a repunit of length n (where n is strictly
    positive) is the strictly positive integer that when
    expressed in the chosen base is the concatenation of
    n 1s. For instance, the repunit of length 3 for base 10
    is 111 and the repunit of length 4 for base 2 is
    15 (which, when expressed in base 2 i.e. binary is 1111).

    Args:
        Required positional:
        n_to_sum (int): The number of integers with the described
                property to be included in the sum.
        
        Optional named:
        base (int): The base in which the repunits are expressed.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of the smallest n_to_sum composite
    integers with the described property in strictly increasing
    order.
    """
    since = time.time()
    res = sum(compositesWithPrimeRepunitProperty(n_to_sum, base=base))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

if __name__ == "__main__":
    to_evaluate = {130}

    if not to_evaluate or 101 in to_evaluate:
        res = optimumPolynomial(((1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1)))
        print(f"Solution to Project Euler #101 = {res}")

    if not to_evaluate or 102 in to_evaluate:
        res = triangleContainment(p=(0, 0), doc="0102_triangles.txt", relative_to_program_file_directory=True, include_surface=True)
        print(f"Solution to Project Euler #102 = {res}")
    
    if not to_evaluate or 103 in to_evaluate:
        res = specialSubsetSumsOptimum(n=7)
        print(f"Solution to Project Euler #103 = {res}")
    
    if not to_evaluate or 104 in to_evaluate:
        res = pandigitalFibonacciEnds(base=10)
        print(f"Solution to Project Euler #104 = {res}")

    if not to_evaluate or 105 in to_evaluate:
        res = specialSubsetSumsTesting(doc="0105_sets.txt", relative_to_program_file_directory=True)
        print(f"Solution to Project Euler #105 = {res}")
    
    if not to_evaluate or 106 in to_evaluate:
        res = specialSubsetSumsComparisons(n=12)
        print(f"Solution to Project Euler #106 = {res}")
    
    if not to_evaluate or 107 in to_evaluate:
        res = minimalNetwork(doc="0107_network.txt", relative_to_program_file_directory=True)
        print(f"Solution to Project Euler #107 = {res}")
    
    if not to_evaluate or 108 in to_evaluate:
        res = diophantineReciprocals(min_n_solutions=1001)
        print(f"Solution to Project Euler #108 = {res}")

    if not to_evaluate or 109 in to_evaluate:
        res = dartCheckouts(mx_score=99)
        print(f"Solution to Project Euler #109 = {res}")
    
    if not to_evaluate or 110 in to_evaluate:
        res = diophantineReciprocals(min_n_solutions=4 * 10 ** 6 + 1)
        print(f"Solution to Project Euler #110 = {res}")

    if not to_evaluate or 111 in to_evaluate:
        res = primesWithRuns(n_dig=10, base=10, ps=None)
        print(f"Solution to Project Euler #111 = {res}")

    if not to_evaluate or 112 in to_evaluate:
        res = bouncyProportions(prop_numer=99, prop_denom=100)
        print(f"Solution to Project Euler #112 = {res}")

    if not to_evaluate or 113 in to_evaluate:
        res = nonBouncyNumbers(mx_n_dig=100, base=10)
        print(f"Solution to Project Euler #113 = {res}")

    if not to_evaluate or 114 in to_evaluate:
        res = countingBlockCombinations(tot_len=50, min_large_len=3)
        print(f"Solution to Project Euler #114 = {res}")
    
    if not to_evaluate or 115 in to_evaluate:
        res = countingBlockCombinationsII(min_large_len=50, target_count=10 ** 6 + 1)
        print(f"Solution to Project Euler #115 = {res}")
    
    if not to_evaluate or 116 in to_evaluate:
        res = redGreenOrBlueTiles(tot_len=50, min_large_len=2, max_large_len=4)
        print(f"Solution to Project Euler #116 = {res}")
    
    if not to_evaluate or 117 in to_evaluate:
        res = redGreenAndBlueTiles(tot_len=50, min_large_len=2, max_large_len=4)
        print(f"Solution to Project Euler #117 = {res}")
    
    if not to_evaluate or 118 in to_evaluate:
        res = pandigitalPrimeSets(base=10)
        print(f"Solution to Project Euler #118 = {res}")
    
    if not to_evaluate or 119 in to_evaluate:
        res = digitPowerSum(n=30, base=10)
        print(f"Solution to Project Euler #119 = {res}")
    
    if not to_evaluate or 120 in to_evaluate:
        res = squareRemainders(a_min=3, a_max=1000)
        print(f"Solution to Project Euler #120 = {res}")

    if not to_evaluate or 121 in to_evaluate:
        res = diskGameMaximumNonLossPayout(15)
        print(f"Solution to Project Euler #121 = {res}")

    if not to_evaluate or 122 in to_evaluate:
        res = efficientExponentiation(sum_min=1, sum_max=200, method="exact")
        print(f"Solution to Project Euler #122 = {res}")

    if not to_evaluate or 123 in to_evaluate:
        res = primeSquareRemainders(target_remainder=10 ** 10 + 1)
        print(f"Solution to Project Euler #123 = {res}")
        
    if not to_evaluate or 124 in to_evaluate:
        res = orderedRadicals(n=100000, k=10000)
        print(f"Solution to Project Euler #124 = {res}")
    
    if not to_evaluate or 125 in to_evaluate:
        res = palindromicConsecutiveSquareSums(mx=100000000 - 1, base=10)
        print(f"Solution to Project Euler #125 = {res}")
    
    if not to_evaluate or 126 in to_evaluate:
        res = cuboidLayers(target_layer_size_count=1000, step_size=10000)
        print(f"Solution to Project Euler #126 = {res}")
    
    if not to_evaluate or 127 in to_evaluate:
        res = abcHits(c_max=119999)
        print(f"Solution to Project Euler #127 = {res}")

    if not to_evaluate or 128 in to_evaluate:
        res = hexagonalTileDifferences(sequence_number=2000)
        print(f"Solution to Project Euler #128 = {res}")
    
    if not to_evaluate or 129 in to_evaluate:
        res = repunitDivisibility(target_repunit_length=1000000, base=10)
        print(f"Solution to Project Euler #129 = {res}")

    if not to_evaluate or 130 in to_evaluate:
        res = sumCompositesWithPrimeRepunitProperty(n_to_sum=25, base=10)
        print(f"Solution to Project Euler #130 = {res}")