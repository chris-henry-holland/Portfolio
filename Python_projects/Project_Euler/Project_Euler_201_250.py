#!/usr/bin/env python

import bisect
import heapq
import itertools
import math
import numpy as np
import os
import random
import sys
import time

from collections import deque, defaultdict
from sortedcontainers import SortedDict, SortedList, SortedSet
from typing import Dict, List, Tuple, Set, Union, Generator, Callable, Optional, Any, Hashable, Iterable

sys.path.append(os.path.join(os.path.dirname(__file__), "../Algorithms_and_Datastructures/Algorithms"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../Algorithms_and_Datastructures/Data_structures"))
from misc_mathematical_algorithms import CustomFraction, gcd, lcm, isqrt, integerNthRoot
from prime_sieves import PrimeSPFsieve, SimplePrimeSieve
from pseudorandom_number_generators import generalisedLaggedFibonacciGenerator
from Pythagorean_triple_generators import pythagoreanTripleGeneratorByHypotenuse
from geometry_algorithms import grahamScan

# Problem 201
def subsetsWithUniqueSumTotal(nums: Set[int], k: int) -> int:
    """
    Given a set of integers nums, finds the sum of all integers for
    which there is exactly one subset of nums with size k whose sum
    is equal to that integer.

    Args:
        Required positional:
        nums (set of ints): The set of integers of interest. Note
                that this cannot contain repeated elements.
        k (int): The size of the subsets of nums for which an integer
                included in the sum can be the sum of exactly one
                of these subsets.
    
    Returns:
    Integer (int) giving the sum of all integers for which there is
    exactly one subset of nums with size k whose sum is equal to that
    integer.

    Outline of rationale:
    This is solved using bottom up dynamic programming, for integer m
    increasing from 0 to n finding the subset sums including the first m
    elements of the set, by adding the m:th element of nums to the
    possible subset sums including the first (m - 1) elements (i.e.
    the result of the previous step), keeping track of the number of
    included elements (capping this at k) and whether that sum for that
    number of included elements can be achieved in more than one way.
    This is optimised for k > len(nums) / 2 by noting that if an integer
    a is a unique sum among subsets of size k, then as its complement,
    (sum(nums) - a) is a unique sum among subsets of size (len(nums) - k).
    We can additionally save space by noting that sums of sets including
    more elements cannot by inserting further elements affect the sums
    of sets including fewer elements. This allows us to maintain a single
    structure with the results, taking care when adding a new element
    to iterate over the structure in decreasing order of number of
    included elements.
    Furthermore, for the final elements added, we need only focus on
    those subsets that have a number of elements that stand a chance
    of reaching the required number of elements (so those for which,
    even if all the rest of the elements to be inserted are included
    would still result in fewer than the required number of elements
    can be ignored).
    """
    n = len(nums)
    rev = False
    if (k << 1) > n:
        k = n - k
        rev = True
    if k < 0: return 0
    elif not k: return 1
    nums2 = sorted(nums)
    curr = [{0: True}]
    for j, num in enumerate(nums2):
        if len(curr) <= k: curr.append({})
        for i in reversed(range(max(0, k - (n - j)), len(curr) - 1)):
            for num2, b in curr[i].items():
                num3 = num + num2
                curr[i + 1][num3] = b and num3 not in curr[i + 1].keys()
    if len(curr) <= k: return 0
    res = sum(x for x, y in curr[k].items() if y)
    return sum(nums) * sum(curr[k].values()) - res if rev else res

def subsetsOfSquaresWithUniqueSumTotal(n_max: int=100, k: int=50) -> int:
    """
    Solution to Project Euler #201

    Given a set of all perfect squares from 1 up to n_max ** 2 inclusive,
    finds the sum of all integers for which there is exactly one subset
    of nums with size k whose sum is equal to that integer.

    Args:
        Optional named:
        n_max (int): Integer whose square is the largest perfect
                square to be included in the set.
            Defualt: 100
        k (int): The size of the subsets of the set of perfect squares
                from 1 up to n_max ** 2 inclusve for which an integer
                included in the sum can be the sum of exactly one
                of these subsets.
            Default: 50
    
    Returns:
    Integer (int) giving the sum of all integers for which there is
    exactly one subset of the set of perfect squares from 1 up to
    n_max ** 2 inclusive with size k whose sum is equal to that
    integer.

    Outline of rationale:
    See outline of rationale for subsetsWithUniqueSumTotal().
    """
    nums = {x ** 2 for x in range(1, n_max + 1)}
    res = subsetsWithUniqueSumTotal(nums, k)
    return res

# Problem 202
def equilateralTriangleReflectionCountNumberOfWays(n_reflect: int=12017639147) -> int:
    """
    Solution to Project Euler #202

    TODO
    """
    if not n_reflect & 1: return 0
    md3 = n_reflect % 3
    if not md3: return 0
    if n_reflect == 1: return 1

    def primeFactors(num: int) -> Set[int]:
        res = set()
        num2 = num
        if not num2 & 1:
            res.add(2)
            num2 >>= 1
            while not num2 & 1:
                num2 >>= 1
        for p in range(3, num + 1, 2):
            if p ** 2 > num2: break
            num3, r = divmod(num2, p)
            if r: continue
            res.add(p)
            num2 = num3
            num3, r = divmod(num2, p)
            while not r:
                num2 = num3
                num3, r = divmod(num2, p)
        if num2 != 1: res.add(num2)
        return res

    res = 0
    target = (n_reflect + 3) >> 1

    pf = sorted(primeFactors(target))
    print("factored")
    print(pf)
    pf_md3 = {p: p % 3 for p in pf}

    res = 0
    for bm in range(1 << len(pf)):
        neg = False
        num = 1
        bm2 = bm
        md3_2 = 1
        for i, p in enumerate(pf):
            if not bm2: break
            if bm2 & 1:
                neg = not neg
                num *= p
                md3_2 = (md3_2 * pf_md3[p]) % 3
            bm2 >>= 1
        ans = target // num
        md3_2 = (md3_2 * md3) % 3
        ans = (ans - md3_2) // 3

        res += -ans if neg else ans
    return res
    """
    cnt = 0
    for m in range(md3, target >> 1, 3):
        cnt += 1
        if not cnt % 1000000: print(m, target >> 1)
        for p in pf:
            if not m % p: break
        else: res += 1
    return res << 1
    """

def distinctSquareFreeBinomialCoefficientSum(n_max: int=51) -> int:
    """
    Solution to Project Euler #203

    Calculates the sum of distinct square free integers that are equal
    to a binomial coefficients (n choose k) for which n does not exceed
    n_max and 0 <= k <= n.

    An integer is square free if and only if it is strictly positive
    and is not divisible by the square of any prime number.

    Args:
        Optional named:
        n_max (int): The largest value of n for which square free integers
                equal to a binomial coefficient (n choose k) is included
                in the sum.
            Default: 51
    
    Returns:
    Integer (int) giving the sum of distinct square free integers that are
    equal to a binomial coefficients (n choose k) for which n does not
    exceed n_max and 0 <= k <= n.

    Outline of rationale:
    We simply iterate over the binomial coefficients in question, checking
    whether it is square free, utilising the following optimisations:
     1) Given that (n choose k) = n! / (k! * (n - k)!), and n! is not
        divisible by any prime greater than n, such a binomial coefficient
        cannot be divisible by any prime greater than n, and so cannot
        be divisible by the square of any prime number greater than n.
        Therefore, to check if a binomial coefficient is square free
        we only need to check if it is divisible by the square of the
        primes not exceeding n.
     2) Given that (n choose (n - k)) = (n choose k), for each n we
        need only check the values of k not exceeding n / 2.
    """
    if n_max < 0: return 0
    ps = SimplePrimeSieve(n_max - 1)
    p_lst = ps.p_lst
    res = {1}
    curr = [1]
    for n in range(1, n_max):
        
        prev = curr
        curr = [1]
        i_mx = bisect.bisect_right(p_lst, n)
        for k in range(1, len(prev)):
            num = prev[k - 1] + prev[k]
            curr.append(num)
            if num in res: continue
            for i in range(i_mx):
                p = p_lst[i]
                if not num % p ** 2: break
            else: res.add(num)
        #print(curr)
        if n & 1: curr.append(curr[-1])
    return sum(res)

# Problem 204
def generalisedHammingNumberCount(typ: int=100, n_max: int=10 ** 9) -> int:
    """
    Solution to Project Euler #204

    Calculates the number of generalised Hamming numbers of type typ not
    exceeding n_max.

    A generalised Hamming number of type n (where n is a strictly positive
    integer) is a strictly positive integer with no prime factor greater than
    n.

    Args:
        Optional named:
        typ (int): The type of generalised Hamming number to be counted.
            Default: 100
        n_max (int): Integer giving the upper bound on the generalised Hamming
                numbers of type typ to be included in the count.
            Default: 10 ** 9
    
    Returns:
    Integer (int) giving the number of generalised Hamming numbers of type
    typ not exceeding n_max.
    """
    if n_max <= 0: return 0
    ps = SimplePrimeSieve(typ)
    p_lst = ps.p_lst
    if not p_lst: return 1

    n_p = len(p_lst)
    print(p_lst)

    memo = {}
    def recur(idx: int, mx: int) -> int:
        if idx == n_p: return 1
        args = (idx, mx)
        if args in memo.keys(): return memo[args]
        p = p_lst[idx]
        mx2 = mx
        res = 0
        while mx2 > 0:
            res += recur(idx + 1, mx2)
            mx2 //= p
        memo[args] = res
        return res
    
    res = recur(0, n_max)
    #print(memo)
    return res

# Problem 205
def probabilityDieOneSumWinsFraction(die1_face_values: Tuple[int], n_die1: int, die2_face_values: Tuple[int], n_die2: int) -> CustomFraction:
    """
    
    """
    d1 = {}
    for num in die1_face_values:
        d1[num] = d1.get(num, 0) + 1
    d2 = {}
    for num in die2_face_values:
        d2[num] = d2.get(num, 0) + 1
    n1, n2 = len(die1_face_values), len(die2_face_values)

    def dieSumValues(d: Dict[int, int], n_d: int) -> Dict[int, int]:
        #print(d, n_d)
        res = {0: 1}
        curr_bin = d
        n_d2 = n_d
        while n_d2:
            if n_d2 & 1:
                prev = res
                res = {}
                for num1, f1 in prev.items():
                    for num2, f2 in curr_bin.items():
                        sm = num1 + num2
                        res[sm] = res.get(sm, 0) + f1 * f2
            prev_bin = curr_bin
            curr_bin = {}
            vals = sorted(prev_bin)
            for i1, num1 in enumerate(vals):
                f1 = prev_bin[num1]
                sm = num1 * 2
                curr_bin[sm] = curr_bin.get(sm, 0) + f1 ** 2
                for i2 in range(i1):
                    num2 = vals[i2]
                    f2 = prev_bin[num2]
                    sm = num1 + num2
                    curr_bin[sm] = curr_bin.get(sm, 0) + f1 * f2 * 2
            n_d2 >>= 1
        return res

    sms1 = dieSumValues(d1, n_die1)
    sms2 = dieSumValues(d2, n_die2)
    #print(sms1)
    #print(sms2)
    sm_vals1 = sorted(sms1.keys())
    sm_vals2 = sorted(sms2.keys())
    n_v2 = len(sm_vals2)
    i2 = 0
    res = 0
    cnt2 = 0
    for num1 in sm_vals1:
        f1 = sms1[num1]
        for i2 in range(i2, n_v2):
            if sm_vals2[i2] >= num1: break
            cnt2 += sms2[sm_vals2[i2]]
        res += cnt2 * f1
    return CustomFraction(res, n1 ** n_die1 * n2 ** n_die2)

def probabilityDieOneSumWinsFloat(die1_face_values: Tuple[int]=(1, 2, 3, 4), n_die1: int=9, die2_face_values: Tuple[int]=(1, 2, 3, 4, 5, 6), n_die2: int=6) -> float:
    """
    Solution to Project Euler #205
    """
    res = probabilityDieOneSumWinsFraction(die1_face_values, n_die1, die2_face_values, n_die2)
    #print(res)
    return res.numerator / res.denominator

# Problem 206
def concealedSquare(pattern: List[Optional[int]]=[1, None, 2, None, 3, None, 4, None, 5, None, 6, None, 7, None, 8, None, 9, None, 0], base: int=10) -> List[int]:
    """
    Solution to Project Euler #206

    Calculates all the possible strictly positive integers whose squares,
    when expressed in the chosen base, are consistent with pattern, where
    pattern gives the digit values when read from left to right, with
    an integer representing the digit value that must go at the corresponding
    location in the representation of the square and None representing that
    in that position any digit is allowed.

    Args:
        Optional named:
        pattern (list of ints/None): The pattern that the square of any
                returned value must be consistent with, as outlined above.
            Default: [1, None, 2, None, 3, None, 4, None, 5, None, 6, None, 7, None, 8, None, 9, None, 0]
        base (int): Integer strictly greater than 1 giving the base in which
                the squares of integers should be expressed when assessing
                whether they are consistent with pattern.
            Defualt: 10
    
    Returns:
    List of integers (int) giving all the strictly positive integers whose
    squares, when expressed in the chosen base, are consistent with pattern
    (as outlined above) in strictly increasing order of size.
    """
    mn = 0
    mx = 0
    for i, d in enumerate(pattern):
        mn *= base
        mx *= base
        if d is None:
            mx += base - 1
            if not i: mn += 1
        else:
            mn += d
            mx += d
    sqrt_mn = isqrt(mn)
    sqrt_mx = isqrt(mx)
    print(sqrt_mn, sqrt_mx)
    
    def isMatch(num: int) -> bool:
        for i in reversed(range(len(pattern))):
            if pattern[i] is None:
                num //= base
                continue
            num, d = divmod(num, base)
            if d != pattern[i]: return False
        if num: return False
        return True
    
    def isPartialMatch(num: int, n_digs: int) -> bool:
        for i in range(n_digs):
            if pattern[~i] is None:
                num //= base
                continue
            num, d = divmod(num, base)
            if d != pattern[~i]: return False
        return True

    poss_tails = []
    n_tail_digs = len(pattern) >> 2
    for num_sqrt in range(base ** n_tail_digs):
        num = num_sqrt ** 2
        if isPartialMatch(num, n_tail_digs):
            poss_tails.append(num_sqrt)
    #print(f"n possible tails = {len(poss_tails)}")
    #print(poss_tails)
    res = []
    div = base ** n_tail_digs
    for num_sqrt_head in range(sqrt_mn // div, (sqrt_mx // div) + 1):
        for num_sqrt_tail in poss_tails:
            num_sqrt = num_sqrt_head * div + num_sqrt_tail
            num = num_sqrt ** 2
            if isMatch(num):
                print(num_sqrt, num)
                res.append(num_sqrt)
    return res
    """
    for num_sqrt in range(sqrt_mn, sqrt_mx + 1):
        #print(num_sqrt)
        num = num_sqrt ** 2
        if isMatch(num):
            print(num)
            res.append(num)
    return res
    """

# Problem 207
def findSmallestPartitionBelowGivenProportion(proportion: CustomFraction=CustomFraction(1, 12345)) -> int:
    """
    Solution to Project Euler #207
    """
    def findExponent(proportion: CustomFraction) -> int:
        comp = lambda m: m * proportion.denominator < 2 * (2 ** m - 1) * proportion.numerator
        if comp(0): return 0
        n = 1
        while True:
            if comp(n):
                break
            n <<= 1
        lft, rgt = n >> 1, n
        #print(lft, rgt)
        while lft < rgt:
            mid = lft + ((rgt - lft) >> 1)
            if comp(mid): rgt = mid
            else: lft = mid + 1
        #print(lft)
        return lft
    
    n = findExponent(proportion)
    l = ((n * proportion.denominator) // proportion.numerator) + 1
    #print(n, l)
    return ((2 * l + 1) ** 2 - 1) >> 2

# Problem 208
def robotWalks(reciprocal: int=5, n_steps: int=70) -> int:
    """
    Solution to Project Euler #208
    """
    # Review- attempt more efficient solution with either binary lifting
    # or double ended approach
    # Review- does reciprocal need to be prime for this to work?
    if n_steps % reciprocal: return 0
    curr = {}
    start = tuple([1] + [0] * (reciprocal - 2))
    curr[(0, start)] = 1
    target = n_steps // reciprocal
    for step in range(n_steps - 1):
        prev = curr
        curr = {}
        for (i, cnts), f in prev.items():
            for j in ((i + 1) % reciprocal, (i - 1) % reciprocal):
                if j == reciprocal - 1:
                    cnt = step - sum(cnts)
                    if cnt > target: continue
                    k = (j, cnts)
                else:
                    cnts2 = list(cnts)
                    cnts2[j] += 1
                    if cnts2[j] > target: continue
                    k = (j, tuple(cnts2))
                curr[k] = curr.get(k, 0) + f
        #print(step, curr)
    target_cnts = tuple([target] * (reciprocal - 1))
    #print(curr)
    return curr.get((1, target_cnts), 0) + curr.get(((reciprocal - 1), target_cnts), 0)

# Problem 209
def countZeroMappings(n_inputs: int=6) -> int:
    """
    Solution to Project Euler #209
    """
    def bitmaskFunction(bm: int) -> int:
        res = (bm & ((1 << (n_inputs - 1)) - 1)) << 1
        bm2 = bm >> (n_inputs - 3)
        res |= ((bm2 & 4) >> 2) ^ (((bm2 & 2) >> 1) & (bm2 & 1))
        return res
    
    bm_mapping = [bitmaskFunction(bm) for bm in range(1 << n_inputs)]
    #print(bm_mapping)
    #print(len(bm_mapping))

    seen = set()
    cycle_lens = {}
    for bm in range(1 << n_inputs):
        if bm in seen: continue
        l = 0
        bm2 = bm
        while bm2 not in seen:
            seen.add(bm2)
            l += 1
            bm2 = bm_mapping[bm2]
        cycle_lens[l] = cycle_lens.get(l, 0) + 1
    mx_cycle_len = max(cycle_lens.keys())
    n_opts = [0, 1, 3]
    for _ in range(3, mx_cycle_len + 1):
        n_opts.append(n_opts[-1] + n_opts[-2])
    res = 1
    for l, f in cycle_lens.items():
        res *= n_opts[l] ** f
    return res

# Problem 210
def countObtuseTriangles(r: Union[int, float]=10 ** 9, div: Union[int, float]=4) -> int:
    """
    Solution to Project Euler #210
    """
    r2 = math.floor(r)
    d = math.floor(2 * r / div)
    tot = ((r2 * (r2 + 1)) << 1) + 1
    diag1_cnt = ((r2 >> 1) << 1) + 1
    diag2_cnt = (((r2 + 1) >> 1) << 1)
    strip_n_diag1 = (d >> 1) + 1
    strip_n_diag2 = (d + 1) >> 1
    strip_cnt_wo_diag = (diag1_cnt - 1) * strip_n_diag1 + diag2_cnt * strip_n_diag2
    print(f"tot = {tot}, strip_cnt_wo_diag = {strip_cnt_wo_diag}, diag1_cnt = {diag1_cnt}")
    res = tot - strip_cnt_wo_diag - diag1_cnt
    
    
    # small triangles
    d2 = r / div
    c = d2 / 2
    small_cnt = 0
    for x in range(math.floor((1 - math.sqrt(2)) * c) + 1, math.ceil((1 + math.sqrt(2)) * c)):
        discr = c ** 2 - x ** 2 + 2 * c * x
        discr_sqrt = math.sqrt(discr)
        y_mn = math.floor(c - discr_sqrt) + 1
        y_mx = math.ceil(c + discr_sqrt) - 1
        small_cnt += y_mx - y_mn + 1
    small_cnt -= math.ceil(d2) - 1
    print(f"small count = {small_cnt}")
    
    return res + small_cnt

# Problem 211
def divisorSquareSumIsSquareTotal(n_max: int=64 * 10 ** 6 - 1) -> int:
    """
    Solution to Project Euler #211
    """
    # Review- try to speed up
    def isSquare(num: int) -> bool:
        num_sqrt = isqrt(num)
        return num_sqrt ** 2 == num

    arr = [1] * (n_max + 1)
    res = 1
    for num in range(2, n_max + 1):
        if not num % 500: print(num)
        num_sq = num ** 2
        for num2 in range(num, n_max + 1, num):
            arr[num2] += num_sq
        if isSquare(arr[num]):
            print(num, arr[num])
            res += num
    return res

# Problem 212
def cuboidUnionVolume(cuboids: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]) -> int:

    # Review- try to speed up
    # TODO- investigate why repeats occur for triple or more intersections
    # where the two cuboids with the largest x0-value have the same x0-value.

    def cuboidVolume(cuboid: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> int:
        res = 1
        for l in cuboid[1]:
            res *= l
        return res

    def cuboidIntersection(cuboid1: Tuple[Tuple[int, int, int], Tuple[int, int, int]], cuboid2: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        res = [[], []]
        for i in range(len(cuboid1[0])):
            x1 = max(cuboid1[0][i], cuboid2[0][i])
            x2 = min(cuboid1[0][i] + cuboid1[1][i], cuboid2[0][i] + cuboid2[1][i])
            if x1 >= x2: return None
            res[0].append(x1)
            res[1].append(x2 - x1)
        return tuple(tuple(x) for x in res)

    cuboids2 = SortedList([(x, {i}) for i, x in enumerate(cuboids)])

    res = 0
    i1 = 0
    seen = set()
    cnt = 0
    while cuboids2:
        cuboid, inds = cuboids2.pop(0)
        cnt += 1
        if not cnt % 1000:
            print(f"seen {cnt}, x0 = {cuboid[0][0]}, list length currently {len(cuboids2)}")
        inds_tup = tuple(sorted(inds))
        if inds_tup in seen:
            print(f"repeated index combination: {inds_tup}")
            for idx in inds_tup:
                print(cuboids[idx])
            continue
        seen.add(inds_tup)
        vol = cuboidVolume(cuboid)
        # Inclusion-exclusion
        res += vol if (len(inds) & 1) else -vol
        add_lst = []
        mx_x = cuboid[0][0] + cuboid[1][0]
        if len(inds) > 1: continue
        for i2 in range(len(cuboids2)):
            cuboid2, inds2 = cuboids2[i2]
            if cuboid2[0][0] >= mx_x: break
            elif not inds.isdisjoint(inds2): continue
            intersect = cuboidIntersection(cuboid, cuboid2)
            if intersect is None: continue
            add_lst.append((intersect, inds.union(inds2)))
        for tup in add_lst: cuboids2.add(tup)
        i1 += 1
    print(f"total seen = {cnt}, unique seen = {len(seen)}")
    return res

def laggedFibonacciNDimensionalHyperCuboidGenerator(
    hypercuboid_smallest_coord_ranges: Tuple[Tuple[int, int]]=((0, 9999), (0, 9999), (0, 9999)),
    hypercuboid_dim_ranges: Tuple[Tuple[int, int]]=((1, 399), (1, 399), (1, 399)),
    n_hypercuboids: Optional[int]=None,
    l_fib_modulus: int=10 ** 6,
    l_fib_poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007),
    l_fib_lags: Tuple[int]=(24, 55),
) -> Generator[Tuple[Tuple[int, int, int], Tuple[int, int, int]], None, None]:
    """
    Generator yielding TODO

    The generalisation of the lagged Fibonacci generator sequence for the
    given tuple of integers l_fib_poly_coeffs, tuple of strictly positive integers
    l_fib_lags, and the integers min_val and max_val is the sequence such that
    for integer i >= 1, the i:th term in the sequence is:
        t_i = (sum j from 0 to len(l_fib_poly_coeffs) - 1) (l_fib_poly_coeffs[j] * i ** j) % n_vertices
                for i <= max(l_fib_lags)
              ((sum j fro 0 to len(l_fib_lags) - 1) (t_(i - l_fib_lags[i]))) % n_vertices
                otherwise
    where % signifies modular division (i.e. the remainder of the integer
    preceding that symbol by the integer succeeding it). This sequence contains
    integer values between 0 and (n_vertices - 1) inclusive.

    The terms where i <= max(l_fib_lags) are referred as the polynomial
    terms and the terms where i > max(l_fib_lags) are referred to as the
    recursive terms.

    Note that if n_hypercuboids is not specified, the generator never terminates and
    thus any iterator over this generator must include provision to terminate
    (e.g. a break or return statement), otherwise it would result in an infinite
    loop.

    Args:
        Optional named:
        TODO
        l_fib_poly_coeffs (tuple of ints): Tuple of integers giving the
                coefficients of the polynomial used to calculate the
                polynomial terms of the generalisation of the lagged
                Fibonacci generator sequence used to generate the
                edges.
            Default: (100003, -200003, 0, 300007)
        l_fib_lags (tuple of ints): Tuple of strictly positive integers,
                which when calculating the recursive terms of the
                generlisation of the lagged Fibonacci generator sequence
                used to generate the edges, indicates how many steps back
                in the sequence the previous terms summed should each be
                from the position of the term being generated. Additionally,
                the maximum value determines at which term the transition
                from the polynomial terms to the recursive terms will occur
                in this sequence.
            Default: (24, 55)
    
    Yields:
    TODO
    If n_hypercuboids is specified as a non-negative integer then (unless
    externally terminated first) exactly n_edges such values are yielded,
    otherwise the generator never of itself terminates.
    """
    n_dim = len(hypercuboid_smallest_coord_ranges)
    it = generalisedLaggedFibonacciGenerator(poly_coeffs=l_fib_poly_coeffs, lags=l_fib_lags, min_val=0, max_val=l_fib_modulus - 1)
    it2 = range(n_hypercuboids) if isinstance(n_hypercuboids, int) and n_hypercuboids >= 0 else itertools.count(0)
    for _ in it2:
        cuboid = [[], []]
        for rng in hypercuboid_smallest_coord_ranges:
            cuboid[0].append(rng[0] + (next(it) % (rng[1] - rng[0] + 1)))
        for rng in hypercuboid_dim_ranges:
            cuboid[1].append(rng[0] + (next(it) % (rng[1] - rng[0] + 1)))
        yield tuple(tuple(x) for x in cuboid)
    return 

def laggedFibonacciCuboidUnionVolume(
    n_cuboids: int=50000,
    cuboid_smallest_coord_ranges: Tuple[Tuple[int, int]]=((0, 9999), (0, 9999), (0, 9999)),
    cuboid_dim_ranges: Tuple[Tuple[int, int]]=((1, 399), (1, 399), (1, 399)),
    l_fib_modulus: int=10 ** 6,
    l_fib_poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007),
    l_fib_lags: Tuple[int]=(24, 55),
) -> int:
    """
    Solution to Project Euler #212
    """
    cuboids = [c for c in laggedFibonacciNDimensionalHyperCuboidGenerator(
        hypercuboid_smallest_coord_ranges=cuboid_smallest_coord_ranges,
        hypercuboid_dim_ranges=cuboid_dim_ranges,
        n_hypercuboids=n_cuboids,
        l_fib_modulus=l_fib_modulus,
        l_fib_poly_coeffs=l_fib_poly_coeffs,
        l_fib_lags=l_fib_lags,
    )]
    res = cuboidUnionVolume(cuboids)
    return res

# Problem 213
def fleaCircusExpectedNumberOfUnoccupiedSquaresFraction(dims: Tuple[int, int], n_steps: int) -> CustomFraction:

    def transferFunction(pos: Tuple[int, int]) -> Tuple[Set[Tuple[int, int]], CustomFraction]:
        denom = 4
        denom -= (pos[0] == 0) + (pos[0] == (dims[0] - 1)) + (pos[1] == 0) + (pos[1] == (dims[1] - 1))
        p = CustomFraction(1, denom)
        res = set()
        if pos[0] > 0: res.add((pos[0] - 1, pos[1]))
        if pos[0] < dims[0] - 1: res.add((pos[0] + 1, pos[1]))
        if pos[1] > 0: res.add((pos[0], pos[1] - 1))
        if pos[1] < dims[1] - 1: res.add((pos[0], pos[1] + 1))
        return (res, p)

    is_square = (dims[0] == dims[1])

    p_arr = [[CustomFraction(1, 1)] * dims[1] for _ in range(dims[0])]
    sym_funcs = []
    sym_funcs.append(lambda pos: pos)
    sym_funcs.append(lambda pos: (dims[0] - 1 - pos[0], pos[1]))
    sym_funcs.append(lambda pos: (pos[0], dims[1] - 1 - pos[1]))
    sym_funcs.append(lambda pos: (dims[0] - 1 - pos[0], dims[1] - 1 - pos[1]))
    if is_square:
        sym_funcs.append(lambda pos: (pos[1], pos[0]))
        sym_funcs.append(lambda pos: (dims[0] - 1 - pos[1], pos[0]))
        sym_funcs.append(lambda pos: (pos[1], dims[1] - 1 - pos[0]))
        sym_funcs.append(lambda pos: (dims[0] - 1 - pos[1], dims[1] - 1 - pos[0]))

    for i1 in range((dims[0] + 1) >> 1):
        for i2 in range(i1 if is_square else 0, (dims[1] + 1) >> 1):
            arr = [[0] * dims[1] for _ in range(dims[0])]
            arr[i1][i2] = 1
            for m in range(n_steps):
                arr0 = arr
                arr = [[0] * dims[1] for _ in range(dims[0])]
                for j1 in range(dims[0]):
                    odd = (i1 + i2 + j1 + m) & 1
                    for j2 in range(odd, dims[1], 2):
                        if arr0[j1][j2] == 0: continue
                        t_set, p = transferFunction((j1, j2))
                        for pos in t_set:
                            arr[pos[0]][pos[1]] += arr0[j1][j2] * p
            seen = set()
            sym_funcs2 = []
            for sym_func in sym_funcs:
                pos = sym_func((i1, i2))
                if pos in seen: continue
                seen.add(pos)
                sym_funcs2.append(sym_func)
            
            for j1 in range(dims[0]):
                odd = (i1 + i2 + j1 + n_steps) & 1
                for j2 in range(odd, dims[1], 2):
                    if arr[j1][j2] == 0: continue
                    for sym_func in sym_funcs2:
                        pos2 = sym_func((j1, j2))
                        p_arr[pos2[0]][pos2[1]] *= 1 - arr[j1][j2]
    res = sum(sum(row) for row in p_arr)
    return res

def fleaCircusExpectedNumberOfUnoccupiedSquaresFloatDirect(dims: Tuple[int, int], n_steps: int) -> float:
    """
    Solution to Project Euler #213
    """

    def transferFunction(pos: Tuple[int, int]) -> Tuple[Set[Tuple[int, int]], float]:
        denom = 4
        denom -= (pos[0] == 0) + (pos[0] == (dims[0] - 1)) + (pos[1] == 0) + (pos[1] == (dims[1] - 1))
        p = 1 / denom
        res = set()
        if pos[0] > 0: res.add((pos[0] - 1, pos[1]))
        if pos[0] < dims[0] - 1: res.add((pos[0] + 1, pos[1]))
        if pos[1] > 0: res.add((pos[0], pos[1] - 1))
        if pos[1] < dims[1] - 1: res.add((pos[0], pos[1] + 1))
        return (res, p)

    is_square = (dims[0] == dims[1])

    p_arr = [[1] * dims[1] for _ in range(dims[0])]
    sym_funcs = []
    sym_funcs.append(lambda pos: pos)
    sym_funcs.append(lambda pos: (dims[0] - 1 - pos[0], pos[1]))
    sym_funcs.append(lambda pos: (pos[0], dims[1] - 1 - pos[1]))
    sym_funcs.append(lambda pos: (dims[0] - 1 - pos[0], dims[1] - 1 - pos[1]))
    if is_square:
        sym_funcs.append(lambda pos: (pos[1], pos[0]))
        sym_funcs.append(lambda pos: (dims[0] - 1 - pos[1], pos[0]))
        sym_funcs.append(lambda pos: (pos[1], dims[1] - 1 - pos[0]))
        sym_funcs.append(lambda pos: (dims[0] - 1 - pos[1], dims[1] - 1 - pos[0]))

    for i1 in range((dims[0] + 1) >> 1):
        for i2 in range(i1 if is_square else 0, (dims[1] + 1) >> 1):
            arr = [[0] * dims[1] for _ in range(dims[0])]
            arr[i1][i2] = 1
            for m in range(n_steps):
                arr0 = arr
                arr = [[0] * dims[1] for _ in range(dims[0])]
                for j1 in range(dims[0]):
                    odd = (i1 + i2 + j1 + m) & 1
                    for j2 in range(odd, dims[1], 2):
                        if arr0[j1][j2] == 0: continue
                        t_set, p = transferFunction((j1, j2))
                        for pos in t_set:
                            arr[pos[0]][pos[1]] += arr0[j1][j2] * p
            seen = set()
            sym_funcs2 = []
            for sym_func in sym_funcs:
                pos = sym_func((i1, i2))
                if pos in seen: continue
                seen.add(pos)
                sym_funcs2.append(sym_func)
            
            for j1 in range(dims[0]):
                odd = (i1 + i2 + j1 + n_steps) & 1
                for j2 in range(odd, dims[1], 2):
                    if arr[j1][j2] == 0: continue
                    for sym_func in sym_funcs2:
                        pos2 = sym_func((j1, j2))
                        p_arr[pos2[0]][pos2[1]] *= 1 - arr[j1][j2]
    res = sum(sum(row) for row in p_arr)
    return res

def fleaCircusExpectedNumberOfUnoccupiedSquaresFloatFromFraction(dims: Tuple[int, int]=(30, 30), n_steps: int=50) -> float:
    """
    Alternative (more precise) solution to Project Euler #213
    """
    res = fleaCircusExpectedNumberOfUnoccupiedSquaresFraction(dims, n_steps)
    print(res)
    return res.numerator / res.denominator

# Problem 214
def primesOfTotientChainLengthSum(p_max: int=4 * 10 ** 7 - 1, chain_len: int=25) -> int:
    """
    Solution to Project Euler #214
    """
    # It appear that for chain lengths len no less than 2, the last integer
    # with that chain length is 2 * 3 ** (len - 2). Can this be proved?
    ps = PrimeSPFsieve(p_max)
    print("calculated prime sieve")
    totient_vals = [0, 1, 2]
    totient_lens = [0, 1, 2]
    last_chain_lens = [0, 1, 2]
    last_prime_chain_lens = [-1, -1, 2]
    
    res = 0
    for num in range(3, p_max + 1):
        p, exp, num2 = ps.sieve[num]
        #print(num, (p, exp, num2), totient_vals)
        if p == num:
            # num is prime
            totient_vals.append(num - 1)
            totient_lens.append(totient_lens[totient_vals[-1]] + 1)
            if totient_lens[-1] == chain_len:
                #print(num)
                res += num
            last_prime_chain_lens += [-1] * max(0, totient_lens[-1] + 1 - len(last_prime_chain_lens))
            last_prime_chain_lens[totient_lens[-1]] = num
        else:
            totient_vals.append(totient_vals[num2] * (p - 1) * p ** (exp - 1))
            totient_lens.append(totient_lens[totient_vals[-1]] + 1)
        last_chain_lens += [-1] * max(0, totient_lens[-1] + 1 - len(last_chain_lens))
        last_chain_lens[totient_lens[-1]] = num
    #print(totient_vals)
    #print(totient_lens)
    print(last_chain_lens)
    return res

# Problem 215
def crackFreeWalls(n_rows: int=32, n_cols: int=10) -> int:
    """
    Solution to Project Euler #215
    """
    if not n_rows or not n_cols: return 1
    row_opts = []
    row_opts_dict = {}
    transfer = []

    step_opts = {2, 3}
    mn_step_opt = min(step_opts)

    #memo = {}
    def recur(mn_remain: int=n_cols, diff: int=0) -> Generator[Tuple[Tuple[int], Tuple[int]], None, None]:
        if not mn_remain:
            if not diff:
                yield ((), ())
                return
            elif diff in step_opts:
                ans = ((), (diff,))
                yield ans
                return
            return
        elif mn_remain < mn_step_opt: return
        #args = (mn_remain, diff)
        #if args in memo.keys():
        #    return memo[args]
        res = []
        for step in step_opts:
            diff2 = diff - step
            if not diff2: continue
            if diff2 > 0:
                for ans in recur(mn_remain=mn_remain, diff=diff2):
                    ans2 = (ans[0], tuple([step] + list(ans[1])))
                    #print(1, mn_remain, diff, ans2)
                    yield ans2
            else:
                mn_remain2 = mn_remain + diff2
                for ans in recur(mn_remain=mn_remain2, diff=-diff2):
                    ans2 = (ans[1], tuple([step] + list(ans[0])))
                    #print(2, mn_remain, diff, ans2)
                    yield ans2

        #res = tuple(res)
        #memo[args] = res
        #return res
        return

    transfer = []
    for pair in recur(mn_remain=n_cols, diff=0):
        #print(pair)
        for tup in pair:
            if tup in row_opts_dict.keys(): continue
            row_opts_dict[tup] = len(row_opts)
            row_opts.append(tup)
            transfer.append(set())
        idx1, idx2 = row_opts_dict[pair[0]], row_opts_dict[pair[1]]
        transfer[idx1].add(idx2)
    
    #print(row_opts)
    #print(transfer)
    n_opts = len(row_opts)
    #print(f"n_opts = {n_opts}")
    curr = [1] * n_opts
    for _ in range(n_rows - 1):
        prev = curr
        curr =  [0] * n_opts
        for i1 in range(n_opts):
            for i2 in transfer[i1]:
                curr[i2] += prev[i1]
    return sum(curr)

# Problem 216
def countPrimesOneLessThanTwiceASquare(n_max: int=5 * 10 ** 7) -> int:
    """
    Solution to Project Euler #216
    """
    # Review- look into the more efficient methods as outlined in
    # the PDF document accompanying the problem.
    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> bool:
        res = ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)
        return res[0]

    res = 0
    for num in range(2, n_max + 1):
        if not num % 10000: print(num)
        res += primeCheck(2 * num ** 2 - 1)
    return res

# Problem 217
def balancedNumberCount(max_n_dig: int=47, base: int=10, md: Optional[int]=3 ** 15) -> int:
    """
    Solution to Project Euler #217
    """
    # Review- see if it can be made more efficient using cumulative totals
    if max_n_dig < 1: return 0
    dig_sum = (base * (base - 1)) >> 1
    #if max_n_dig == 1:
    #    res = dig_sum
    #    return res if md is None else res % md
    #elif max_n_dig == 2:
    #    res = dig_sum * (base + 1)
    #    return res if md is None else res % md
    #elif max_n_dig == 3:
    #    res = dig_sum * (base ** 2 + 1) + base * 

    def calculateRunningTotalWithoutMod(i: int, res_init: int=0) -> int:
        res = res_init
        if (i << 1) + 1 > max_n_dig:
            for j in range(1, len(row_lft)):
                res += row_lft[j][1] * (base ** i) * row_rgt[j][0] + row_rgt[j][1] * row_lft[j][0]
            return res
        for j in range(1, len(row_lft)):
            res += (row_lft[j][1] * (base ** 2 + 1) + dig_sum * row_lft[j][0])* (base ** i) * row_rgt[j][0] + (base + 1) * row_rgt[j][1] * row_lft[j][0]
        return res
    
    def calculateRunningTotalWithMod(i: int, res_init: int=0) -> int:
        res = res_init
        if (i << 1) + 1 > max_n_dig:
            for j in range(1, len(row_lft)):
                res = (res + row_lft[j][1] * pow(base, i, md) * row_rgt[j][0] + row_rgt[j][1] * row_lft[j][0]) % md
            return res
        for j in range(1, len(row_lft)):
            res = (res + (row_lft[j][1] * (base ** 2 + 1) + dig_sum * row_lft[j][0])* pow(base, i, md) * row_rgt[j][0] + (base + 1) * row_rgt[j][1] * row_lft[j][0]) % md
        return res
    
    calculateRunningTotal = calculateRunningTotalWithoutMod if md is None else calculateRunningTotalWithMod
    
    m = max_n_dig >> 1
    #print(f"m = {m}")
    row_rgt = [[1, x] for x in range(base)]
    row_lft = [[1, x] for x in range(base)]
    row_lft[0] = [0, 0]
    
    res = dig_sum
    if max_n_dig == 1:
        return res
    res = calculateRunningTotal(1, res_init=res)
    #print(1)
    #print(row_lft)
    #print(row_rgt)
    #print(res)
    for i in range(2, m + 1):
        prev_rgt = row_rgt
        row_rgt = [[0, 0] for x in range(len(prev_rgt) + base - 1)]
        prev_lft = row_lft
        row_lft = [[0, 0] for x in range(len(prev_lft) + base - 1)]
        for row, prev, mn in [(row_lft, prev_lft, 1), (row_rgt, prev_rgt, 0)]:
            for j in range(0, len(prev)):
                for d in range(base):
                    j2 = j + d
                    row[j2][0] += prev[j][0]
                    row[j2][1] += (prev[j][1] * base) + d * prev[j][0]
        res = calculateRunningTotal(i, res_init=res)
        #print(f"i = {i}")
        #print(row_lft)
        #print(row_rgt)
        #print(res)
        
        """
        if md is None:
            if (i << 1) + 1 > max_n_dig:
                for j in range(1, len(row_lft)):
                    res += row_lft[j][1] * (base ** m) * row_rgt[j][0] + row_rgt[j][1] * row_lft[j][0]
                continue
            for j in range(1, len(row_lft)):
                res += (row_lft[j][1] * (base ** 2 + 1) + dig_sum * row_lft[j][0])* (base ** m) * row_rgt[j][0] + 2 * row_rgt[j][1] * row_lft[j][0]
            continue
        if (i << 1) + 1 > max_n_dig:
            for j in range(1, len(row_lft)):
                res = (res + row_lft[j][1] * pow(base, m, md) * row_rgt[j][0] + row_rgt[j][1] * row_lft[j][0]) % md
            continue
        for j in range(1, len(row_lft)):
            res = (res + (row_lft[j][1] * (base ** 2 + 1) + dig_sum * row_lft[j][0])* pow(base, m, md) * row_rgt[j][0] + 2 * row_rgt[j][1] * row_lft[j][0]) % md
        """
            
    return res


# Problem 218
def perfectRightAngledTriangleGenerator(max_hypotenuse: Optional[int]=None) -> Generator[Tuple[Tuple[int, int, int], bool], None, None]:

    #m = 1
    heap = []
    if max_hypotenuse is None: max_hypotenuse = float("inf")
    perfect_cnt = 0
    for m in itertools.count(1):
        m += 1
        m_odd = m & 1
        n_mn = 1 + m_odd
        m_sq = m ** 2

        #m2_mn
        min_hyp = (m_sq + n_mn ** 2) ** 2
        while heap and heap[0][0] < min_hyp:
            ans = heapq.heappop(heap)
            yield tuple(ans[::-1])
            perfect_cnt += 1
        if min_hyp > max_hypotenuse: break
        n_mx = min(m - 1, isqrt(isqrt(max_hypotenuse) - m_sq)) if max_hypotenuse != float("inf") else m - 1
        # Note that since m and n are coprime and not both can be odd,
        # m and n must have different parity (as if they were both
        # even then they would not be coprime)
        n = 1 + m_odd
        for n in range(n_mn, n_mx + 1, 2):
            if gcd(m, n) != 1:
                n += 1
                continue
            m2, n2 = m_sq - n ** 2, 2 * m * n
            if m2 ** 2 + n2 ** 2 > max_hypotenuse: break
            # Note that since m and n are of different parity and coprime, m2 and n2
            # are also guaranteed to be of different parit and coprime
            #if m2 & 1 == n2 & 1: continue
            if m2 < n2: m2, n2 = n2, m2
            m2_sq, n2_sq = m2 * m2, n2 * n2
            a, b, c = m2_sq - n2_sq, 2 * m2 * n2, m2_sq + n2_sq
            if b < a: a, b = b, a
            heapq.heappush(heap, (c, b, a))
            n += 1
        #for n in range(1 + m_odd, max_n + 1, 2):
        #    if gcd(m, n) != 1: continue
        #    a, b, c = m_sq - n ** 2, 2 * m * n, m_sq + n ** 2
        #    if b < a: a, b = b, a
        #    heapq.heappush(heap, ((c, b, a), (c, b, a), True))
    print(f"perfect count = {perfect_cnt}")
    return

    """
    def isSquare(num: int) -> bool:
        rt = isqrt(num)
        if rt * rt == num: return True
    tot_cnt = 0
    perfect_cnt = 0
    for tri in pythagoreanTripleGeneratorByHypotenuse(primitive_only=True, max_hypotenuse=max_hypotenuse):
        tot_cnt += 1
        if isSquare(tri[0][2]):
            print(tri)
            perfect_cnt += 1
            yield tri[0]
    print(f"perfect count = {perfect_cnt} of {tot_cnt}")
    return
    """

def nonSuperPerfectPerfectRightAngledTriangleCount(max_hypotenuse: int=10 ** 16) -> int:
    """
    Solution to Project Euler #218
    """
    # Note that it can be proved that there are no perfect right angled triangles
    # that are not also super-perfect, and therefore for any max_hypotenuse the
    # solution is 0
    mults = [6, 28]
    m = 1
    for num in mults:
        m = lcm(m, num)
    m <<= 1

    res = 0
    perfect_cnt = 0
    for tri in perfectRightAngledTriangleGenerator(max_hypotenuse=max_hypotenuse):
        #print(tri)
        res += ((tri[0] * tri[1]) % m)
    return res

# Problem 219
def prefixFreeCodeMinimumTotalSkewCost(n_words: int=10 ** 9, cost1: int=1, cost2: int=4) -> int:
    """
    Solution to Project Euler #219
    """
    # Using Huffman enconding
    if n_words < 1: return 0
    elif n_words == 1: return min(cost1, cost2)

    remain = n_words - 2
    cost_sm = cost1 + cost2
    g = gcd(cost1, cost2)
    curr = {cost1: 1}
    curr[cost2] = curr.get(cost2, 0) + 1
    res = cost_sm
    for c in itertools.count(min(cost1, cost2), step=g):
        f = curr.pop(c, 0)
        if not f: continue
        if remain < f:
            res += (cost_sm + c) * remain
            break
        res += (cost_sm + c) * f
        remain -= f
        for add in (cost1, cost2):
            c2 = c + add
            curr[c2] = curr.get(c2, 0) + f
    return res

# Problem 220
def heighwayDragon(order: int=50, n_steps: int=10 ** 12, init_pos: Tuple[int, int]=(0, 0), init_direct: Tuple[int, int]=(0, 1), initial_str: str="Fa", recursive_strs: Dict[str, str]={"a": "aRbFR", "b": "LFaLb"}) -> Optional[Tuple[int, int]]:
    """
    Solution to Project Euler #220
    """
    if n_steps <= 0: return init_pos
    direct_dict = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    if init_direct not in direct_dict.keys(): raise ValueError("The initial direction given is not valid")
    state0 = (init_pos, direct_dict[init_direct])

    basic_effects = {"L": (((0, 0), 1), 0), "R": (((0, 0), -1), 0), "F": (((1, 0), 0), 1)}
    recursive_effects = [{l: (((0, 0), 0), 0) for l in recursive_strs.keys()}]

    def applyCharacter(curr_state: Tuple[Tuple[int, int], int], l: str, curr_order: int) -> Tuple[Tuple[int, int], int, int]:
        effect, n_steps = basic_effects[l] if l in basic_effects.keys() else recursive_effects[curr_order][l]
        #print(l, effect)
        new_direct = (curr_state[1] + effect[1]) % 4
        if effect[0] == (0, 0): return ((curr_state[0], new_direct), n_steps)
        if curr_state[1] == 0:
            pos_effect = effect[0]
        elif curr_state[1] == 1:
            pos_effect = (-effect[0][1], effect[0][0])
        elif curr_state[1] == 2:
            pos_effect = (-effect[0][0], -effect[0][1])
        elif curr_state[1] == 3:
            pos_effect = (effect[0][1], -effect[0][0])
        return ((tuple(x + y for x, y in zip(curr_state[0], pos_effect)), new_direct), n_steps)

    for ordr in range(1, order + 1):
        recursive_effects.append({})
        for l, s in recursive_strs.items():
            ordr2 = ordr - 1
            state = ((0, 0), 0)
            tot_steps = 0
            for l2 in s:
                state, add_steps = applyCharacter(state, l2, ordr2)
                tot_steps += add_steps
            recursive_effects[-1][l] = (state, tot_steps)
    #print(recursive_effects)
    remain_steps = n_steps
    ordr = order
    state = state0

    for l in initial_str:
        state2, add_steps = applyCharacter(state, l, ordr)
        if add_steps == remain_steps: return state2[0]
        elif add_steps > remain_steps: break
        state = state2
        remain_steps -= add_steps
    else: return None
    
    while ordr > 0:
        ordr -= 1
        for l2 in recursive_strs[l]:
            state2, add_steps = applyCharacter(state, l2, ordr)
            if add_steps == remain_steps: return state2[0]
            elif add_steps > remain_steps:
                l = l2
                break
            state = state2
            remain_steps -= add_steps
    return None

# Problem 121
def calculatePrimeFactorisation(num: int) -> Dict[int, int]:
    """
    For a strictly positive integer, calculates its prime
    factorisation.

    This is performed using direct division.

    Args:
        Required positional:
        num (int): The strictly positive integer whose prime
                factorisation is to be calculated.
    
    Returns:
    Dictionary (dict) giving the prime factorisation of num, whose
    keys are strictly positive integers (int) giving the prime
    numbers that appear in the prime factorisation of num, with the
    corresponding value being a strictly positive integer (int)
    giving the number of times that prime appears in the
    factorisation (i.e. the power of that prime in the prime
    factorisation of the factor num). An empty dictionary is
    returned if and only if num is the multiplicative identity
    (i.e. 1).
    """
    exp = 0
    while not num & 1:
        num >>= 1
        exp += 1
    res = {2: exp} if exp else {}
    for p in range(3, num, 2):
        if p ** 2 > num: break
        exp = 0
        while not num % p:
            num //= p
            exp += 1
        if exp: res[p] = exp
    if num > 1:
        res[num] = 1
    return res

def calculateFactorsUpToMax(num: int, fact_max: Optional[int]) -> Set[int]:
    pf = calculatePrimeFactorisation(num)
    #print(num, pf)
    if fact_max is None: fact_max = num
    curr = {1}
    for p, f in pf.items():
        prev = set(curr)
        for m in prev:
            m2 = m
            for i in range(f + 1):
                if m2 > fact_max: break
                curr.add(m2)
                m2 *= p
    return curr


def alexandrianIntegerGenerator() -> Generator[int, None, None]:
    h = []
    cnt = 0
    for m in itertools.count(1):
        #print(f"m = {m}, count = {cnt}")
        m2 = m ** 2 + 1
        mn = 2 * m * m2
        while h and h[0] <= mn:
            cnt += 1
            yield heapq.heappop(h)
        #print(f"heap size = {len(h)}")
        d_set = calculateFactorsUpToMax(m2, fact_max=m)
        #print(m2, d_set)
        for d in d_set:
            heapq.heappush(h, m * (m + d) * (m + m2 // d))
        #for d in range(1, m + 1):
        #    d2, r = divmod(m2, d)
        #    if r: continue
        #    heapq.heappush(h, m * (m + d) * (m + d2))
    return

def nthAlexandrianInteger(n: int=15 * 10 ** 4) -> int:
    """
    Solution to Project Euler #221
    """
    it = iter(alexandrianIntegerGenerator())
    for _ in range(n):
        num = next(it)
    return num

# Problem 222
def shortestSpherePackingInTube(tube_radius: int=50, radii: List[int]=list(range(30, 51))) -> float:
    """
    Solution to Project Euler #222
    """
    n = len(radii)
    if not n: return 0.

    radii.sort()
    if radii[0] * 2 < tube_radius or (n > 1 and radii[1] * 2 <= tube_radius) or radii[-1] > tube_radius:
        raise ValueError("Radii must be no less than half of tube_diameter and "
            "no more than tube_diameter")
    d = tube_radius * 2
    
    if n == 1: return 2 * radii[0]

    def packingAddDistance(r1: int, r2: int) -> float:
        return math.sqrt(d * (2 * (r1 + r2) - d))

    res = packingAddDistance(radii[0], radii[1])

    for i in range(2, n):
        res += packingAddDistance(radii[i], radii[i - 2])
    return res + radii[-1] + radii[-2]

# Problem 223
def barelyAcuteIntegerSidedTrianglesAscendingPerimeterGenerator(max_perim: Optional[int]=None) -> Generator[Tuple[int, int, int], None, None]:

    if max_perim is None: max_perim = float("inf")
    if max_perim < 3: return

    A = [[1, -2, 2], [2, -1, 2], [2, -2, 3]]
    B = [[1, 2, 2], [2, 1, 2], [2, 2, 3]]
    C = [[-1, 2, 2], [-2, 1, 2], [-2, 2, 3]]

    def matrixMultiplyVector(M: List[List[int]], v: List[int]) -> List[int]:
        return [sum(x * y for x, y in zip(row, v)) for row in M]

    #seen = {(1, 0, 0), (1, 1, 1)}
    h = [[1, (1, 0, 0)], [3, (1, 1, 1)]]
    while h:
        triple = heapq.heappop(h)[1]
        #seen.remove(triple)
        if min(triple) > 0: yield triple
        seen = set()
        for M in (A, B, C):
            triple2 = tuple(sorted(matrixMultiplyVector(M, triple)))
            if triple2[0] <= 0:
                continue
            elif triple2 in seen:
                print(f"repeat: {triple2}")
                continue
            seen.add(triple2)
            perim = sum(triple2)
            if perim > max_perim: continue
            heapq.heappush(h, [perim, triple2])
            """
            triple2 = tuple(sorted(matrixMultiplyVector(M, triple)))
            triple3 = tuple(sorted(matrixMultiplyVector(M, [triple[1], triple[0], triple[2]])))
            for t in (triple2, triple3):
                if t[0] <= 0 or t in seen:
                    continue
                seen.add(t)
                heapq.heappush(h, [sum(t), t])
            """
    return

def countBarelyAcuteIntegerSidedTrianglesUpToMaxPerimeter(max_perimeter: int=25 * 10 ** 6) -> int:
    """
    Solution to Project Euler #223
    """
    # Review- give proof that this approach works

    # Review- look into method using factorisation (a - 1)(a + 1) = (c - b)(c + 1)
    """
    #it = barelyAcuteIntegerSidedTrianglesAscendingPerimeterGenerator(max_perim=max_perimeter)
    print_intvl = 10 ** 4
    nxt_perim = print_intvl
    for i, triple in enumerate(barelyAcuteIntegerSidedTrianglesAscendingPerimeterGenerator(max_perim=max_perimeter), start=1):
        perim = sum(triple)
        if perim > nxt_perim:
            print(triple, perim, i)
            nxt_perim += print_intvl
    return i
    """
    if max_perimeter < 3: return

    A = [[1, -2, 2], [2, -1, 2], [2, -2, 3]]
    B = [[1, 2, 2], [2, 1, 2], [2, 2, 3]]
    C = [[-1, 2, 2], [-2, 1, 2], [-2, 2, 3]]

    def matrixMultiplyVector(M: List[List[int]], v: List[int]) -> List[int]:
        return [sum(x * y for x, y in zip(row, v)) for row in M]

    #seen = {(1, 0, 0), (1, 1, 1)}
    #h = [[1, (1, 0, 0)], [3, (1, 1, 1)]]
    stk = [(1, 0, 0), (1, 1, 1)]
    res = 1
    while stk:
        triple = stk.pop()
        seen = set()
        for M in (A, B, C):
            triple2 = matrixMultiplyVector(M, triple)
            triple3 = tuple(sorted(triple2))
            if triple3[0] <= 0: continue
            elif triple3 in seen:
                print(f"repeat: {triple2}")
                continue
            seen.add(triple3)
            perim = sum(triple2)
            if perim > max_perimeter: continue
            stk.append(triple2)
            res += 1
    return res

# Problem 224
def barelyObtuseIntegerSidedTrianglesAscendingPerimeterGenerator(max_perim: Optional[int]=None) -> Generator[Tuple[int, int, int], None, None]:

    
    if max_perim is None: max_perim = float("inf")
    if max_perim < 3: return

    A = [[1, -2, 2], [2, -1, 2], [2, -2, 3]]
    B = [[1, 2, 2], [2, 1, 2], [2, 2, 3]]
    C = [[-1, 2, 2], [-2, 1, 2], [-2, 2, 3]]

    def matrixMultiplyVector(M: List[List[int]], v: List[int]) -> List[int]:
        return [sum(x * y for x, y in zip(row, v)) for row in M]

    #seen = {(1, 0, 0), (1, 1, 1)}
    h = [[1, (0, 0, 1)]]
    while h:
        triple = heapq.heappop(h)[1]
        #seen.remove(triple)
        if min(triple) > 0: yield triple
        seen = set()
        for M in (A, B, C):
            triple2 = tuple(sorted(matrixMultiplyVector(M, triple)))
            if triple2[0] <= 0:
                continue
            elif triple2 in seen:
                print(f"repeat: {triple2}")
                continue
            seen.add(triple2)
            perim = sum(triple2)
            if perim > max_perim: continue
            heapq.heappush(h, [perim, triple2])
            """
            triple2 = tuple(sorted(matrixMultiplyVector(M, triple)))
            triple3 = tuple(sorted(matrixMultiplyVector(M, [triple[1], triple[0], triple[2]])))
            for t in (triple2, triple3):
                if t[0] <= 0 or t in seen:
                    continue
                seen.add(t)
                heapq.heappush(h, [sum(t), t])
            """
    return

def countBarelyObtuseIntegerSidedTrianglesUpToMaxPerimeter(max_perimeter: int=75 * 10 ** 6) -> int:
    """
    Solution to Project Euler #224
    """
    # Review- give proof that this approach works including justification
    # of the initial values

    """
    print_intvl = 10 ** 4
    nxt_perim = print_intvl
    for i, triple in enumerate(barelyObtuseIntegerSidedTrianglesAscendingPerimeterGenerator(max_perim=max_perimeter), start=1):
        perim = sum(triple)
        if perim > nxt_perim:
            print(triple, perim, i)
            nxt_perim += print_intvl
    return i
    """
    if max_perimeter < 3: return

    A = [[1, -2, 2], [2, -1, 2], [2, -2, 3]]
    B = [[1, 2, 2], [2, 1, 2], [2, 2, 3]]
    C = [[-1, 2, 2], [-2, 1, 2], [-2, 2, 3]]

    def matrixMultiplyVector(M: List[List[int]], v: List[int]) -> List[int]:
        return [sum(x * y for x, y in zip(row, v)) for row in M]

    #seen = {(1, 0, 0), (1, 1, 1)}
    #h = [[1, (1, 0, 0)], [3, (1, 1, 1)]]
    stk = [(0, 0, 1)]
    res = 0
    while stk:
        triple = stk.pop()
        seen = set()
        for M in (A, B, C):
            triple2 = matrixMultiplyVector(M, triple)
            triple3 = tuple(sorted(triple2))
            if triple3[0] <= 0: continue
            elif triple3 in seen:
                print(f"repeat: {triple2}")
                continue
            seen.add(triple3)
            perim = sum(triple2)
            if perim > max_perimeter: continue
            stk.append(triple2)
            res += 1
    return res    

# Problem 225
def tribonacciOddNonDivisorGenerator(init_terms: Tuple[int, int, int]=(1, 1, 1)) -> Generator[int, None, None]:

    #Trie = lambda: defaultdict(Trie)
    ref = list(init_terms)
    for num in itertools.count(3, step=2):
        #seen_triples = Trie()
        curr = [x % num for x in init_terms]
        #t = seen_triples
        #for m in curr:
        #    t = t[m]
        #t[True] = True
        while True:
            curr = [curr[1], curr[2], sum(curr) % num]
            if not curr[-1]:
                break
            if curr == ref:
                yield num
                break
            #t = seen_triples
            #for m in curr:
            #    t = t[m]
            #if True in t.keys():
            #    yield num
            #    break
            #t[True] = True
    return

def nthSmallestTribonacciOddNonDivisors(odd_non_divisor_number: int=124, init_terms: Tuple[int, int, int]=(1, 1, 1)) -> int:
    """
    Solution to Project Euler #225
    """
    it = iter(tribonacciOddNonDivisorGenerator(init_terms=init_terms))
    num = -1
    for i in range(odd_non_divisor_number):
        num = next(it)
        #print(i + 1, num)
    return num

# Problem 226
def findBlacmangeValue(x: CustomFraction) -> CustomFraction:
    x -= x.numerator // x.denominator
    if x.denominator < 2 * x.numerator:
        x = 1 - x
    a0, b0 = 0, CustomFraction(1, 1)
    while x != 0 and not x.denominator & 1:
        a0 += b0 * x
        b0 /= 2
        x.denominator >>= 1
        if x.denominator < 2 * x.numerator:
            x = 1 - x
    if x == 0: return a0
    a, b = x, CustomFraction(1, 2)
    x2 = 2 * x
    if x2.denominator < 2 * x2.numerator:
        x2 = 1 - x2
    while x2 != x and x2 != 0:
        #print(x2, x)
        a += b * x2
        b /= 2
        x2 = 2 * x2
        if x2.denominator < 2 * x2.numerator:
            x2 = 1 - x2
    
    res = a if x2 == 0 else a / (1 - b)
    #print(a0, b0, res)
    return a0 + b0 * res
    #return res

def findBlacmangeIntegralValue(x: CustomFraction) -> CustomFraction:
    q = CustomFraction(x.numerator // x.denominator, 1)
    x -= q
    a0, b0 = 0, CustomFraction(1, 1)
    if x.denominator < 2 * x.numerator:
        x = 1 - x
        a0 += b0 / 2
        b0 = -b0
    a0, b0 = 0, CustomFraction(1, 1)
    while x != 0 and not x.denominator & 1:
        a0 += b0 * x * x / 2
        b0 /= 4
        x.denominator >>= 1
        if x.denominator < 2 * x.numerator:
            x = 1 - x
            a0 += b0 / 2
            b0 = -b0
    if x == 0: return a0 + q / 2
    a, b = x * x / 2, CustomFraction(1, 4)
    x2 = 2 * x
    if x2.denominator < 2 * x2.numerator:
        x2 = 1 - x2
        a += b / 2
        b = -b
    while x2 != x and x2 != 0:
        #print(x2, x)
        a += b * x2 * x2 / 2
        b /= 4
        x2 = 2 * x2
        if x2.denominator < 2 * x2.numerator:
            x2 = 1 - x2
            a += b / 2
            b = -b
    
    res = a if x2 == 0 else a / (1 - b)
    #print(a0, b0, res)
    return a0 + b0 * res + q / 2

def blacmangeCircleIntersectionArea(eps: float=10 ** -9) -> float:

    # Review- try to generalise to any circle

    # Rightmost intersection point is at (1 / 2, 1 / 2)
    # Find leftmost intersection point
    centre = (CustomFraction(1, 4), CustomFraction(1, 2))
    rad = CustomFraction(1, 4)
    rad_sq = rad * rad

    lft, rgt = CustomFraction(0, 1), CustomFraction(1, 4)
    while rgt - lft >= eps:
        x = lft + (rgt - lft) / 2
        y = findBlacmangeValue(x)
        v = (x - centre[0], y - centre[1])
        d_sq = v[0] * v[0] + v[1] * v[1]
        if d_sq == rad_sq:
            print("exact found")
            break
        elif d_sq > rad_sq:
            lft = x
        else: rgt = x
    else:
        x = lft + (rgt - lft) / 2
        y = findBlacmangeValue(x)
        v = (x - centre[0], y - centre[1])
        d_sq = v[0] * v[0] + v[1] * v[1]
        print(d_sq, d_sq.numerator / d_sq.denominator)
    #print(lft, rgt)
    #print((x, y), (x.numerator / x.denominator, y.numerator / y.denominator))
    b_area = findBlacmangeIntegralValue(CustomFraction(1, 2)) - findBlacmangeIntegralValue(x)
    x2 = CustomFraction(1, 2)
    y2 = findBlacmangeValue(x2)
    trap_area = (y + y2) * (x2 - x) / 2
    area1 = (b_area - trap_area)
    area1 = area1.numerator / area1.denominator
    angle = math.acos((v[0].numerator / v[0].denominator) * (rad.denominator / rad.numerator))
    #print(area1)
    #print(angle * 180 / math.pi)
    res = area1 + .5 * (angle - math.sin(angle)) * (rad_sq.numerator / rad_sq.denominator)
    return res


# Problem 227
def chaseGameExpectedNumberOfTurns(die_n_faces: int=6, n_opts_left: int=1, n_opts_right: int=1, n_players: int=100, separation_init: int=50) -> float:
    m = n_players >> 1
    n_opts_still = die_n_faces - n_opts_left - n_opts_right
    n_unchanged = (n_opts_still ** 2 + n_opts_left ** 2 + n_opts_right ** 2)
    n_shift1 = (n_opts_left + n_opts_right) * n_opts_still
    n_shift2 = n_opts_left * n_opts_right
    #print(n_unchanged, n_shift1, n_shift2, n_unchanged + 2 * (n_shift1 + n_shift2), die_n_faces ** 2)
    T = np.zeros([m, m])
    for i in range(m):
        T[i, i] = n_unchanged
    T[0, 0] += n_shift2
    for i in range(1, m):
        T[i, i - 1] += n_shift1
    for i in range(2, m):
        T[i, i - 2] += n_shift2
    for i in range(m):
        #print(i)
        j1 = min(i + 1, n_players - i - 3)
        j2 = min(i + 2, n_players - i - 4)
        #print(i, j1, j2)
        T[i, j1] += n_shift1
        T[i, j2] += n_shift2
    T /= die_n_faces ** 2
    #print(T)
    eig_vals, eig_vecs = np.linalg.eig(T)
    #print(eig_vals)
    #print(eig_vals)
    #print(eig_vecs)
    P = np.zeros([m, m])
    D = np.zeros([m, m])
    D2 = np.zeros([m, m])
    for i in range(m):
        D[i, i] = eig_vals[i]
        D2[i, i] = 1 / (1 - eig_vals[i])
        P[i] = eig_vecs[i]
    P = P.transpose()
    P_inv = np.linalg.inv(P)
    #print(P, P_inv, D)
    #print(np.matmul(P, P_inv))
    #print(np.dot(eig_vecs[0], eig_vecs[1]))
    #print(P)
    #print(D2)
    M = np.matmul(P_inv, np.matmul(D2, P))
    #print(T)
    #print(M)
    v = np.zeros([m])
    i = min(separation_init, n_players - separation_init) - 1
    #print(f"i = {i}")
    v[i] = 1
    #print(v)
    v2 = np.matmul(M, v)
    #print(v2)
    res = sum(v2)
    #print(res2)
    return res

# Problem 228
def convexPolygonsAroundOriginMinkowskiSum(poly1: List[Tuple[float, Tuple[float, float]]], poly2: List[Tuple[float, Tuple[float, float]]]) -> List[Tuple[float, float]]:
    poly1.sort()
    poly2.sort()
    if poly1[0][0] > poly2[0][0]: poly1, poly2 = poly2, poly1
    n1, n2 = len(poly1), len(poly2)
    i1 = 0
    v_lst = []
    for i2 in range(n2):
        for i1 in range(i1, n1 - 1):
            if poly1[i1 + 1][0] >= poly2[i2][0]:
                break
        else: break
        v_lst.append(tuple(x + y for x, y in zip(poly1[i1][1], poly2[i2][1])))
        v_lst.append(tuple(x + y for x, y in zip(poly1[i1 + 1][1], poly2[i2][1])))
    for i2 in range(i2, n2):
        v_lst.append(tuple(x + y for x, y in zip(poly1[-1][1], poly2[i2][1])))
        v_lst.append(tuple(x + y for x, y in zip(poly1[0][1], poly2[i2][1])))
    
    return grahamScan(v_lst, include_border_points=True)

def regularPolygonMinkowskiSum(vertex_counts: List[int]) -> List[Tuple[float, float]]:

    poly_lst = []
    for cnt in vertex_counts:
        lst = []
        for i in range(cnt):
            angle = (2 * i + 1) * math.pi / cnt
            lst.append((angle, (math.cos(angle), math.sin(angle))))
        poly_lst.append(lst)
    
    curr = poly_lst[0]
    for i in range(1, len(poly_lst)):
        print(i, len(curr))
        lst = convexPolygonsAroundOriginMinkowskiSum(curr, poly_lst[i])
        curr = []
        for v in lst:
            angle = math.atan2(v[1], v[0])
            if angle < 0: angle += 2 * math.pi
            curr.append((angle, v))
    return [x[1] for x in curr]

    """
    while len(poly_lst) > 1:
        print(len(poly_lst))
        print([len(x) for x in poly_lst])
        prev = poly_lst
        poly_lst = []
        for i in range(0, len(prev) - 1, 2):
            lst = convexPolygonsAroundOriginMinkowskiSum(prev[i], prev[i + 1])
            #print(lst)
            lst2 = []
            for v in lst:
                angle = math.atan2(v[1], v[0])
                if angle < 0: angle += 2 * math.pi
                lst2.append((angle, v))
            poly_lst.append(lst2)
        if len(prev) & 1:
            poly_lst.append(prev[-1])
    return [x[1] for x in poly_lst[0]]
    """

def regularPolygonMinkowskiSumSideCount(vertex_counts: List[int]=list(range(1864, 1910))) -> int:

    #res = regularPolygonMinkowskiSum(vertex_counts)
    #for v in res:
    #    print(v[0], v[1])
    #return res
    mn, mx = min(vertex_counts), max(vertex_counts)
    rng = mx - mn
    
    res = sum(vertex_counts) - (len(vertex_counts) - 1)
    for num in range(2, rng + 1):
        d_cnt = 0
        for v_cnt in vertex_counts:
            d_cnt += (not v_cnt % num)
        if d_cnt == 1: continue
        euler_tot = 1
        for num2 in range(2, num):
            euler_tot += (gcd(num2, num) == 1)
        res -= euler_tot * (d_cnt - 1)
    return res

# Problem 229
def fourRepresentationsUsingSquaresCount(mults: Tuple[int]=(1, 2, 3, 7), num_max: int=2 * 10 ** 9) -> int:
    part_size = 10 ** 8
    n_mults = len(mults)
    mults = sorted(set(mults))
    bm_target = (1 << n_mults) - 1
    res = 0
    md168_set = set()
    sq_cnt = 0
    for part_start in range(0, num_max + 1, part_size):
        print(part_start, res)
        part_end = min(part_start + part_size - 1, num_max)
        bm_sieve = [0] * (part_end - part_start + 1)
        part_end_sqrt = isqrt(part_end)
        for a in range(1, part_end_sqrt + 1):
            a_sq = a ** 2
            b_mn = max(1, isqrt((part_start - a_sq - 1) // mults[-1]) + 1 if a_sq < part_start else 0)
            b_mx = isqrt(num_max - a_sq)
            #if not a % 100: print(f"a = {a} of {part_end_sqrt}, b_max = {b_mx}")
            j_mn = len(mults) - 1
            j_mx = len(mults)
            for b in range(b_mn, b_mx + 1):
                b_sq = b ** 2
                #print(a, b)
                for j_mn in reversed(range(1, j_mn + 1)):
                    if a_sq + mults[j_mn - 1] * b_sq < part_start: break
                else: j_mn = 0
                
                for j in range(j_mn, j_mx):
                    m = mults[j]
                    num = a_sq + m * b_sq
                    if num > part_end:
                        j_mx = j
                        break
                    bm2 = 1 << j
                    num2 = num - part_start
                    #print(num, part_start, num2, len(bm_sieve))
                    if bm_sieve[num2] & bm2: continue
                    bm_sieve[num2] |= bm2
                    if (bm_sieve[num2] == bm_target):
                        res += 1
                        if isqrt(num) ** 2 == num:
                            #print(num)
                            sq_cnt += 1
                        #sq_cnt += (isqrt(num) ** 2 == num)
                        md168_set.add(num % 168)
    print(sorted(md168_set))
    print(f"square count = {sq_cnt}")
    return res

def fourSquaresRepresentationCountSpecialised(num_max: int=2 * 10 ** 9) -> int:

    # Try to generalise to arbitrary k using the Legendre/Jacobi symbol
    # and quadratic reciprocity. Possibly in first instance, restricting
    # to either prime or unit values of k allowed.

    ps = SimplePrimeSieve()
    def primeCheck(num: int) -> bool:
        return ps.millerRabinPrimalityTestWithKnownBounds(num, max_n_additional_trials_if_above_max=10)[0]

    p_prods = SortedSet()
    res = 0
    for num in range(0, num_max, 168):
        for r in (1, 25, 121):
            p = num + r
            b = primeCheck(p)
            #print(p, b)
            if not b: continue
            add_set = {p}
            res += isqrt(num_max // p)
            for p_prod in p_prods:
                num2 = p * p_prod
                if num2 > num_max: break
                res += isqrt(num_max // num2)
                add_set.add(num2)
            for num2 in add_set:
                p_prods.add(num2)
            #print(p, res)
    #print(res)
    for num in range(1, isqrt(num_max) + 1):
        pf = calculatePrimeFactorisation(num)
        remain_criterion = {0, 1, 2, 3}
        if 2 in pf.keys():
            remain_criterion.remove(2)
            if pf[2] >= 2: remain_criterion.remove(3)
        for p in pf.keys() - {2}:
            if 0 in remain_criterion and p % 4 == 1:
                remain_criterion.remove(0)
                if not remain_criterion:
                    break
            if 1 in remain_criterion and p % 8 in {1, 3}:
                remain_criterion.remove(1)
                if not remain_criterion: break
            if 2 in remain_criterion and p % 3 == 1:
                remain_criterion.remove(2)
                if not remain_criterion: break
            if 3 in remain_criterion and p % 7 in {1, 2, 4}:
                remain_criterion.remove(3)
                if not remain_criterion: break
        else:
            continue
        #print(num, num ** 2)
        res += 1
    return res

# Problem 230
def fibonacciWordsSum(
    A: int=1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679,
    B: int=8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196,
    poly_coeffs: Tuple[int]=(127, 19),
    exp_base: int=7,
    n_max: int=17,
    base: int=10
) -> int:
    
    def evaluateTermNumber(n: int) -> int:
        res = 0
        for c in reversed(poly_coeffs):
            res = res * n + c
        return res * exp_base ** n

    mx_term = 0
    for n in range(0, n_max + 1):
        mx_term = max(mx_term, evaluateTermNumber(n))
    
    A_digs = []
    A2 = A
    while A2:
        A2, r = divmod(A2, base)
        A_digs.append(r)
    A_digs = A_digs[::-1]
    B_digs = []
    B2 = B
    while B2:
        B2, r = divmod(B2, base)
        B_digs.append(r)
    B_digs = B_digs[::-1]

    A_len = len(A_digs)
    B_len = len(B_digs)
    len_lst = [B_len, A_len + B_len]
    while len_lst[-1] < mx_term:
        len_lst.append(len_lst[-2] + len_lst[-1])
    
    #print(mx_term)
    #print(len_lst)
    
    def findDigit(term: int) -> int:
        #print(f"term = {term}")
        i = term - 1
        if i <= A_len:
            return A_digs[i]
        j = bisect.bisect_right(len_lst, i)
        while j > 1:
            #print(j, i, len_lst[j])
            if i >= len_lst[j - 2]:
                i -= len_lst[j - 2]
                j -= 1
            else: j -= 2
        if j == 1:
            #print(i)
            return B_digs[i - A_len] if i >= A_len else A_digs[i]
        return B_digs[i]

    res = 0
    for n in reversed(range(0, n_max + 1)):
        res = res * base + findDigit(evaluateTermNumber(n))
    return res


# Problem 231
def binomialCoefficientPrimeFactorisation(n: int, k: int) -> Dict[int, int]:

    ps = SimplePrimeSieve(n)
    k2 = n - k
    res = {}
    if n < 0 or k < 0 or k2 < 0:
        return res
    for p in ps.p_lst:
        cnt = 0
        n2 = n
        while n2:
            n2 //= p
            cnt += n2
        for num in (k, k2):
            while num:
                num //= p
                cnt -= num
        res[p] = cnt
    return res

def binomialCoefficientPrimeFactorisationSum(n: int=20 * 10 ** 6, k: int=15 * 10 ** 6) -> int:
    """
    Solution to Project Euler #231
    """

    res = sum(p * f for p, f in binomialCoefficientPrimeFactorisation(n, k).items())
    return res

# Problem 232
def probabilityPlayer2WinsFractionGivenSuccessProbabilitiesAndScores(
    points_required: int,
    player1_success_prob: CustomFraction=CustomFraction(1, 2),
    player1_success_points: int=1,
    player2_success_prob: CustomFraction=CustomFraction(1, 2),
    player2_success_points: int=1,
) -> CustomFraction:
    # Used in related problem where the value of T for Player 2 cannot be
    # changed
    memo = {}  
    def recur(player1_remain: int, player2_remain: int) -> CustomFraction:
        if player1_remain <= 0: return CustomFraction(0, 1)
        elif player2_remain <= 0: return CustomFraction(1, 1)
        
        args = (player1_remain, player2_remain)
        if args in memo.keys(): return memo[args]

        # At least one player succeeds on this turn
        #print("hi")
        res = player1_success_prob * (player2_success_prob * recur(player1_remain - player1_success_points, player2_remain - player2_success_points) +\
                                       (1 - player2_success_prob) * recur(player1_remain - player1_success_points, player2_remain)) +\
                (1 - player1_success_prob) *  player2_success_prob * recur(player1_remain, player2_remain - player2_success_points)

        unchanged_prob = (1 - player1_success_prob) * (1 - player2_success_prob)
        res /= (1 - unchanged_prob)

        memo[args] = res
        #print("hi2", memo)
        return res
    #print(player2_success_points, memo)
    res = recur(points_required, points_required)
    print(player2_success_points, memo)
    return res

def probabilityPlayer2WinsFraction(points_required: int=100) -> CustomFraction:
    res = 0
    player1_success_prob = CustomFraction(1, 2)
    player1_success_points = 1

    memo = {}  
    def recur(player1_remain: int, player2_remain: int) -> CustomFraction:
        if player2_remain <= 0: return CustomFraction(1, 1)
        if player1_remain <= 0: return CustomFraction(0, 1)
        
        args = (player1_remain, player2_remain)
        if args in memo.keys(): return memo[args]

        # At least one player succeeds on this turn
        #print("hi")
        res = 0
        for T in itertools.count(1):
            player2_success_prob = CustomFraction(1, 2 ** T)
            player2_success_points = 2 ** (T - 1)
            ans = player1_success_prob * (player2_success_prob * recur(player1_remain - player1_success_points, player2_remain - player2_success_points) +\
                                        (1 - player2_success_prob) * recur(player1_remain - player1_success_points, player2_remain)) +\
                    (1 - player1_success_prob) *  player2_success_prob * recur(player1_remain, player2_remain - player2_success_points)

            unchanged_prob = (1 - player1_success_prob) * (1 - player2_success_prob)
            ans /= (1 - unchanged_prob)
            res = max(res, ans)
            if player2_success_points >= player2_remain: break

        memo[args] = res
        #print("hi2", memo)
        return res
    #print(player2_success_points, memo)

    # Transpose problem into one where each turn player 2 goes first, so we
    # can avoid at each step accounting for player 2 responding to the outcome
    # of player 1's turn
    res = player1_success_prob * recur(points_required - 1, points_required) + (1 - player1_success_prob) * recur(points_required, points_required)
    #print(player2_success_points, memo)
    #print(memo)
    return res

    """
    for T in itertools.count(1):
        player2_success_prob = CustomFraction(1, 2 ** T)
        player2_success_points = 2 ** (T - 1)
        frac = probabilityPlayer2WinsFractionGivenSuccessProbabilitiesAndScores(
            points_required=points_required,
            player1_success_prob=player1_success_prob,
            player1_success_points=player1_success_points,
            player2_success_prob=player2_success_prob,
            player2_success_points=player2_success_points,
        )
        print(T, frac.numerator / frac.denominator)
        res = max(res, frac)
        if player2_success_points >= points_required: break
    return res
    """

def probabilityPlayer2WinsFloat(points_required: int=100) -> CustomFraction:
    """
    Solution to Project Euler #132
    """
    res = probabilityPlayer2WinsFraction(points_required=points_required)
    #print(res)
    return res.numerator / res.denominator


# Problem 233
def factorisationsGenerator(num: int) -> Generator[Dict[int, int], None, None]:
    pf = calculatePrimeFactorisation(num)
    print(pf)
    p_lst = sorted(pf.keys())
    n_p = len(p_lst)
    f_lst = [pf[p] for p in p_lst]
    z_cnt = [0]

    print(p_lst, f_lst)

    curr = {}
    def recur(idx: int, num: int, prev: int=2) -> Generator[Dict[int, int], None, None]:
        #print(idx, num, prev, curr)
        if idx == n_p:
            #print("hi2")
            if num < prev: return
            #print(idx, num, prev, f_lst, curr, z_cnt[0])
            curr[num] = curr.get(num, 0) + 1
            if z_cnt[0] == n_p:
                yield dict(curr)
            else:
                yield from recur(0, 1, prev=num)
            curr[num] -= 1
            if not curr[num]: curr.pop(num)
            return
        f0 = f_lst[idx]
        num0 = num
        #print(f"idx = {idx}, f0 = {f0}, f_lst = {f_lst}, z_cnt = {z_cnt}, curr = {curr}")
        if not f0:
            yield from recur(idx + 1, num, prev=prev)
            return
        for i in range(f0):
            #print(f"i = {i}")
            yield from recur(idx + 1, num, prev=prev)
            #print(f"i = {i}, idx = {idx}, f_lst[idx] = {f_lst[idx]}")
            num *= p_lst[idx]
            f_lst[idx] -= 1
        #print("finished loop")
        z_cnt[0] += 1
        #print(idx, num, prev)
        yield from recur(idx + 1, num, prev=prev)
        num = num0
        z_cnt[0] -= 1
        f_lst[idx] = f0
        return
    
    yield from recur(0, 1, prev=2)
    return

def circleInscribedSquareSideLengthWithLatticePointCount(n_lattice_points: int=420, max_inscribed_square_side_length: int=10 ** 11) -> int:
    """
    Solution to Project Euler #233
    """
    q, r = divmod(n_lattice_points, 8)
    if r != 4: return 0
    
    target = (q << 1) + 1
    print(target)
    p_pow_opts = []
    mx_n_p = 0
    for fact in factorisationsGenerator(target):
        #print(fact)
        p_pows = {}
        for num, f in fact.items():
            if not num & 1: break
            num2 = num >> 1
            p_pows[num2] = f
        else:
            p_pow_opts.append(p_pows)
            mx_n_p = max(mx_n_p, len(p_pows))
    print(p_pow_opts)

    ps = SimplePrimeSieve()
    p_gen = iter(ps.endlessPrimeGenerator())
    
    p_r1_lst = []
    p_other_lst = []
    while True:
        p = next(p_gen)
        r = p & 3
        if r == 1:
            p_r1_lst.append(p)
            if len(p_r1_lst) >= mx_n_p: break
            continue
        p_other_lst.append(p)
    p_pow_opts2 = []
    print(p_r1_lst)
    mx_p_r1 = 0
    for opts in p_pow_opts:
        print(opts)
        i_p = 0#sum(opts.values()) - 1
        num = 1
        opts2 = sorted(opts.keys())
        for j in reversed(range(1, len(opts2))):
            m = opts2[j]
            f = opts[m]
            for _ in range(f):
                num *= p_r1_lst[i_p] ** m
                print(i_p, p_r1_lst[i_p], m, num)
                i_p += 1
                if num > max_inscribed_square_side_length: break
            else: continue
            break
        else:
            j = 0
            m = opts2[j]
            f = opts[m]
            for _ in range(f - 1):
                num *= p_r1_lst[i_p] ** m
                print(i_p, p_r1_lst[i_p], m, num)
                i_p += 1
                if num > max_inscribed_square_side_length: break
            else:
                mx_p_r1 = max(mx_p_r1, integerNthRoot(max_inscribed_square_side_length // num, m))
                num *= p_r1_lst[i_p] ** m
                print(i_p, p_r1_lst[i_p], m, num)
                i_p += 1
                if num > max_inscribed_square_side_length: continue
                p_pow_opts2.append((num, opts))
    
    p_pow_opts = sorted(p_pow_opts2)
    print(p_pow_opts)
    print(mx_p_r1)
    if not p_pow_opts: return 0
    mult_mx = max_inscribed_square_side_length // p_pow_opts[0][0]
    print(mult_mx)

    mults = SortedList([1])
    for p in p_other_lst:
        #mults.add(p)
        for i in itertools.count(0):
            num = mults[i] * p
            if num > mult_mx: break
            mults.add(num)
    while True:
        p = next(p_gen)
        if p & 3 == 1:
            if p <= mx_p_r1:
                p_r1_lst.append(p)
            continue
        if p > mult_mx: break
        #mults.add(p)
        for i in itertools.count(0):
            num = mults[i] * p
            if num > mult_mx: break
            mults.add(num)
    
    print(len(mults))
    print(mults[:50])
    mults_cumu = [0]
    for num in mults:
        mults_cumu.append(mults_cumu[-1] + num)

    while True:
        p = next(p_gen)
        if p > mx_p_r1: break
        if p & 3 == 1:
            p_r1_lst.append(p)
    print(len(p_r1_lst))
    #n_p_r1 = len(p_r1_lst)

    def primeProductGenerator(pow_opts: Dict[int, int], p_lst: List[int], mx: int) -> Generator[int, None, None]:
        pow_lst = []
        for num in reversed(sorted(pow_opts.keys())):
            f = pow_opts[num]
            for f_ in range(f - 1):
                pow_lst.append((num, True))
            pow_lst.append((num, False))
        n_pow = len(pow_lst)
        n_p = len(p_lst)
        print(pow_opts, pow_lst)

        curr_incl = set()
        def recur(idx: int=0, curr: int=1, mn_p_idx: int=0) -> Generator[int, None, None]:
            if idx == n_pow:
                yield curr
                return
            for p_idx in range(mn_p_idx, n_p):
                if p_idx in curr_incl: continue
                p = p_lst[p_idx]
                exp, b = pow_lst[idx]
                nxt = curr * p ** exp
                if nxt > mx: break
                curr_incl.add(p_idx)
                yield from recur(idx=idx + 1, curr=nxt, mn_p_idx=p_idx + 1 if b else 0)
                curr_incl.remove(p_idx)
            return

        yield from recur(idx=0, curr=1, mn_p_idx=0)
        return

    res = 0
    cnt = 0
    for opts in p_pow_opts:
        for num in primeProductGenerator(opts[1], p_r1_lst, max_inscribed_square_side_length):
            j = mults.bisect_right(max_inscribed_square_side_length // num)
            #print(num, max_inscribed_square_side_length // num, mults[j - 1])
            cnt += j
            res += num * mults_cumu[j]
    print(f"total count = {cnt}")
    return res

# Problem 234
def semiDivisibleNumberCount(n_max: int=999966663333) -> int:
    """
    Solution to Project Euler #234
    """
    if n_max < 4: return 0

    #sqrt_max = isqrt(n_max - 1) + 1
    #print(f"sqrt_max = {sqrt_max}")
    ps = SimplePrimeSieve()
    p_gen = iter(ps.endlessPrimeGenerator())


    def singleDivisibleCount(p1: int, p2: int, mn: int, mx: int) -> int:
        if mn > mx: return 0
        q = p1 * p2
        i1, i2 = (mn - 1) // p1, mx // p1
        j1, j2 = (mn - 1) // p2, mx // p2
        k1, k2 = (mn - 1) // q, mx // q
        ans1 = p1 * (i2 * (i2 + 1) - i1 * (i1 + 1)) >> 1
        ans2 = p2 * (j2 * (j2 + 1) - j1 * (j1 + 1)) >> 1
        ans3 = q * (k2 * (k2 + 1) - k1 * (k1 + 1))
        #print(p1, p2, ans1, ans2, ans3)
        return ans1 + ans2 - ans3

    res = 0
    p2 = next(p_gen)
    p2_sq = p2 ** 2
    while True:
        p1, p1_sq = p2, p2_sq
        p2 = next(p_gen)
        p2_sq = p2 ** 2
        if p2_sq > n_max:
            ans = singleDivisibleCount(p1, p2, p1_sq + 1, n_max)
            print(p1, p2, p1_sq + 1, n_max, ans)
            res += ans
            break
        ans = singleDivisibleCount(p1, p2, p1_sq + 1, p2_sq - 1)
        #print(p1, p2, p1_sq + 1, p2_sq - 1, ans)
        res += ans
    return res


# Problem 235
def arithmeticGeometricSeries(a: float=900, b: int=-3, n: int=5000, val: float=-6 * 10 ** 11, eps: float=10 ** -13) -> float:

    if b < 0:
        a, b, val = -a, -b, -val
    #r0 = -a / b
    print(a, b, val)

    def func(r: float, n: int) -> float:
        #print(r)
        if r == 1: return 900 * n - 1.5 * n * (n + 1)
        res = (a + b * n) * r ** (n + 1) - (a + b * (n + 1)) * r ** n - a * r + (a + b)
        return float(res) / (r - 1) ** 2

    print(f"values at 1 - 10 ** -10, 1 and 1 + 10 ** -10")
    print(func(1 - 10 ** -9, n), func(1, n), func(1 + 10 ** -11, n))

    r0 = 0
    r = 0
    diff = 1
    while True:
        r2 = r0 + diff
        y = func(r2, n)
        if y >= val: break
        r = r2
        print(r, y)
        diff *= 1.1
    print(r, y)
    diff0 = 0 if diff == 1 else diff / 2
    lft, rgt = r, r2
    while rgt - lft >= eps:
        mid = lft + (rgt - lft) * .5
        
        y = func(mid, n)
        print(mid, y)
        if y == val:
            return mid
        elif y < val: lft = mid
        else: rgt = mid
    return lft + (rgt - lft) * .5


if __name__ == "__main__":
    to_evaluate = {234}
    since0 = time.time()

    if not to_evaluate or 201 in to_evaluate:
        since = time.time()
        res = subsetsOfSquaresWithUniqueSumTotal(n_max=100, k=50)
        #res = subsetsWithUniqueSumTotal({1, 3, 6, 8, 10, 11}, 3)
        print(f"Solution to Project Euler #201 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 202 in to_evaluate:
        since = time.time()
        res = equilateralTriangleReflectionCountNumberOfWays(n_reflect=12017639147)
        print(f"Solution to Project Euler #202 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 203 in to_evaluate:
        since = time.time()
        res = distinctSquareFreeBinomialCoefficientSum(n_max=51)
        print(f"Solution to Project Euler #203 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 204 in to_evaluate:
        since = time.time()
        res = generalisedHammingNumberCount(typ=100, n_max=10 ** 9)
        print(f"Solution to Project Euler #204 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 205 in to_evaluate:
        since = time.time()
        res = probabilityDieOneSumWinsFloat(die1_face_values=(1, 2, 3, 4), n_die1=9, die2_face_values=(1, 2, 3, 4, 5, 6), n_die2=6)
        print(f"Solution to Project Euler #205 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 206 in to_evaluate:
        since = time.time()
        res = concealedSquare(pattern=[1, None, 2, None, 3, None, 4, None, 5, None, 6, None, 7, None, 8, None, 9, None, 0], base=10)
        print(f"Solution to Project Euler #206 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 207 in to_evaluate:
        since = time.time()
        res = findSmallestPartitionBelowGivenProportion(proportion=CustomFraction(1, 12345))
        print(f"Solution to Project Euler #207 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 208 in to_evaluate:
        since = time.time()
        res = robotWalks(reciprocal=5, n_steps=70)
        print(f"Solution to Project Euler #208 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 209 in to_evaluate:
        since = time.time()
        res = countZeroMappings(n_inputs=6)
        print(f"Solution to Project Euler #209 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 210 in to_evaluate:
        since = time.time()
        res = countObtuseTriangles(r=10 ** 9, div=4)
        print(f"Solution to Project Euler #210 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 211 in to_evaluate:
        since = time.time()
        res = divisorSquareSumIsSquareTotal(n_max=64 * 10 ** 6 - 1)
        print(f"Solution to Project Euler #211 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 212 in to_evaluate:
        since = time.time()
        res = laggedFibonacciCuboidUnionVolume(
            n_cuboids=50000,
            cuboid_smallest_coord_ranges=((0, 9999), (0, 9999), (0, 9999)),
            cuboid_dim_ranges=((1, 399), (1, 399), (1, 399)),
            l_fib_modulus=10 ** 6,
            l_fib_poly_coeffs=(100003, -200003, 0, 300007),
            l_fib_lags=(24, 55),
        )
        print(f"Solution to Project Euler #212 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 213 in to_evaluate:
        since = time.time()
        res = fleaCircusExpectedNumberOfUnoccupiedSquaresFloatDirect(dims=(30, 30), n_steps=50)
        print(f"Solution to Project Euler #213 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 214 in to_evaluate:
        since = time.time()
        res = primesOfTotientChainLengthSum(p_max=4 * 10 ** 7 - 1, chain_len=25)
        print(f"Solution to Project Euler #214 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 215 in to_evaluate:
        since = time.time()
        res = crackFreeWalls(n_rows=10, n_cols=32)
        print(f"Solution to Project Euler #215 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 216 in to_evaluate:
        since = time.time()
        res = countPrimesOneLessThanTwiceASquare(n_max=5 * 10 ** 7)
        print(f"Solution to Project Euler #216 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 217 in to_evaluate:
        since = time.time()
        res = balancedNumberCount(max_n_dig=47, base=10, md=3 ** 15)
        print(f"Solution to Project Euler #217 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 218 in to_evaluate:
        since = time.time() 
        res = nonSuperPerfectPerfectRightAngledTriangleCount(max_hypotenuse=10 ** 16)
        print(f"Solution to Project Euler #218 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 219 in to_evaluate:
        since = time.time() 
        res = prefixFreeCodeMinimumTotalSkewCost(n_words=10 ** 9, cost1=1, cost2=4)
        print(f"Solution to Project Euler #219 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 220 in to_evaluate:
        since = time.time() 
        res = heighwayDragon(order=50, n_steps=10 ** 12, init_pos=(0, 0), init_direct=(0, 1), initial_str="Fa", recursive_strs={"a": "aRbFR", "b": "LFaLb"})
        print(f"Solution to Project Euler #220 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 221 in to_evaluate:
        since = time.time() 
        res = nthAlexandrianInteger(n=15 * 10 ** 4)
        print(f"Solution to Project Euler #221 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 222 in to_evaluate:
        since = time.time() 
        res = shortestSpherePackingInTube(tube_radius=50, radii=list(range(30, 51)))
        print(f"Solution to Project Euler #222 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 223 in to_evaluate:
        since = time.time() 
        res = countBarelyAcuteIntegerSidedTrianglesUpToMaxPerimeter(max_perimeter=25 * 10 ** 6)
        print(f"Solution to Project Euler #223 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 224 in to_evaluate:
        since = time.time() 
        res = countBarelyObtuseIntegerSidedTrianglesUpToMaxPerimeter(max_perimeter=75 * 10 ** 6)
        print(f"Solution to Project Euler #224 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 225 in to_evaluate:
        since = time.time() 
        res = nthSmallestTribonacciOddNonDivisors(odd_non_divisor_number=124, init_terms=(1, 1, 1)) 
        print(f"Solution to Project Euler #225 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 226 in to_evaluate:
        since = time.time() 
        res = blacmangeCircleIntersectionArea(eps=10 ** -9)
        print(f"Solution to Project Euler #226 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 227 in to_evaluate:
        since = time.time() 
        res = chaseGameExpectedNumberOfTurns(die_n_faces=6, n_opts_left=1, n_opts_right=1, n_players=100, separation_init=50)
        print(f"Solution to Project Euler #227 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 228 in to_evaluate:
        since = time.time() 
        res = regularPolygonMinkowskiSumSideCount(vertex_counts=list(range(1864, 1910)))
        print(f"Solution to Project Euler #228 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 229 in to_evaluate:
        since = time.time() 
        #res = fourRepresentationsUsingSquaresCount(mults=(1, 2, 3, 7), num_max=2 * 10 ** 6)
        res = fourSquaresRepresentationCountSpecialised(num_max=2 * 10 ** 9) 
        print(f"Solution to Project Euler #229 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 230 in to_evaluate:
        since = time.time() 
        res = fibonacciWordsSum(
            A=1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679,
            B=8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196,
            poly_coeffs=(127, 19),
            exp_base=7,
            n_max=17,
            base=10
        )
        print(f"Solution to Project Euler #230 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 231 in to_evaluate:
        since = time.time() 
        res = binomialCoefficientPrimeFactorisationSum(n=20 * 10 ** 6, k=15 * 10 ** 6)
        print(f"Solution to Project Euler #231 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 232 in to_evaluate:
        since = time.time() 
        res = probabilityPlayer2WinsFloat(points_required=100)
        print(f"Solution to Project Euler #232 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 233 in to_evaluate:
        since = time.time() 
        res = circleInscribedSquareSideLengthWithLatticePointCount(n_lattice_points=420, max_inscribed_square_side_length=10 ** 11)
        print(f"Solution to Project Euler #233 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 234 in to_evaluate:
        since = time.time()
        res = semiDivisibleNumberCount(n_max=999966663333)
        print(f"Solution to Project Euler #234 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 235 in to_evaluate:
        since = time.time() 
        res = arithmeticGeometricSeries(a=900, b=-3, n=5000, val=-6 * 10 ** 11, eps=10 ** -13)
        print(f"Solution to Project Euler #235 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

"""
mx = 10 ** 3
cnt = 0
for b in range(2, mx + 1):
    b_sq = b * b
    for a in range(2, b + 1):
        c_sq = a * a + b_sq - 1
        c = isqrt(c_sq)
        if c * c == c_sq:
            cnt += 1
            #print(a, b, c)
print(cnt)
"""

#for frac in (CustomFraction(1, 1), CustomFraction(1, 2), CustomFraction(1, 3), CustomFraction(1, 4), CustomFraction(1, 5), CustomFraction(2, 5), CustomFraction(1, 6), CustomFraction(1, 7), CustomFraction(1, 8), CustomFraction(3, 8)):
#    print(frac, findBlacmangeValue(frac), findBlacmangeIntegralValue(frac))

def func(n: int) -> int:
    res = 0
    for x in range(1, (n >> 1) + 1):
        y_sq = 2 * n ** 2 - (2 * x - n) ** 2
        res += (isqrt(y_sq) ** 2 == y_sq)
    return res

def func2(n: int) -> int:
    pf = calculatePrimeFactorisation(n)
    res = 1
    for p, f in pf.items():
        if p & 3 != 1: continue
        #res1 += (f << 1) + 1
        res *= (f << 1) + 1
    return res >> 1

#print(func2(10000))

"""
n_max = 1000
for n in range(1, n_max + 1):
    res = func(n)
    res2 = func2(n)
    if res != res2:
        print(n, res, res2)
"""