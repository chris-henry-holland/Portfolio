#!/usr/bin/env python

import bisect
import heapq
import itertools
import math
import os
import random
import sys
import time

from collections import deque
from sortedcontainers import SortedDict, SortedList
from typing import Dict, List, Tuple, Set, Union, Generator, Callable, Optional, Any, Hashable, Iterable

sys.path.append(os.path.join(os.path.dirname(__file__), "../Algorithms_and_Datastructures/Algorithms"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../Algorithms_and_Datastructures/Data_structures"))
from misc_mathematical_algorithms import CustomFraction, gcd, lcm, isqrt
from prime_sieves import PrimeSPFsieve, SimplePrimeSieve
from pseudorandom_number_generators import generalisedLaggedFibonacciGenerator

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
    while cuboids2:
        cuboid, inds = cuboids2.pop(0)
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

if __name__ == "__main__":
    to_evaluate = {212}
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
        print(f"Solution to Project Euler #211 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 213 in to_evaluate:
        since = time.time()
        res = fleaCircusExpectedNumberOfUnoccupiedSquaresFloatDirect(dims=(30, 30), n_steps=50)
        print(f"Solution to Project Euler #213 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 214 in to_evaluate:
        since = time.time()
        res = primesOfTotientChainLengthSum(p_max=4 * 10 ** 7 - 1, chain_len=25)
        print(f"Solution to Project Euler #214 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")