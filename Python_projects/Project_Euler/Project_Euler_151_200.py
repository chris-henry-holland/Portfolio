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
from typing import Dict, List, Tuple, Set, Union, Generator, Callable, Optional, Any, Hashable, Iterable

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
    #return a if not b else gcd(b, a % b)
    while b != 0:
        a, b = b, a % b
    return a
    
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
    Uses the Newton-Raphson method.
    
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

def integerNthRoot(m: int, n: int) -> int:
    """
    For an integer m and a strictly positive integer n,
    finds the largest integer a such that a ** n <= m (or
    equivalently, the floor of the largest real n:th root
    of m. Note that for even n, m must be non-negative.
    Uses the Newton-Raphson method.
    
    Args:
        Required positional:
        m (int): Integer giving the number whose root is
                to be calculated. Must be non-negative
                if n is even.
        n (int): Strictly positive integer giving the
                root to be calculated.
    
    Returns:
    Integer (int) giving the largest integer a such that
    m ** n <= a.
    
    Examples:
    >>> integerNthRoot(4, 2)
    2
    >>> integerNthRoot(15, 2)
    3
    >>> integerNthRoot(27, 3)
    3
    >>> integerNthRoot(-26, 3)
    -3
    """

    # Finds the floor of the n:th root of m, using the positive
    # root in the case that n is even.
    # Newton-Raphson method
    if n < 1:
        raise ValueError("n must be strictly positive")
    if m < 0:
        if n & 1:
            neg = True
            m = -m
        else:
            raise ValueError("m can only be negative if n is odd")
    else: neg = False
    if not m: return 0
    x2 = float("inf")
    x1 = m
    while x1 < x2:
        x2 = x1
        x1 = ((n - 1) * x2 + m // x2 ** (n - 1)) // n
    if not neg: return x2
    if x2 ** n < m:
        x2 += 1
    return -x2

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
    #print(frac1, frac2, (numer // g, denom // g))
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
    neg = (frac1[0] < 0) ^ (frac1[1] < 0) ^ (frac2[0] < 0) ^ (frac2[1] < 0)
    frac_prov = (abs(frac1[0] * frac2[0]), abs(frac1[1] * frac2[1]))
    #print(frac_prov)
    g = gcd(frac_prov[0], frac_prov[1])
    return (-(frac_prov[0] // g) if neg else (frac_prov[0] // g), frac_prov[1] // g)

def floorHarmonicSeries(n: int) -> int:
    """
    Calculates the value of the sum:
        (sum i from 1 to n) floor(n / i)
    using the identity that:
        (sum i from 1 to n) floor(n / i) = ((sum i from 1 to k) floor(n / i)) - k ** 2
    where k = floor(sqrt(n))
    
    Args:
        Required positional:
        n (int): Strictly positive integer giving the value
                for which the value of the above formula is
                to be calculated.
    
    Returns:
    Integer (int) giving the value of:
        sum (i from 1 to n) floor(n / i)
    for the given value of n.
    """
    k = isqrt(n)
    return sum(n // i for i in range(1, k + 1)) - k ** 2

# Problem 151
def singleSheetCountExpectedValueFraction(n_halvings: int) -> Tuple[int, int]:
    """
    Consider a sheet of paper in an envelope. This sheet should
    produce 2 ** n_halvings sheets of paper, each 1 / 2 ** n_halvings
    the size of the original sheet (by area) by repeatedly cutting a
    sheet derived from the original in half.

    This is achieved in multiple stages. Each stage consists of
    randomly selecting a sheet currently in the envelope (where
    all sheets in the envelope have equal probability of being
    selected), and termed as the current sheet. The following
    process is then followed until completion:
        1) If the current sheet is the desired size (i.e.
           1 / 2 ** n_halvings the size of the original sheet)
           then the process is complete. Otherwise, go to step
           2.
        2) Cut the current sheet in half. One half is placed
           back in the envelope, the other half is now the
           current sheet. Go to step 1.
    
    This process is repeated until the envelope is empty.
    Regardless of the order in which the sheets are selected,
    the process is guaranteed to be completed in exactly
    2 ** (n_halvings + 1) - 1 steps.

    This function calculates the expected number of stages in
    this process excluding the first and last stage that start
    with exactly one sheet in the envelope, giving the value
    as a fraction.
    
    Args:
        Required positional:
        n_halvings (int): Non-negative integer giving the
                relative size (in terms of area) of the desired
                sheet size compared to the original sheet size,
                expressed as the number of repeated halvings of
                the original sheet are required to achieve the
                desired size.
    
    Returns:
    2-tuple of integers (ints), giving the expected value for
    the number of stages in the described process excluding the
    first and last stage that start with exactly one sheet in
    the envelope, expressed as a fraction in lowest terms (i.e.
    numerator and denominator are coprime), where index 0 is
    a non-negative integer giving the numerator and index 1
    is a strictly positive integer giving the denominator.

    Outline of rationale:
    We define a state as (where the number of each size sheet
    of paper in the envelope.
    We then observe that given that in any given run through
    of the process, the total area of paper is strictly
    decreasing with every stage (as no paper is added and a
    sheet of the desired size is always removed at the end
    of the stage), a state cannot occur more than once in
    the same process.
    Consequently the solution is then simply the sum of the
    probabilities of encountering each state where there is
    only one sheet of paper that is not the original size or
    the desired size.
    These probabilities are calculated with top-down dynamic
    programming with memoisation, working backwards from each
    chosen state  to a state containing only a single sheet
    of the original size to calculate the proportion of the
    possible applications of the process encounter that state
    at some stage. Given that sheets are selected uniform
    randomly is equal to the probability that that state is
    encountered in a single run through of the process, the
    number required.
    TODO- revise for clarity
    """
    #mx_counts = [1 << i for i in range(n_halvings)]
    mx_tot = 1 << n_halvings

    memo = {}
    def recur(state: List[int], tot: int, sm: int) -> Tuple[int, int]:
        if tot >= mx_tot:
            return (int(tot == mx_tot and state[0] == 1), 1)
        args = tuple(state)
        #print(args, sm, tot)
        if args in memo.keys(): return memo[args]
        res = (0, 1)
        for i in reversed(range(n_halvings + 1)):
            #if state[i] == mx_counts[i]:
            #    sub = 1
            #    state[i] -= sub
            #    if state[i] < 0: break
            #    sm -= sub * (1 << i)
            #    continue
            add = 1
            tot += add * (1 << (n_halvings - i))
            sm += add
            state[i] += add
            #print(i)
            res = addFractions(res, multiplyFractions(recur(state, tot, sm), (state[i], sm)))
            sub = 2
            state[i] -= sub
            if state[i] < 0: break
            sm -= sub
            tot -= sub * (1 << (n_halvings - i))
        else: i = 0
        for i in range(i, len(state)):
            state[i] += 1
            sm -= tot
            tot += (1 << (n_halvings - i))
        memo[args] = res
        return res

    state = [0] * (n_halvings + 1)
    state[0] = 1
    res = (0, 1)
    for i in range(1, n_halvings):
        state[i] += 1
        state[i - 1] -= 1
        #print(state)
        ans = recur(state, 1 << (n_halvings - i), 1)
        #print(ans)
        res = addFractions(res, ans)
    #print(memo)
    return res

def singleSheetCountExpectedValueFloat(n_halvings: int=5) -> float:
    """
    Solution to Project Euler #151

    Consider a sheet of paper in an envelope. This sheet should
    produce 2 ** n_halvings sheets of paper, each 1 / 2 ** n_halvings
    the size of the original sheet (by area) by repeatedly cutting a
    sheet derived from the original in half.

    This is achieved in multiple stages. Each stage consists of
    randomly selecting a sheet currently in the envelope (where
    all sheets in the envelope have equal probability of being
    selected), and termed as the current sheet. The following
    process is then followed until completion:
        1) If the current sheet is the desired size (i.e.
           1 / 2 ** n_halvings the size of the original sheet)
           then the process is complete. Otherwise, go to step
           2.
        2) Cut the current sheet in half. One half is placed
           back in the envelope, the other half is now the
           current sheet. Go to step 1.
    
    This process is repeated until the envelope is empty.
    Regardless of the order in which the sheets are selected,
    the process is guaranteed to be completed in exactly
    2 ** (n_halvings + 1) - 1 steps.

    This function calculates the expected number of stages in
    this process excluding the first and last stage that start
    with exactly one sheet in the envelope, giving the value
    as a float.
    
    Args:
        Required positional:
        n_halvings (int): Non-negative integer giving the
                relative size (in terms of area) of the desired
                sheet size compared to the original sheet size,
                expressed as the number of repeated halvings of
                the original sheet are required to achieve the
                desired size.
    
    Returns:
    Float, giving the expected value for the number of stages in
    the described process excluding the first and last stage that
    start with exactly one sheet in the envelope.

    Outline of rationale:
    See doumentation for singleSheetCountExpectedValueFraction().
    """
    since = time.time()
    frac = singleSheetCountExpectedValueFraction(n_halvings=n_halvings)
    res = frac[0] / frac[1]
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 152
def sumsOfSquareReciprocals(target: Tuple[int, int]=(1, 2), denom_min: int=2, denom_max: int=80) -> Set[Tuple[int]]:
    """
    Finds the distinct ways the fraction target can be constructed
    by summing recpirocals of squares of integers, where the
    integers being squared are between denom_min and denom_max
    inclusive, and no two integers in the sum are the same. Two
    such sums are considered distinct if and only if there is a
    term that is present in one sum that is not present in the
    other sum (so two sums where the terms are simply a permutation
    of each other are not distinct).

    Args:
        Optional named:
        target (2-tuple of ints): Two strictly positive integers,
                representing the numerator and denominator
                respectively of the target fraction that the
                sums must equal.
            Default: (1, 2)- representing a half
        denom_min (int): Strictly positive integer giving the
                smallest integer for whose squared reciprocal
                can appear in the sum.
            Default: 2
        denom_max (int): Strictly positive integer giving the
                largest integer for whose squared reciprocal
                can appear in the sum.
            Default: 80
    
    Returns:
    Set of tuples of ints, where each tuple contains a distinct
    list of integers in strictly increasing order (each distinct
    and between denom_min and denom_max) such that the sum of the
    reciprocals of their squares equals target, and collectively
    they represent all such distinct sums.

    Example:
        >>> sumsOfSquareReciprocals(target=(1, 2), denom_min=2, denom_max=45))
        {(2, 3, 4, 6, 7, 9, 10, 20, 28, 35, 36, 45), (2, 3, 4, 6, 7, 9, 12, 15, 28, 30, 35, 36, 45), (2, 3, 4, 5, 7, 12, 15, 20, 28, 35)}

        This indicates that there are exactly three distinct sums
        of distinct integer square reciprocals equal to a half,
        such that each integer in between 2 and 45 inclusive,
        including:
            1 / 2 ** 2 + 1 / 3 ** 2 + 1 / 4 ** 2 + 1 / 5 ** 2
                + 1 / 7 ** 2 + 1 / 9 ** 2 + 1 / 10 ** 2 + 20 / 2 ** 2
                + 1 / 28 ** 2 + 1 / 35 ** 2 + 1 / 36 ** 2 + 1 / 45 ** 2
        which it can be verified is indeed equal to a half.
    
    Outline of rationale:
    TODO
    """
    # TODO- need to check that finds all solutions for any
    # target
    # Add check for whether the smallest denominators must be
    # included (i.e. without them the sum cannot reach the
    # target) or cannot be included (i.e. the value of that
    # term alone exceeds the target) and adjust the target
    # and other parameters accordingly
    # Try to make more efficient and rule out more combinations
    # at an early stage

    g = gcd(*target)
    target = tuple(t // g for t in target)

    #since = time.time()

    def checkSolution(denoms: Tuple[int]) -> bool:
        res = (0, 1)
        for d in denoms:
            res = addFractions(res, (1, d ** 2))
        #print(f"solution check: {denoms}, {res}")
        return res == target
    
    # Find the maximum prime factor of denominators possible
    curr = 0
    prod = 1
    i = 1
    while True:
        curr = curr * i ** 2 + prod
        #print(i, curr)
        if curr * i >= denom_max ** 2:
            break
        prod *= i ** 2
        i += 1
    #print(denom_max // i)

    ps = PrimeSPFsieve(n_max=denom_max // i)

    p_set = set(ps.p_lst)
    target_denom_pf = ps.primeFactorisation(target[1])
    p_lst = sorted(p_set.union(target_denom_pf.keys()))

    p_split_idx = 2
    #print(ps.p_lst)
    p_lst1 = p_lst[:p_split_idx]
    p_lst2 = p_lst[p_split_idx:]

    
    #print(p_lst)
    p_opts = {}
    if p_lst2:
        for bm in range(1, 1 << (denom_max // p_lst2[0])):
            bm2 = bm
            i = 1
            curr = 0
            prod = 1
            mx = 0
            #print(bm2)
            while bm2:
                if bm2 & 1:
                    curr = curr * i ** 2 + prod
                    prod *= i ** 2
                    mx = i
                i += 1
                bm2 >>= 1
            for p in p_lst2:
                if mx * p > denom_max: break
                #if p == 7 and bm == 7:
                #    print(curr)
                #print(p, curr, bm)
                if p in target_denom_pf:
                    exp = 1
                    bm2 = bm
                    i = 1
                    while bm2:
                        if bm2 & 1:
                            i2, r = divmod(i, p)
                            while not r:
                                exp += 1
                                i2, r = divmod(i2, p)
                        i += 1
                        bm2 >>= 1
                    if 2 * exp < target_denom_pf[p]: continue
                    q, r = divmod(curr, p ** (2 * exp - target_denom_pf[p]))
                    if r or not q % p:
                        continue
                    #print(p, curr, mx_pow, bm)
                    p_opts.setdefault(p, set())
                    p_opts[p].add(bm)
                    continue
                if curr % (p ** 2): continue
                exp = 1
                bm2 = bm
                i = 1
                while bm2:
                    if bm2 & 1:
                        i2, r = divmod(i, p)
                        while not r:
                            exp += 1
                            i2, r = divmod(i2, p)
                    i += 1
                    bm2 >>= 1
                if exp > 1 and curr % (p ** (2 * exp)):
                    continue
                #print(p, curr, mx_pow, bm)
                p_opts.setdefault(p, set())
                p_opts[p].add(bm)
    if not set(target_denom_pf.keys()).issubset(set(p_lst1).union(set(p_opts.keys()))):
        return []
    for st in p_opts.values():
        st.add(0)
    #print(p_opts)
    
    n_pairs = 0
    p_lst2_2 = sorted(p_opts.keys())
    #print(p_lst2_2)
    allowed_pairs = {}
    for i2 in reversed(range(1, len(p_lst2_2))):
        p2 = p_lst2_2[i2]
        allowed_pairs[p2] = {}
        for i1 in reversed(range(i2)):
            p1 = p_lst2_2[i1]
            allowed_pairs[p2][p1] = {}
            for bm1 in p_opts[p1]:
                for bm2 in p_opts[p2]:
                    bm1_ = bm1 >> (p2 - 1)
                    bm2_ = bm2 >> (p1 - 1)
                    while bm1_ or bm2_:
                        if bm1_ & 1 != bm2_ & 1: break
                        bm1_ >>= p2
                        bm2_ >>= p1
                    else:
                        allowed_pairs[p2][p1].setdefault(bm2, set())
                        allowed_pairs[p2][p1][bm2].add(bm1)
                        n_pairs += 1

    if p_lst2_2:
        #print(allowed_pairs)
        #print(n_pairs)
        curr_allowed = [(x,) for x in p_opts[p_lst2_2[-1]]]
        tot = 0
        for i1 in reversed(range(len(p_lst2_2) - 1)):
            prev_allowed = curr_allowed
            curr_allowed = []
            p1 = p_lst2_2[i1]
            for bm_lst in prev_allowed:
                curr_set = set(allowed_pairs[p_lst2_2[-1]][p1].get(bm_lst[-1], set()))
                if not curr_set: break
                for j in range(1, len(bm_lst)):
                    p2 = p_lst2_2[~j]
                    bm2 = bm_lst[~j]
                    curr_set &= allowed_pairs[p2][p1].get(bm2, set())
                    if not curr_set: break
                for bm in curr_set:
                    curr_allowed.append((bm, *bm_lst))
                    tot += 1
    else: curr_allowed = [()]
    #print(len(curr_allowed), len(set(curr_allowed)))
    #print(curr_allowed)
    #print(tot)
    #frac_counts = {}
    frac_sets = {}
    seen = {}
    for bm_lst in curr_allowed:
        denom_set = set()
        for p, bm in zip(p_lst2_2, bm_lst):
            mult = 1
            while bm:
                if bm & 1:
                    denom_set.add(p * mult)
                bm >>= 1
                mult += 1
        denom_tup = tuple(sorted(denom_set))
        #print(denom_tup)
        if denom_tup and denom_tup[0] < denom_min: continue
        tot = target
        for denom in denom_set:
            tot = addFractions(tot, (-1, denom ** 2))
        #frac_counts[tot] = frac_counts.get(tot, 0) + 1
        frac_sets.setdefault(tot, set())
        #if denom_tup in frac_sets[tot]: print(f"repeated denominator tuple: {denom_tup}")
        frac_sets[tot].add(denom_tup)
        if denom_tup in seen.keys():
            print(f"repeated denominator tuple: {denom_tup}, bm_lsts: {seen[denom_tup]}, {bm_lst}")
        else: seen[denom_tup] = bm_lst
        #print(tot[1])
        #print(tot, denom_set, bm_lst, ps.primeFactorisation(tot[1]))
    #print(frac_sets)
    #print("finished stage 1")

    n_p1 = len(p_lst1)
    curr = {(0, 1): [()]}
    seen = set()
    # Ensure the results that already equal the target with only
    # the denominators considered in the previous step are included
    # in the answer
    res = list(frac_sets.get((0, 1), set()))
    #print(res)

    def recur(p_idx: int=0, denom: int=1) -> None:
        #print(p_idx, denom)
        if p_idx == n_p1:
            if denom < denom_min: return
            #print(f"denom = {denom}")
            val = (1, denom ** 2)
            prev = {x: list(y) for x, y in curr.items()}
            #print(prev)
            for frac, tups in prev.items():
                frac2 = addFractions(frac, val)
                #if frac2 == (31, 72): print("found")
                if frac2[0] * target[1] > frac2[1] * target[0]: continue
                curr.setdefault(frac2, [])
                for tup in tups:
                    denom_tup2 = (*tup, denom)
                    curr[frac2].append(denom_tup2)
                    #print(pow2 * 3 ** exp3)
                    for denom_tup1 in frac_sets.get(frac2, set()):
                        ans = tuple(sorted([*denom_tup1, *denom_tup2]))
                        if ans in seen:
                            print(f"repeated solution: {ans}")
                            continue
                        if not checkSolution(ans):
                            print(f"Incorrect solution: {ans}")
                        elif ans[-1] > denom_max:
                            print(f"Solution has denominator that is too large: {ans}")
                        else:
                            seen.add(ans)
                            res.append(ans)
                        #print(exp2, exp3, nxt2[frac2], frac2)
                        #print("solution:", sorted(denom_set.union(set(denom_tup))))
            return
        p = p_lst1[p_idx]
        #denom2 = 2 * denom
        #pow_mx = 0
        #while denom2 < denom_max:
        #    pow_mx += 1
        #    denom2 *= p
        d = denom
        d_mx = denom_max# >> 1
        while d <= d_mx:
            recur(p_idx=p_idx + 1, denom=d)
            d *= p
        return

    recur(p_idx=0, denom=1)
    #print(res)
    """
    pow2_mx = 0
    num = denom_max // 3
    while num:
        pow2_mx += 1
        #print(num, pow2_mx)
        num >>= 1
    
    res = []
    curr2 = {(0, 1): [()]}
    seen = set()
    for exp2 in range(pow2_mx + 1):
        pow2 = 1 << exp2
        pow3_mx = denom_max // pow2
        #print(exp2, denom_max, pow3_mx)
        exp3_mx = -1
        pow3_mx2 = pow3_mx
        while pow3_mx2:
            pow3_mx2 //= 3
            exp3_mx += 1
        #print(f"exp3_mx for exp2 = {exp2} is {exp3_mx}")
        for exp3 in range(not exp2, exp3_mx + 1):
            #print("powers: ", exp2, exp3)
            d = pow2 * (3 ** exp3)
            if d < denom_min: continue
            val = (1, d ** 2)
            nxt2 = {x: list(y) for x, y in curr2.items()}
            for frac, tups in curr2.items():
                frac2 = addFractions(frac, val)
                #if frac2 == (31, 72): print("found")
                if frac2[0] * 2 >= frac2[1]: continue
                nxt2.setdefault(frac2, [])
                for tup in tups:
                    denom_tup2 = (*tup, d)
                    nxt2[frac2].append(denom_tup2)
                    #print(pow2 * 3 ** exp3)
                    for denom_tup1 in frac_sets.get(frac2, set()):
                        ans = tuple(sorted([*denom_tup1, *denom_tup2]))
                        if ans in seen:
                            print(f"repeated solution: {ans}")
                            continue
                        if not checkSolution(ans):
                            print(f"Incorrect solution: {ans}")
                        elif ans[-1] > denom_max:
                            print(f"Solution has denominator that is too large: {ans}")
                        else:
                            seen.add(ans)
                            res.append(ans)
                        #print(exp2, exp3, nxt2[frac2], frac2)
                        #print("solution:", sorted(denom_set.union(set(denom_tup))))
            curr2 = nxt2
        #print(exp2, curr)
    """
    """
    opts3_0 = []
    for bm in range(1 << (pow3_mx + 1)):
        mx_denom = 1
        bm2 = bm
        curr = 1
        frac = (0, 1)
        while bm2:
            if bm2 & 1:
                frac = addFractions(frac, (1, curr ** 2))
                mx_denom = curr
            curr *= 3
            bm2 >>= 1
        opts3_0.append((mx_denom, frac))
    opts3_0.sort()
    
    opts3_1 = []
    for bm in range(1 << pow3_mx):
        mx_denom = 1
        bm2 = bm
        curr = 3
        frac = (0, 1)
        while bm2:
            if bm2 & 1:
                frac = addFractions(frac, (1, curr ** 2))
                mx_denom = curr
            curr *= 3
            bm2 >>= 1
        opts3_1.append((mx_denom, frac))
    opts3_1.sort()
    res = 0
    for bm in range(1 << (pow2_mx + 1)):
        mx_denom = 1
        bm2 = bm
        curr = 1
        frac = (0, 1)
        while bm2:
            if bm2 & 1:
                frac = addFractions(frac, (1, curr ** 2))
                mx_denom = curr
            curr *= 2
            bm2 >>= 1
        mx_denom3 = denom_max // mx_denom
        for opt3 in (opts3_1 if bm & 1 else opts3_0):
            if opt3[0] > mx_denom3: break
            frac2 = multiplyFractions(frac, opt3[1])
            print(frac, opt3[1], frac2)
            res += frac_counts.get(frac2, 0)
    #print(pow2_mx, pow3_mx, 2 ** pow2_mx * 3 ** pow3_mx)

    #for bm2 in range()

    print(len(frac_counts))
    """
    #print(f"Time taken = {time.time() - since:.4f} seconds")
    return set(res)

def sumsOfSquareReciprocalsCount(target: Tuple[int, int]=(1, 2), denom_min: int=2, denom_max: int=80) -> int:
    """
    Solution to Project Euler #152

    Finds the number of distinct ways the fraction target can be
    constructed by summing recpirocals of squares of integers,
    where the integers being squared are between denom_min and
    denom_max inclusive, and no two integers in the sum are the
    same. Two such sums are considered distinct if and only if
    there is a term that is present in one sum that is not
    present in the other sum (so two sums where the terms are
    simply a permutation of each other are not distinct).

    Args:
        Optional named:
        target (2-tuple of ints): Two strictly positive integers,
                representing the numerator and denominator
                respectively of the target fraction that the
                sums must equal.
            Default: (1, 2)- representing a half
        denom_min (int): Strictly positive integer giving the
                smallest integer for whose squared reciprocal
                can appear in the sum.
            Default: 2
        denom_max (int): Strictly positive integer giving the
                largest integer for whose squared reciprocal
                can appear in the sum.
            Default: 80
    
    Returns:
    Integer (int) giving the number of distinct sums of square
    reciprocals of different integers between denom_min and denom_max
    inclusive that are equal to target.
    """
    since = time.time()
    sols = set(sumsOfSquareReciprocals(target=target, denom_min=denom_min, denom_max=denom_max))
    #for sol in sols:
    #    print(sol)
    res = len(sols)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 153
"""
def findIntegerCountGaussianIntegerDivides(a: int, b: int, n_max: int) -> int:
    if not b: return n_max // a
    k = gcd(a, b)
    a_, b_ = a // k, b // k
    res = n_max // ((a_ ** 2 + b_ ** 2) * k)
    #print((a, b), res)
    return res
"""

def findRealPartSumOverGaussianIntegerDivisors(n_max: int=10 ** 8) -> int:
    """
    Solution to Project Euler #153

    Calculates the sum of the real part of all Gaussian integer
    factors with positive real part over all strictly positive
    integers no greater than
    n_max.

    Args:
        Optional named:
        n_max (int): The largest integer whose Gaussian integer
                factors are considered in the sum
            Default: 10 ** 8
    
    Returns:
    Integer (int) representing the sum of the real part of all
    Gaussian integer factors with positive real part over all
    strictly positive integers no greater that n_max.

    Example:
        >>> findRealPartSumOverGaussianIntegerDivisors(n_max=5)
        35

        This indicates that the sum over the Gaussian integer
        factors of the integers between 1 and 5 inclusive is
        35. The Gaussian integer factors of these integers with
        positive real part are:
            1: 1
            2: 1, 1 + i, 1 - i, 2
            3: 1, 3
            4: 1, 1 + i, 1 - i, 2, 2 + 2 * i, 2 - 2 * i, 4
            5: 1, 1 + 2 * i, 1 - 2 * i, 2 + i, 2 - i, 5
        The sum over the real part of all of these factors is
        indeed 35.
    
    Brief outline of rationale:
    We consider this sum from the perspective of the Gaussian
    integer factors rather than the integers, rephrasing the
    question into the equivalent one of, calculate the sum over
    all Gaussian integers with positive real part of the
    real part multiplied by the number of strictly positive
    integers no greater than n_max for which that Gaussian
    integer is a factor.
    We observe that (a + bi) is a factor of an integer m if
    and only if (a - bi), (b + ai) and (b - ai) are factors
    of m (TODO- prove this)
    There are three cases to consider: Gaussian integers with no
    imaginary part (i.e. integers); Gaussian integers that are
    positive integer multiples of (1 + i) and (1 - i); and the
    other Gaussian integers with positive real part (i.e. those
    with non-zero imaginary parts whose real and imaginary parts
    are different sizes).
    Case 1: For Gaussian integers with no imaginary parts, the
    number of strictly positive integers no greater than n_max
    for which this is a factor is simply the number of positive
    integer multiples of this number that are no greater than
    n_max, which is given by the floor of n_max divided by
    the number. Therefore the contribution of these numbers to
    the overall sum is simply:
        (sum a from 1 to n_max) (a * (n_max // a))
    Case 2: As previously noted, an integer is divisible by (1 + i)
    if and only if it is also divisible by (1 - i). As such,
    an integer is divisible by (1 + i) or (1 - i) if and only if
    it is divisible by (1 + i) * (1 - i) = 2. It follows that for
    positive integer a, the positive integer m is divisible by
    a * (1 + i) if and only if it is divisible by a * (1 - i) and
    is a multiple of 2 * a. Consequently, the contribution to the
    sum of all Gaussian integers of the forms a * (1 + i) and
    a * (1 - i) for positive integer a is:
        2 * (sum a from 1 to n_max) a * (n_max // (2 * a))
    Case 3: TODO
    """
    since = time.time()

    memo = {}
    def divisorSum(mag_sq: int) -> int:
        
        args = mag_sq
        if args in memo.keys(): return memo[args]
        
        def countSum(m: int) -> int:
            return (m * (m + 1)) >> 1

        rt = isqrt(n_max // mag_sq)

        res = 0
        for i in range(1, (n_max // (mag_sq * (1 + rt))) + 1):
            res += i * (n_max // (i * mag_sq))
        for i in range(1, rt + 1):
            res += i * (countSum(n_max // (i * mag_sq)) - countSum(n_max // ((i + 1) * mag_sq)))
        """
        j1, j2 = n_max // mag_sq, n_max // (2 * mag_sq)
        res = ((j1 * (j1 + 1)) >> 1) - ((j2 * (j2 + 1)) >> 1)
        for i in range(1, (n_max // (2 * mag_sq)) + 1):
            res += (n_max // (i * mag_sq)) * i
        """
        memo[args] = res
        return res

    res = 0
    for a in range(1, n_max + 1):
        res += a * (n_max // a)
        a_sq = a ** 2
        for b in range(1, a):
            mag_sq = a_sq + b ** 2
            if mag_sq > n_max: break
            if gcd(a, b) != 1: continue
            res += 2 * (a + b) * divisorSum(mag_sq)
            
        #print(a, res)
    res += 2 * divisorSum(2)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 154
def multinomialCoefficientMultiplesCount(n: int=2 * 10 ** 5, n_k: int=3, factor_p_factorisation: Dict[int, int]={2: 12, 5: 12}) -> int:
    """
    For integers n and n_k and a prime factorisation
    factor_p_factorisation, finds the number of non-zero n_k-nomial
    coefficients with n as the number chosen from that are divisible
    by the integer with the prime factorisation factor_p_factorisation.

    For any list of n_k integers:
        (k_1, k_2, ..., k_(n_k))
    the n_k-nomial coefficient (also termed n_k multinomial coefficients)
    is the number of ways n items can be partitioned into n_k sets with
    sizes corresponding to the integers in the above list, where n is
    the number chosen from, given by the sum of the list of integers,
    the ordering within the sets does not matter and the sets themselves
    are considered to be distinguishable (so that if two of the list of
    n_k integers have the same value, the partitionings created by
    swapping the items from one of the corresponding sets to the
    other results in a different partitioning, despite each item
    being grouped with the same items in both partitionings). If
    any of the n_k integers in the list are negative, then the
    corresponding n_k-nomial coefficient is defined to be zero.
    
    The value of this coefficient for non-negative k_1, k_2, ...,
    k_(n_k) and n being the sum of these integers is given by:
        n! / (k_1! * k_2! * ... * k_(n_k)!)

    Args:
        Optional named:
        n (int): Non-negative integer giving the number chosen from
                for the n_k-nomial coefficients considered, i.e. the
                sum of the partition sizes in the n_k-nomial
                coefficients considered.
            Default: 2 * 10 ** 5
        n_k (int): Strictly positive integer giving the number of
                partitions in the multinomial coefficients considered.
            Default: 3
        factor_p_factorisation (dict): Dictionary representing a
                prime factorisation of the strictly positive integer
                by which the n_k-nomial coefficients for given n
                and n_k counted are to be divisible, whose keys are
                the prime numbers that appear in the prime
                factorisation of the integer in question, with the
                corresponding value being the number of times that
                prime appears in the factorisation (i.e. the power
                of that prime in the prime factorisation of the
                integer). An empty dictionary corresponds to the
                multiplicative identity (i.e. 1).
            Default: {2: 12, 5: 12} (the prime factorisation of 10 ** 12)

    Returns:
    Integer (int) giving the number of non-zero n_k-nomial coefficients
    with n as the number chosen from that are divisible by the integer with
    the prime factorisation factor_p_factorisation.

    TODO- reword for clarity

    Outline of rationale:
    TODO
    """
    since = time.time()
    #ps = PrimeSPFsieve(n_max=factor)
    #factor_p_factorisation = ps.primeFactorisation(factor)
    p_lst = sorted(factor_p_factorisation.keys())
    n_p = len(p_lst)
    target = [factor_p_factorisation[p] for p in p_lst]
    print(p_lst, target)
    #for p in p_lst:
    #    n2 = n
    #    target.append(-factor_p_factorisation[p])
    #    while n2:
    #        n2 //= p
    #        target[-1] += n2
    
    
    def createNDimensionalArray(shape: List[int]) -> list:
        n_dims = len(shape)
        def recur(lst: list, dim_idx: int) -> list:
            if dim_idx == n_dims - 1:
                for _ in range(shape[dim_idx]):
                    lst.append(0)
                return
            for _ in range(shape[dim_idx]):
                lst.append([])
                recur(lst[-1], dim_idx + 1)
            return lst
        return recur([], 0)

    def getNDimensionalArrayElement(arr: list, inds: Tuple[int]) -> int:
        res = arr
        for idx in inds:
            res = res[idx]
        return res

    def setNDimensionalArrayElement(arr: list, inds: Tuple[int], val: int) -> None:
        lst = arr
        for idx in inds[:-1]:
            lst = lst[idx]
        lst[inds[-1]] = val
        return
    
    def modifyNDimensionalArrayElement(arr: list, inds: Tuple[int], delta: int) -> None:
        lst = arr
        for idx in inds[:-1]:
            lst = lst[idx]
        lst[inds[-1]] += delta
        return
    
    def findNDimensionalArrayShape(arr: list) -> Tuple[int]:

        lst = arr
        res = []
        while not isinstance(lst, int):
            res.append(len(lst))
            if not lst: break
            lst = lst[0]
        return tuple(res)
    
    def deepCopyNDimensionalArray(arr: list) -> list:
        shape = findNDimensionalArrayShape(arr)
        res = createNDimensionalArray(shape)

        def recur(inds: List[int], lst1: Union[list, int], lst2: Union[list, int]) -> None:
            dim_idx = len(inds)
            if dim_idx == len(shape) - 1:
                for idx in range(shape[-1]):
                    lst2[idx] = lst1[idx]
                return
            inds.append(0)
            for idx in range(shape[dim_idx]):
                recur(inds, lst1[idx], lst2[idx])
            inds.pop()
            return
        recur([], arr, res)
        return res
    
    def addDisplacedNDimensionalArray(arr1: list, arr2: list, displacement: Tuple[int]) -> None:
        shape1 = findNDimensionalArrayShape(arr1)
        shape2 = findNDimensionalArrayShape(arr2)

        def recur(inds: List[int], lst1: Union[list, int], lst2: Union[list, int]) -> None:
            dim_idx = len(inds)
            if dim_idx == len(shape1) - 1:
                for idx2 in range(shape2[-1], shape1[-1] - displacement[-1]):
                    lst1[idx2 + displacement[-1]] += lst2[idx2]
                return
            inds.append(0)
            for idx2 in range(shape2[dim_idx], shape1[dim_idx] - displacement[dim_idx]):
                recur(inds, lst1[idx2 + displacement[dim_idx]], lst2[dim_idx])
            inds.pop()
            return
        recur([], arr1, arr2)
        return

    def convertSumAllIndicesNoLessThanArray(arr: list) -> list:
        shape = findNDimensionalArrayShape(arr)
        #print(arr)
        #print(shape)
        def recur(inds: Tuple[int], lst: Union[list, int], lst2: Union[list, int], dim_idx: int, add_cnt_parity: bool=False, any_add: bool=False) -> None:
            idx = inds[dim_idx]
            if dim_idx == len(shape) - 1:
                if any_add:
                    lst[idx] += (lst2[idx] if add_cnt_parity else -lst2[idx])
                    if idx + 1 < shape[dim_idx]:
                        lst[idx] += (-lst2[idx + 1] if add_cnt_parity else lst2[idx + 1])
                    return
                    #lst[shape[-1] - 1] += -lst2[shape[-1] - 1] if add_cnt_parity else lst2[shape[-1] - 1]
                    #for idx in reversed(range(shape[-1] - 1)):
                    #    lst[idx] += (lst2[idx] if add_cnt_parity else -lst2[idx]) - (lst2[idx + 1] if add_cnt_parity else -lst2[idx + 1])
                    #return
                #for idx in reversed(range(shape[-1] - 1)):
                #    lst[idx] += lst2[idx + 1]
                if idx + 1 < shape[dim_idx]:
                    lst[idx] += lst2[idx + 1]
                return
                #return lst2 if add_cnt_parity else -lst2
            #print(dim_idx)
            recur(inds, lst[idx], lst2[idx], dim_idx + 1, add_cnt_parity=add_cnt_parity, any_add=any_add)
            if idx + 1 < shape[dim_idx]:
                recur(inds, lst[idx], lst2[idx + 1], dim_idx + 1, add_cnt_parity=not add_cnt_parity, any_add=True)
            #recur(lst[shape[dim_idx] - 1], lst2[shape[dim_idx] - 1], dim_idx + 1, add_cnt_parity=add_cnt_parity, any_add=any_add)
            #for idx in reversed(range(shape[dim_idx] - 1)):
            #    recur(lst[idx], lst2[idx], dim_idx + 1, add_cnt_parity=add_cnt_parity, any_add=any_add)
            #    recur(lst[idx], lst2[idx + 1], dim_idx + 1, add_cnt_parity=add_cnt_parity, any_add=True)
            return
        
        def recur2(inds: List[int]) -> None:
            #print(f"recur2(): {inds}")
            dim_idx = len(inds)
            if dim_idx == len(shape):
                #print(inds)
                recur(inds, arr, arr, 0, add_cnt_parity=False, any_add=False)
                #print(arr)
                return
            inds.append(0)
            for idx in reversed(range(shape[dim_idx])):
                inds[-1] = idx
                recur2(inds)
            inds.pop()
            return
        #recur(arr, arr, 0, add_cnt_parity=False, any_add=False)
        recur2([])
        return arr
        """
        def inclExcl(inds: Tuple[int]) -> int:
            
            def recur2(lst: Union[list, int], dim_idx: int, add_cnt_parity: bool=False) -> int:
                if dim_idx == len(shape):
                    return lst if add_cnt_parity else -lst
                res = recur2(lst[inds[dim_idx]], dim_idx + 1, add_cnt_parity=add_cnt_parity)
                if inds[dim_idx] < shape[dim_idx] - 1:
                    res += recur2(lst[inds[dim_idx] + 1], dim_idx + 1, add_cnt_parity=not add_cnt_parity)
                return res
            return recur2(arr, 0, add_cnt_parity=False)
        
        def recur(inds: List[int], lst: Union[list, int]) -> None:
            if len(inds) == len(shape):
                val = inclExcl(inds)

                return
            inds.append(0)
            for idx in reversed(range(len(inds))):
                inds[-1] = idx
            return
        """
    
    
    shape = tuple(x + 1 for x in target)
    curr = [createNDimensionalArray(shape) for _ in range(2)]
    setNDimensionalArrayElement(curr[0], [0] * len(shape), 1)
    setNDimensionalArrayElement(curr[1], [0] * len(shape), 2)
    counts = [tuple([0] * n_p)] * 2
    counts_curr = [0] * n_p
    for n2 in range(2, n + 1):
        print(f"n2 = {n2}")
        for i, p in enumerate(p_lst):
            n3 = n2
            while True:
                n3, r = divmod(n3, p)
                if r: break
                counts_curr[i] += 1
        counts.append(tuple(counts_curr))
        curr.append(createNDimensionalArray(shape))
        for k in range(n2 - (n2 >> 1)):
            #print(f"k = {k}")
            inds = tuple(min(t, x - y - z) for t, x, y, z in zip(target, counts[n2], counts[k], counts[n2 - k]))
            #print(inds)
            modifyNDimensionalArrayElement(curr[-1], inds, 2)
        if not n2 & 1:
            #print(f"k = {n2 >> 1}")
            inds = tuple(min(t, x - 2 * y) for t, x, y in zip(target, counts[n2], counts[n2 >> 1]))
            #print(inds)
            modifyNDimensionalArrayElement(curr[-1], inds, 1)
        #print(f"n2 = {n2}:")
        #print(curr[-1])
        convertSumAllIndicesNoLessThanArray(curr[-1])
        #print(curr[-1])
    
    #print(curr[n])

    if n_k == 2:
        return getNDimensionalArrayElement(curr[n], target)

    for _ in range(3, n_k):
        for n2 in reversed(range(n + 1)):
            arr = createNDimensionalArray(shape)
            for k in reversed(range(n2 + 1)):
                delta = tuple(x - y - z for x, y, z in zip(counts[n2], counts[k], counts[n2 - k]))
                addDisplacedNDimensionalArray(arr, curr[k], delta)
            curr[n2] = arr
    
    res = 0
    for k in range(n + 1):
        print(f"k = {k}")
        inds = tuple(max(t - (x - y - z), 0) for t, x, y, z in zip(target, counts[n], counts[k], counts[n - k]))
        #print(inds, target)
        res += getNDimensionalArrayElement(curr[k], inds)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    """
    arr = createNDimensionalArray((2, 2, 2))
    setNDimensionalArrayElement(arr, (1, 1, 1), 3)
    setNDimensionalArrayElement(arr, (0, 1, 0), 4)
    setNDimensionalArrayElement(arr, (0, 0, 1), 5)
    print(arr)
    convertSumAllIndicesNoLessThanArray(arr)
    print(arr)
    """

def multinomialCoefficientMultiplesCount2(n: int=2 * 10 ** 5, n_k: int=3, factor_p_factorisation: Dict[int, int]={2: 12, 5: 12}) -> int:
    """
    Solution to Project Euler #154

    For integers n and n_k and a prime factorisation
    factor_p_factorisation, finds the number of non-zero n_k-nomial
    coefficients with n as the number chosen from that are divisible
    by the integer with the prime factorisation factor_p_factorisation.

    For any list of n_k integers:
        (k_1, k_2, ..., k_(n_k))
    the n_k-nomial coefficient (also termed n_k multinomial coefficients)
    is the number of ways n items can be partitioned into n_k sets with
    sizes corresponding to the integers in the above list, where n is
    the number chosen from, given by the sum of the list of integers,
    the ordering within the sets does not matter and the sets themselves
    are considered to be distinguishable (so that if two of the list of
    n_k integers have the same value, the partitionings created by
    swapping the items from one of the corresponding sets to the
    other results in a different partitioning, despite each item
    being grouped with the same items in both partitionings). If
    any of the n_k integers in the list are negative, then the
    corresponding n_k-nomial coefficient is defined to be zero.
    
    The value of this coefficient for non-negative k_1, k_2, ...,
    k_(n_k) and n being the sum of these integers is given by:
        n! / (k_1! * k_2! * ... * k_(n_k)!)

    Args:
        Optional named:
        n (int): Non-negative integer giving the number chosen from
                for the n_k-nomial coefficients considered, i.e. the
                sum of the partition sizes in the n_k-nomial
                coefficients considered.
            Default: 2 * 10 ** 5
        n_k (int): Strictly positive integer giving the number of
                partitions in the multinomial coefficients considered.
            Default: 3
        factor_p_factorisation (dict): Dictionary representing a
                prime factorisation of the strictly positive integer
                by which the n_k-nomial coefficients for given n
                and n_k counted are to be divisible, whose keys are
                the prime numbers that appear in the prime
                factorisation of the integer in question, with the
                corresponding value being the number of times that
                prime appears in the factorisation (i.e. the power
                of that prime in the prime factorisation of the
                integer). An empty dictionary corresponds to the
                multiplicative identity (i.e. 1).
            Default: {2: 12, 5: 12} (the prime factorisation of 10 ** 12)

    Returns:
    Integer (int) giving the number of non-zero n_k-nomial coefficients
    with n as the number chosen from that are divisible by the integer with
    the prime factorisation factor_p_factorisation.

    TODO- reword for clarity

    Outline of rationale:
    TODO
    """
    # TODO- try to make faster
    since = time.time()
    p_lst = sorted(factor_p_factorisation.keys())
    n_p = len(p_lst)
    target = [factor_p_factorisation[p] for p in p_lst]
    #print(p_lst, target)

    counts_n = []
    for p in p_lst:
        counts_n.append(0)
        n_ = n
        while n_:
            n_ //= p
            counts_n[-1] += n_
    #print(f"counts_n = {counts_n}")
    counts = [tuple([0] * n_p)] * 2
    counts_curr = [0] * n_p
    k1_mn = ((n - 1) // n_k) + 1

    def recur(k_num_remain: int, k_sum_remain: int, last_k: int, repeat_streak: int, mult: int, facts_remain: List[int]) -> int:
        #print(facts_remain, target)
        if any(x < y for x, y in zip(facts_remain, target)): return 0
        k1_mn = ((k_sum_remain - 1) // k_num_remain) + 1
        #print(f"k_num_remain = {k_num_remain}, k_sum_remain = {k_sum_remain}, last_k = {last_k}, k1_mn = {k1_mn}")
        if k_num_remain == 2:
            #print("hi")
            res = 0
            if not k_sum_remain & 1:
                k1 = k_sum_remain >> 1
                #print(f"k2 = {k1}, {[x - 2 * y for x, y in zip(facts_remain, counts[k1])]}")
                if all(x - 2 * y >= t for t, x, y in zip(target, facts_remain, counts[k1])):
                    #print("hello1")
                    r_streak = (repeat_streak if k1 == last_k else 0) + 1
                    ans = mult * r_streak * (r_streak + 1)
                    #print(f"k2 = {k1}, k3 = {}, a")
                    res +=  mult // (r_streak * (r_streak + 1))
            k1_mn2 = k1_mn + (k1_mn << 1 == k_sum_remain)
            if last_k >= k1_mn2 and last_k <= k_sum_remain and all(x - y - z >= t for t, x, y, z in zip(target, facts_remain, counts[last_k], counts[k_sum_remain - last_k])):
                res += mult // (repeat_streak + 1)
                
            for k1 in range(k1_mn2, min(last_k, k_sum_remain + 1)):
                #print(f"k2 = {k1}, {[x - y - z for x, y, z in zip(facts_remain, counts[k1], counts[k_sum_remain - k1])]}")
                if all(x - y - z >= t for t, x, y, z in zip(target, facts_remain, counts[k1], counts[k_sum_remain - k1])):
                    #print("hello2")
                    r_streak = (repeat_streak if k1 == last_k else 0) + 1
                    res += mult // r_streak
            return res
        if last_k >= k1_mn and last_k <= k_sum_remain:
            k1 = last_k
            k_num_remain2 = k_num_remain - 1
            k_sum_remain2 = k_sum_remain - k1
            last_k2 = last_k
            repeat_streak2 = repeat_streak + 1
            mult2 //= repeat_streak2
            facts_remain2 = [x - y for x, y in zip(facts_remain, counts[k1])]
            res += recur(k_num_remain2, k_sum_remain2, last_k2, repeat_streak2, mult2, facts_remain2)
        for k1 in range(k1_mn, min(last_k, k_sum_remain + 1)):
            k_num_remain2 = k_num_remain - 1
            k_sum_remain2 = k_sum_remain - k1
            last_k2 = last_k
            facts_remain2 = [x - y for x, y in zip(facts_remain, counts[k1])]
            res += recur(k_num_remain2, k_sum_remain2, last_k2, 1, mult, facts_remain2)
        return res
    res = 0
    #print(f"k1_mn = {k1_mn}")
    for k1 in range(2, n + 1):
        print(f"k1 = {k1}")
        for i, p in enumerate(p_lst):
            k1_ = k1
            while True:
                k1_, r = divmod(k1_, p)
                if r: break
                counts_curr[i] += 1
        counts.append(tuple(counts_curr))
        if k1 < k1_mn: continue
        #print("hello", n_k - 1, n - k1, facts_remain)
        res += recur(n_k - 1, n - k1, k1, 1, math.factorial(n_k), [x - y for x, y in zip(counts_n, counts_curr)])
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 155
def findNewCapacitorCombinationValuesNoLessThanOne(max_n_capacitors: int) -> List[Set[Tuple[int, int]]]:
    """
    Finds the values of capacitances that are possible for
    combinations of a number of capacitors with unit capacitance
    for each number of capacitors up to max_n_capacitors, only
    including values that are not possible for any smaller
    number of capacitors.

    Two capacitor combinations with a certain number of capacitors
    can be combined to form a third by connecting the two either
    in series or in parallel. If the two capacitor combinations
    being combined together have capacitances C1 and C2, and
    then if they are combined in parallel then the resultant
    overall capacitance is (C1 + C2), while if they are combined
    in series then the resultant capacitance is 1 / (1 / C1 + 1 / C2).
    The combinations are built up from individual capacitors with
    unit capacitors.

    For example, if we consider the combinations that contain no
    more than two unit capacitors, the possible capacitances result
    from a single capacitor, with capacitance of 1 (trivially),
    a combination of two capacitors in parallel with capacitance
    (1 + 1) = 2, and a combination of two capacitors in series
    with capacitance 1 / (1 / 1 + 1 / 1) = 1 / 2.

    Args:
        Required positional:
        max_n_capacitors (int): The largest number of capacitors
                considered.
    
    Returns:
    List of sets of 2-tuples of ints, where the i:th element in
    the list is the set of fractions representing the new capacitances
    possible for i capacitors.
    """
    seen = {(1, 1)}
    curr = [set(), {(1, 1)}]
    for i in range(2, max_n_capacitors + 1):
        st = set()
        for j in range(1, (i >> 1) + 1):
            for frac1 in curr[j]:
                for frac2 in curr[i - j]:
                    for frac3 in (addFractions(frac1, frac2), addFractions(frac1, (frac2[1], frac2[0])), addFractions((frac1[1], frac1[0]), frac2)):
                        if frac3 not in seen:
                            st.add(frac3)
                            seen.add(frac3)
                    frac3 = addFractions((frac1[1], frac1[0]), (frac2[1], frac2[0]))
                    if frac3 not in seen and frac3[0] >= frac3[1]:
                        st.add(frac3)
                        seen.add(frac3)
                    frac3 = addFractions((frac1[1], frac1[0]), (frac2[1], frac2[0]))
                    frac3 = (frac3[1], frac3[0])
                    if frac3 not in seen and frac3[0] >= frac3[1]:
                        st.add(frac3)
                        seen.add(frac3)
        curr.append(st)
        print(f"n = {i}, number of new capacitances no less than one = {len(st)}")
        #print(i, st)
        #tot_set |= st
        print(f"n_capacitors = {i}, number of new capacitances no less than one = {len(st)}, cumulative n_combinations = {2 * sum(len(x) for x in curr) - 1}")
    return curr


def possibleCapacitorCombinationValuesNoLessThanOne(max_n_capacitors: int) -> List[Set[Tuple[int, int]]]:
    """
    Finds the values of capacitances that are possible for
    combinations of a number of capacitors with unit capacitance
    for each exact number of capacitors up to max_n_capacitors.

    Two capacitor combinations with a certain number of capacitors
    can be combined to form a third by connecting the two either
    in series or in parallel. If the two capacitor combinations
    being combined together have capacitances C1 and C2, and
    then if they are combined in parallel then the resultant
    overall capacitance is (C1 + C2), while if they are combined
    in series then the resultant capacitance is 1 / (1 / C1 + 1 / C2).
    The combinations are built up from individual capacitors with
    unit capacitors.

    For example, if we consider the combinations that contain no
    more than two unit capacitors, the possible capacitances result
    from a single capacitor, with capacitance of 1 (trivially),
    a combination of two capacitors in parallel with capacitance
    (1 + 1) = 2, and a combination of two capacitors in series
    with capacitance 1 / (1 / 1 + 1 / 1) = 1 / 2.

    Args:
        Required positional:
        max_n_capacitors (int): The largest number of capacitors
                considered.
    
    Returns:
    List of sets of 2-tuples of ints, where the i:th element in
    the list is the set of fractions representing the capacitances
    possible for combinations of exactly i unit capacitors.
    """
    curr = [set(), {(1, 1)}]
    cumu = 1
    tot_set = {(1, 1)}
    for i in range(2, max_n_capacitors + 1):
        st = set()
        for j in range(1, (i >> 1) + 1):
            for frac1 in curr[j]:
                for frac2 in curr[i - j]:
                    st.add(addFractions(frac1, frac2))
                    st.add(addFractions(frac1, (frac2[1], frac2[0])))
                    st.add(addFractions((frac1[1], frac1[0]), frac2))
                    frac3 = addFractions((frac1[1], frac1[0]), (frac2[1], frac2[0]))
                    if frac3[0] >= frac3[1]:
                        st.add(frac3)
                    frac3 = addFractions((frac1[1], frac1[0]), (frac2[1], frac2[0]))
                    if frac3[1] >= frac3[0]:
                        st.add((frac3[1], frac3[0]))
        curr.append(st)
        tot_set |= st
        print(f"n_capacitors = {i}, n_combinations = {2 * len(st) - ((1, 1) in st)}, cumulative n_combinations = {2 * len(tot_set) - ((1, 1) in tot_set)}")
    return curr

def countDistinctCapacitorCombinationValues(max_n_capacitors: int=18) -> int:
    """
    Solution to Project Euler #155

    Finds the number of distinct values of capacitances that are
    possible from at most max_n_capacitors combinations of capacitors
    with unit capacitance when connecting them in parallel or
    series.

    Two capacitor combinations with a certain number of capacitors
    can be combined to form a third by connecting the two either
    in series or in parallel. If the two capacitor combinations
    being combined together have capacitances C1 and C2, and
    then if they are combined in parallel then the resultant
    overall capacitance is (C1 + C2), while if they are combined
    in series then the resultant capacitance is 1 / (1 / C1 + 1 / C2).
    The combinations are built up from individual capacitors with
    unit capacitors.

    For example, if we consider the combinations that contain no
    more than two unit capacitors, the possible capacitances result
    from a single capacitor, with capacitance of 1 (trivially),
    a combination of two capacitors in parallel with capacitance
    (1 + 1) = 2, and a combination of two capacitors in series
    with capacitance 1 / (1 / 1 + 1 / 1) = 1 / 2.

    The solution to this problem is equivalent to the sequence given
    by OEIS A153588.

    Args:
        Optional named:
        max_n_capacitors (int): The largest number of capacitors
                in any combination considered.
            Default: 18
    
    Returns:
    Integer (int) giving the number of distinct capacitance values
    possible from combining up to max_n_capacitors capacitors with unit
    capacitors by combinating them in parallel and in series.
    
    The solution to this problem is equivalent to the sequence given
    by OEIS A153588.
    """
    since = time.time()
    new_combs_lst = findNewCapacitorCombinationValuesNoLessThanOne(max_n_capacitors)
    res = 2 * sum(len(x) for x in new_combs_lst) - 1
    """
    combs_lst = possibleCapacitorCombinationValuesNoLessThanOne(max_n_capacitors)
    tot_combs = combs_lst[0]
    for comb in combs_lst[1:]:
        tot_combs |= comb
    res = 2 * len(tot_combs) - ((1, 1) in tot_combs)
    """
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 156
def cumulativeDigitCount(d: int, n_max: int, base: int=10) -> int:
    """
    For a given base and the value of a non-zero digit in that base (i.e.
    an integer between 1 and (base - 1) inclusive), finds the number of
    times the digit appears in the collective representations in the chosen
    base of all the strictly positive integers no greater than a chosen
    integer.

    Args:
        Required positional:
        d (int): Non-zero integer between 1 and (base - 1) inclusive
                giving the value of the digit of interest in the chosen
                base.
        n_max (int): Integer giving the largest number considered when
                counting the occurrences of the digit d.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which d is a digit and in which all numbers are
                to be expressed when counting the number of
                occurrences of the digit d.
    
    Returns:
    Integer (int) giving the number of occurrences of the digit with
    value d in the collective representations in the base base of all
    the strictly positive integers no greater than n_max.
    """

    def countNumsLTBasePow(base_exponent: int) -> int:
        #if base_exponent == 0: return 0
        #return countNumsLTBasePow(base_exponent - 1) + base ** (base_exponent - 1)
        return 0 if base_exponent <= 0 else base_exponent * base ** (base_exponent - 1)
    
    res = 0
    n2 = n_max + 1
    base_exp = 0
    num2 = 0
    while n2:
        n2, d2 = divmod(n2, base)
        #print(base_exp, countNumsLTBasePow(base_exp))
        res += countNumsLTBasePow(base_exp) * d2
        if d2 > d:
            res += base ** (base_exp)
        elif d2 == d:
            res += num2
        num2 += d2 * base ** base_exp
        base_exp += 1
        #print(num2, res)
    
    return res

def cumulativeNonZeroDigitCountEqualsNumber(d: int, base: int=10) -> List[int]:
    """
    For a given base and the value of a non-zero digit in that base (i.e.
    an integer between 1 and (base - 1) inclusive), finds the all the
    integers such that the total of the number of occurrences of the digit
    d over the collective representations of all the strictly positive
    integers no greater than that integer in the chosen base is equal to
    that number.

    Args:
        Required positional:
        d (int): Non-zero integer between 1 and (base - 1) inclusive
                giving the value of the digit of interest in the chosen
                base.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which d is a digit and in which all numbers are
                to be expressed when counting the number of
                occurrences of the digit d.
            Default: 10
    
    Returns:
    List of integers (ints) giving the value of all the integers with
    the described property for the chosen base and digit in that base
    in strictly increasing order.

    Outline of rationale:
    TODO- particularly the justification of exp_max
    """
    exp_max = base
    res = []

    def recur(exp: int, pref: int=0, pref_d_count: int=0) -> None:
        #print(f"exp = {exp}, pref = {pref}, pref_d_count = {pref_d_count}")
        num = -1
        num2 = pref * base# + (not pref)
        while num2 > num and num2 < (pref + 1) * base:
            num = num2
            num2 = cumulativeDigitCount(d, num * base ** exp, base=base) // (base ** exp)
        if num2 >= (pref + 1) * base: return
        mn = num
        pref2 = num * base ** exp
        num = float("inf")
        num2 = (pref + 1) * base - 1
        while num2 < num and num2 >= mn:
            num = num2
            num2 = cumulativeDigitCount(d, (num + 1) * base ** exp - 1, base=base) // (base ** exp)
        if num2 < mn: return
        pref3 = num * base ** exp

        """
        lft, rgt = pref * base + (not pref), (pref + 1) * base
        while lft < rgt:
            mid = lft + ((rgt - lft) >> 1)
            cnt = cumulativeDigitCount(d, (mid + 1) * base ** exp - 1, base=base)
            if cnt < mid * base ** exp:
                lft = mid + 1
            else: rgt = mid
        if lft == (pref + 1) * base: return
        print("hi")
        pref2 = lft * base ** exp
        lft, rgt = lft, (pref + 1) * base - 1
        while lft < rgt:
            mid = rgt - ((rgt - lft) >> 1)
            cnt = cumulativeDigitCount(d, (mid) * base ** exp, base=base)
            print(f"exp = {exp}, mid = {mid}, num = {(mid + 1) * base ** exp - 1}, cnt = {cnt}")
            if cnt > (mid + 1) * base ** exp:
                rgt = mid - 1
            else: lft = mid
        pref3 = lft * base ** exp
        """
        rng = [pref2, pref3]
        #print(rng)
        #pref2 = (pref * base + (not pref)) * base ** exp
        #pref3 = ((pref + 1) * base ** (exp + 1)) - 1
        pref2_cnt = cumulativeDigitCount(d, pref2, base=base)
        #pref3_cnt = cumulativeDigitCount(d, pref3, base=base)
        #rng = [max(pref2, pref2_cnt), min(pref3, pref3_cnt)]
        #print(rng)
        #if rng[0] > rng[1]: return
        #print(rng, pref2, pref2_cnt, pref3, pref3_cnt)
        #pref0 = (pref // base) * base
        if not exp:
            # Review- optimise (consider cases pref_d_count = 0, pref_d_count > 1
            # and pref_d_count = 1 separately, with the latter being the most
            # complicated)
            #d0 = max(rng[0], int(pref == 0))
            #print(f"pref2_cnt = {pref2_cnt}")
            d0 = rng[0] % base
            d1 = rng[1] % base
            cnt = pref2_cnt + d0 * pref_d_count + (d < d0)
            #cnt_start = cumulativeDigitCount(d, start, base=base)
            num = pref2
            if num == cnt:
                #print("found!")
                #print(num, cumulativeDigitCount(d, num, base=10))
                res.append(num)
            #print(f"num = {num}, cnt = {cnt}")
            #print(d0, rng[1] + 1)
            for d2 in range(d0 + 1, d1 + 1):
                num += 1
                cnt += pref_d_count + (d2 == d)
                #print(d2, cnt, num)
                if cnt == num:
                    #print("found!")
                    #print(num, cumulativeDigitCount(d, num, base=10))
                    res.append(num)
            return
        rng2 = (rng[0] // base ** (exp), rng[1] // base ** (exp))
        #print(rng2[0], rng2[1] + 1)
        for pref4 in range(rng2[0], rng2[1] + 1):
            recur(exp=exp - 1, pref=pref4, pref_d_count=pref_d_count + (pref4 % base == d))
        return
    
    #for exp in range(0, exp_max + 1):
    #    recur(exp, pref=0, pref_d_count=0)
    recur(exp_max, pref=0, pref_d_count=0)
    #print(d, res)
    #print(sum(res))
    return res

def cumulativeNonZeroDigitCountEqualsNumberSum(base: int=10) -> int:
    """
    Solution to Project Euler #156

    Finds the sum of the following sums for all non-zero digit values
    in the chosen base (i.e. the integers between 1 and (base - 1)
    inclusive:
    
    For each non-zero digit value in the chosen base (i.e. an integer
    between 1 and (base - 1) inclusive), calculates the sum of all the
    integers such that the total of the number of occurrences of the digit
    d over the collective representations of all the strictly positive
    integers no greater than that integer in the chosen base is equal to
    that number. These sums are then added together to produce the final
    result.

    Note that if an integer has the described property for more than
    one digit, it will be included in the final sum a number of times
    equal to the number of digits for which it has that property.

    Args:
        Optional named:
        base (int): Integer strictly greater than 1 giving the base
                in which d is a digit and in which all numbers are
                to be expressed when counting the number of
                occurrences of the digit d.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of sums described.

    Outline of rationale:
    See Outline of rationale in the documentation for the function
    cumulativeNonZeroDigitCountEqualsNumber().
    """
    since = time.time()
    res = 0
    for d in range(1, base):
        res += sum(cumulativeNonZeroDigitCountEqualsNumber(d, base=base))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 157
def reciprocalPartnerSumsEqualToMultipleOfReciprocal(a: int, reciprocal: int) -> List[int]:
    """
    Given the strictly positive integers a and reciprocal, identifies
    all strictly positive integers b such that:
        1 / a + 1 / b = p / reciprocal
    for some strictly positive integer p.

    Args:
        Required positional:
        a (int): Strictly positive integer giving the value of a in
                the above equation for which the possible values of
                b should be found.
        reciprocal (int): Strictly positive integer giving the value
                of reciprocal in the above equation for which the
                possible values of b are to be found.
    
    Returns:
    List of strictly positive integers (int) giving all strictly
    positive b for which there exists strictly positive integer
    p for which the above equation holds for the given values
    of a and reciprocal. The list is sorted in increasing size
    of the identified values of b.
    """
    mult_min = (reciprocal) // a + 1
    b_max = (a * reciprocal) // (mult_min * a - reciprocal)
    b_step = a // gcd(a, reciprocal)
    b_min = b_step * (((a - 1) // b_step) + 1)
    #print(b_step, b_min, b_max)
    
    res = []
    for b in range(b_min, b_max + 1, b_step):
        if not reciprocal % addFractions((1, a), (1, b))[1]:
            res.append(b)
    return res

def reciprocalPairSumsEqualToMultipleOfReciprocal(reciprocal: int) -> List[Tuple[int]]:
    """
    Given the strictly positive integer reciprocal, identifies all
    pairs of strictly positive integers (a, b) such that b is no
    less than a and:
        1 / a + 1 / b = p / reciprocal
    for some strictly positive integer p.

    Args:
        Required positional:
        reciprocal (int): Strictly positive integer giving the value
                of reciprocal in the above equation for which the
                possible strictly positive integer pairs (a, b) are
                to be found.
    
    Returns:
    List of 2-tuples of strictly positive integers (int) giving all
    ordered pairs of strictly positive integers (a, b) such that b is
    no less than a and there exists strictly positive integer p for
    which the above equation holds for the given value of reciprocal.
    The list is sorted in increasing size of a, and pairs with the
    same value of a these are sorted in increasing size of b.
    """
    a_step = 1
    a_max = reciprocal * 2
    a_min = 1
    
    res = []
    for a in range(a_min, a_max + 1, a_step):
        for b in reciprocalPartnerSumsEqualToMultipleOfReciprocal(a, reciprocal):
            res.append((a, b))
    return res

def reciprocalPairSumsEqualToFraction(frac: Tuple[int]) -> List[Tuple[int]]:
    """
    For a strictly positive rational number frac, finds all ordered
    pairs of strictly positive integers (a, b) such that b is no
    less than a and:
        1 / a + 1 / b = frac
    
    Args:
        Required positional:
        frac (2-tuple of ints): The strictly positive rational number
                for which the ordered pairs (a, b) satisfying the
                above equation are to be sought, represented as a
                fraction with the strictly positive integers at
                indices 0 and 1 represent the numerator and denominator
                of the fraction respectively.

    Returns: 
    List of 2-tuples of strictly positive integers (int) giving all
    ordered pairs of strictly positive integers (a, b) such that b is
    no less than a and the above equation holds for the given value
    of frac. The list is sorted in increasing size of a.
    """
    g = gcd(*frac)
    frac = tuple(x // g for x in frac)

    if frac[0] > frac[1]:
        return [(1, frac[1])] if frac[0] == frac[1] + 1 else []
    res = []
    for q in range((frac[1] // frac[0]) + 1, (2 * frac[1] // frac[0]) + 1):
        frac2 = addFractions(frac, (-1, q))
        if frac2[0] == 1:
            res.append((q, frac2[1]))
    return res

def countReciprocalPairSumsEqualToFraction(frac: Tuple[int]) -> int:
    """
    For a strictly positive rational number frac, finds the number
    of distinct ordered pairs of strictly positive integers (a, b)
    that exist such that b is no less than a and:
        1 / a + 1 / b = frac
    
    Args:
        Required positional:
        frac (2-tuple of ints): The strictly positive rational number
                for which the number of ordered pairs (a, b) satisfying
                the above equation is to be sought, represented as a
                fraction with the strictly positive integers at
                indices 0 and 1 represent the numerator and denominator
                of the fraction respectively.

    Returns: 
    Integer (int) giving the number of distinct ordered pairs of
    strictly positive integers (a, b) that exists such tha b is no less
    than a and the above equation holds for the given value of frac.
    """
    g = gcd(*frac)
    frac = tuple(x // g for x in frac)

    if frac[0] > frac[1]:
        return int(frac[0] == frac[1] + 1)
    res = 0
    for q in range((frac[1] // frac[0]) + 1, (2 * frac[1] // frac[0]) + 1):
        frac2 = addFractions(frac, (-1, q))
        res += (frac2[0] == 1)
    return res

def reciprocalPairSumsMultipleOfReciprocal2(q_factorisation: Dict[int, int]) -> List[Tuple[int, int]]:
    """
    Given the prime factorisation of a strictly positive integer
    q_factorisation, identifies all distinct ordered pairs of
    strictly positive integers (a, b) such that b is no less than
    a and:
        1 / a + 1 / b = p / q
    for some strictly positive integer p, where q is the integer
    whose prime factorisation is q_factorisation.

    Args:
        Required positional:
        q_factorisation (dict): Dictionary representing a prime
                factorisation of the strictly positive integer q
                in the above equation, whose keys are the prime
                numbers that appear in the prime factorisation of
                q, with the corresponding value being the
                number of times that prime appears in the
                factorisation (i.e. the power of that prime in the
                prime factorisation of q). An empty dictionary
                corresponds to the multiplicative identity (i.e. 1).
    
    Returns:
    List of 2-tuples of strictly positive integers (int) giving all
    ordered pairs of strictly positive integers (a, b) such that b is
    no less than a and there exists strictly positive integer p for
    which the above equation holds for the positive integer q having
    the prime factorisation represented by q_factorisation.
    The list is sorted in increasing size of a, and pairs with the
    same value of a these are sorted in increasing size of b.
    """
    q = 1
    for k, v in q_factorisation.items():
        q *= k ** v
    #print(f"q = {q}")
    res = []
    for p in range(1, (5 * q) // 6 + 1):
        ans = reciprocalPairSumsEqualToFraction((p, q))
        #print(f"{p} / {q}: {ans}")
        res.extend(ans)
    res.append((2, 2))
    p_lst = list(q_factorisation.keys())
    def factorGenerator(idx: int, val: int) -> Generator[int, None, None]:
        if idx == len(p_lst):
            yield val
            return
        val2 = val
        for exp in range(q_factorisation[p_lst[idx]] + 1):
            yield from factorGenerator(idx + 1, val2)
            val2 *= p_lst[idx]
        return

    for factor in factorGenerator(0, 1):
        res.append((1, factor))
    #res += 1 + sum(v + 1 for v in q_factorisation.values())
    return res

def countReciprocalPairSumsMultipleOfReciprocal2(q_factorisation: Dict[int, int]) -> int:
    """
    Given the prime factorisation of a strictly positive integer
    q_factorisation, finds the number of distinct ordered pairs
    of strictly positive integers (a, b) such that b is no less
    than a and:
        1 / a + 1 / b = p / q
    for some strictly positive integer p, where q is the integer
    whose prime factorisation is q_factorisation.

    Args:
        Required positional:
        q_factorisation (dict): Dictionary representing a prime
                factorisation of the strictly positive integer q
                in the above equation, whose keys are the prime
                numbers that appear in the prime factorisation of
                q, with the corresponding value being the
                number of times that prime appears in the
                factorisation (i.e. the power of that prime in the
                prime factorisation of q). An empty dictionary
                corresponds to the multiplicative identity (i.e. 1).
    
    Returns:
    Integer (int) giving the number of distinct ordered pairs of
    strictly positive integers (a, b) that exists such tha b is no less
    than a and the above equation holds for some strictly positive
    integer p with the positive integer q having the prime factorisation
    represented by q_factorisation.
    """
    q = 1
    for k, v in q_factorisation.items():
        q *= k ** v
    #print(f"q = {q}")
    res = 0
    for p in range(1, (5 * q) // 6 + 1):
        ans = reciprocalPairSumsEqualToFraction((p, q))
        #print(f"{p} / {q}: {ans}")
        res += len(ans)
    print(res)
    term = 1
    for v in q_factorisation.values():
        term *= v + 1
    #res += 1 + sum(v + 1 for v in q_factorisation.values())
    return res + term + 1

def factorFactorisationsGenerator(num_p_factorisation: Dict[int, int]) -> Generator[Dict[int, int], None, None]:
    """
    Generator yielding the prime factorisations of each
    distinct positive integer factor of the strictly positive
    integer with the prime factorisation num_p_factorisation,
    with each factor being represented by exactly one yielded.
    value.

    Args:
        Required positional:
        num_p_factorisation (dict): Dictionary representing a prime
                factorisation of the strictly positive integer for
                which the prime factorisations of its factors are
                to be generated, whose keys are the prime numbers
                that appear in the prime factorisation of the integer
                in question, with the corresponding value being the
                number of times that prime appears in the factorisation
                (i.e. the power of that prime in the prime
                factorisation of the integer in question). An empty
                dictionary corresponds to the multiplicative identity
                (i.e. 1).
    
    Yields:
    Dictionary (dict) giving the prime factorisation of a positive
    integer factor of the positive integer with prime factorisation
    num_p_factorisation, whose keys are strictly positive integers
    (int) giving the prime numbers that appear in the prime
    factorisation of the factor in question, with the corresponding
    value being a strictly positive integer (int) giving the number
    of times that prime appears in the factorisation (i.e. the power
    of that prime in the prime factorisation of the factor in
    question). An empty dictionary corresponds to the multiplicative
    identity (i.e. 1).
    The prime factorisation of each distinct positive integer factor
    is yielded exactly once. As such, the number of values yielded
    is equal to the number of positive integer factors of the integer
    (for instance, for prime numbers, exactly two factorisations
    are yielded, represenging to the multiplicative identity and
    the number itself).
    Note that while generally later values yielded tend to represent
    larger integers than those of earlier values yielded (with the
    multiplicative identity factorisation guaranteed to be yielded
    first and the original factorisation last), it is possible for
    factorisations of integers to be yielded after factorisations
    of larger integers, and as such, this ordering should not be
    relied on.
    """
    p_lst = list(num_p_factorisation.keys())
    curr = {}
    def recur(idx: int) -> Generator[int, None, None]:
        if idx == len(p_lst):
            yield dict(curr)
            return
        p = p_lst[idx]
        for exp in range(num_p_factorisation[p] + 1):
            yield from recur(idx + 1)
            curr[p] = curr.get(p, 0) + 1
        if p in curr.keys(): curr.pop(p)
        return
    
    yield from recur(0)
    return

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

def countReciprocalPairSumsMultipleOfReciprocal(q_factorisation: Dict[int, int]) -> int:
    """
    Given the prime factorisation of a strictly positive integer
    q_factorisation, finds the number of distinct ordered pairs
    of strictly positive integers (a, b) such that b is no less
    than a and:
        1 / a + 1 / b = p / q
    for some strictly positive integer p, where q is the integer
    whose prime factorisation is q_factorisation.

    Args:
        Required positional:
        q_factorisation (dict): Dictionary representing a prime
                factorisation of the strictly positive integer q
                in the above equation, whose keys are the prime
                numbers that appear in the prime factorisation of
                q, with the corresponding value being the
                number of times that prime appears in the
                factorisation (i.e. the power of that prime in the
                prime factorisation of q). An empty dictionary
                corresponds to the multiplicative identity (i.e. 1).
    
    Returns:
    Integer (int) giving the number of distinct ordered pairs of
    strictly positive integers (a, b) that exists such tha b is no less
    than a and the above equation holds for some strictly positive
    integer p with the positive integer q having the prime factorisation
    represented by q_factorisation.
    """
    q = 1
    for k, v in q_factorisation.items():
        q *= k ** v
    res = 0
    #print(f"q = {q}")
    for d_fact in factorFactorisationsGenerator({k: 2 * v for k, v in q_factorisation.items()}):
        d = 1
        for k, v in d_fact.items():
            d *= k ** v
        if d > q: continue
        #print(d)
        #d_inv_fact = {k: 2 * v - d_fact.get(k, 0) for k, v in q_factorisation.items()}
        p_mult_fact = {k: min(v, 2 * q_factorisation[k] - v) for k, v in d_fact.items()}
        q_div = 1
        for k, v in q_factorisation.items(): q_div *= k ** (v - p_mult_fact.get(k, 0))

        numer1 = 1
        numer2 = 1
        for k, v in q_factorisation.items():
            numer1 *= k ** (d_fact.get(k, 0) - p_mult_fact.get(k, 0))
            numer2 *= k ** (2 * v - d_fact.get(k, 0) - p_mult_fact.get(k, 0))
        numer1 += q_div
        numer2 += q_div
        g = gcd(numer1, numer2)
        #print(f"d = {d}, g = {g}")
        #pf = ps.factorCount(g)
        #for p_fact in factorFactorisationsGenerator(p_mult_fact):
        #    p = 1
        #    for k, v in p_fact.items(): p *= k ** v
        #    print(p, d, ((q + d) // p, (q + q ** 2 // d) // p), ((q - d) // p, (q - q ** 2 // d) // p))
        pf = calculatePrimeFactorisation(g)
        for k, v in pf.items():
            p_mult_fact[k] = p_mult_fact.get(k, 0) + v
        ans = 1
        for v in p_mult_fact.values():
            ans *= v + 1
        res += ans
    print(f"result for q = {q}: {res}")
    return res


def countReciprocalPairSumsMultipleOfReciprocalPower2(reciprocal_factorisation: Dict[int, int]={2: 1, 5: 1}, min_power: int=1, max_power: int=9) -> int:
    """
    Given the prime factorisation of a strictly positive integer
    reciprocal_factorisation, finds the number of distinct ordered
    triples (a, b, n) such that a and b are strictly positive
    integers, b is no less than a, n is a non-negative integer
    between min_power and max_power inclusive and:
        1 / a + 1 / b = p / q ^ n
    for some strictly positive integer p, where q is the integer
    whose prime factorisation is reciprocal_factorisation.

    Args:
        Optional named:
        reciprocal_factorisation (dict): Dictionary representing a
                prime factorisation of the strictly positive integer
                q in the above equation, whose keys are the prime
                numbers that appear in the prime factorisation of
                q, with the corresponding value being the
                number of times that prime appears in the
                factorisation (i.e. the power of that prime in the
                prime factorisation of q). An empty dictionary
                corresponds to the multiplicative identity (i.e. 1).
            Default: {2: 1, 5: 1} (the prime factorisation of 10)
        min_power (int): Non-negative integer giving the smallest
                value of n considered for counted solutions.
            Default: 1
        max_power (int): Non-negative integer giving the largest
                value of n considered for counted solutions.
            Default: 9
    
    Returns:
    Integer (int) giving the number of distinct ordered triples
    (a, b, n) such that a and b are strictly positive integers,
    b is no less than a, n is a non-negative integer between
    min_power and max_power inclusive and the above equation holds
    for some strictly positive integer p with the positive integer
    q having the prime factorisation represented by
    reciprocal_factorisation.
    """
    since = time.time()
    b = 1
    b2 = 1
    for k, v in reciprocal_factorisation.items():
        b *= k ** (v * max_power)
        b2 *= k ** v
    #print(f"b = {b}, b2 = {b2}")
    res = 0
    for a in range(1, (5 * b) // 6 + 1):
        if not a % 1000:
            print(f"a = {a}, b = {b}")
        g = gcd(a, b)
        (a_, b_) = (a // g, b // g)
        ans = countReciprocalPairSumsEqualToFraction((a_, b_))
        if not ans: continue
        mx = (max_power - min_power)
        
        for p, v in reciprocal_factorisation.items():
            curr = g
            num = p ** v
            for i in range(mx + 1):
                curr, r = divmod(curr, num)
                if r: break
            else: continue
            mx = i
        #print(f"{p} / {q}: {ans}")
        res += ans * (mx + 1)
    #print(res)
    res += (max_power - min_power + 1)
    p_lst = list(reciprocal_factorisation.keys())
    def factorGenerator(idx: int, val: int) -> Generator[int, None, None]:
        if idx == len(p_lst):
            yield val
            return
        val2 = val
        for exp in range(reciprocal_factorisation[p_lst[idx]] * max_power + 1):
            yield from factorGenerator(idx + 1, val2)
            val2 *= p_lst[idx]
        return

    for factor in factorGenerator(0, 1):
        curr = factor
        r = 0
        ans = 0
        while not r and ans < (max_power - min_power + 1):
            ans += 1
            curr, r = divmod(curr, b2)
        res += ans
    #res += 1 + sum(v + 1 for v in q_factorisation.values())
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

def countReciprocalPairSumsMultipleOfReciprocalPower(reciprocal_factorisation: Dict[int, int]={2: 1, 5: 1}, min_power: int=1, max_power: int=9) -> int:
    """
    Solution to Project Euler #157

    Given the prime factorisation of a strictly positive integer
    reciprocal_factorisation, finds the number of distinct ordered
    triples (a, b, n) such that a and b are strictly positive
    integers, b is no less than a, n is a non-negative integer
    between min_power and max_power inclusive and:
        1 / a + 1 / b = p / q ^ n
    for some strictly positive integer p, where q is the integer
    whose prime factorisation is reciprocal_factorisation.

    Args:
        Optional named:
        reciprocal_factorisation (dict): Dictionary representing a
                prime factorisation of the strictly positive integer
                q in the above equation, whose keys are the prime
                numbers that appear in the prime factorisation of
                q, with the corresponding value being the
                number of times that prime appears in the
                factorisation (i.e. the power of that prime in the
                prime factorisation of q). An empty dictionary
                corresponds to the multiplicative identity (i.e. 1).
            Default: {2: 1, 5: 1} (the prime factorisation of 10)
        min_power (int): Non-negative integer giving the smallest
                value of n considered for counted solutions.
            Default: 1
        max_power (int): Non-negative integer giving the largest
                value of n considered for counted solutions.
            Default: 9
    
    Returns:
    Integer (int) giving the number of distinct ordered triples
    (a, b, n) such that a and b are strictly positive integers,
    b is no less than a, n is a non-negative integer between
    min_power and max_power inclusive and the above equation holds
    for some strictly positive integer p with the positive integer
    q having the prime factorisation represented by
    reciprocal_factorisation.
    """
    since = time.time()
    res = 0
    for exp in range(min_power, max_power + 1):
        q_factorisation = {k: v * exp for k, v in reciprocal_factorisation.items()}
        res += countReciprocalPairSumsMultipleOfReciprocal(q_factorisation)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 158
def countAllDifferentLetterStringsWithNSmallerLeftNeighbours(n_chars: int, max_len: int, n_smaller_left_neighbours: int) -> List[int]:
    """
    For strings constructed from an ordered alphabet consisting
    of n_chars characters such that no character is used more
    than once, calculates the number of such strings that can be
    formed where exactly n_smaller_left_neighbours of the letters
    in the string have the letter directly to its left be a
    character that appears earlier in the alphabet, for all string
    lengths not exceeding max_len.

    Args:
        Required positional:
        n_chars (int): The number of characters in the alphabet
                being considered.
        max_len (int): The largest length string considered.
                Should be between 1 and n_chars inclusive.
        n_smaller_left_neighbours (int): The exact number of
                letters in the strings counted for which the
                letter to its left in the string is a character
                that appears earlier in the alphabet that that
                of the letter in question.
    
    Returns:
    List of integers (ints) with length max_len + 1 where for
    non-negative integer i <= max_len, the i:th element (0-indexed)
    of the list gives the count of strings satisfying the conditions
    for strings with length i.
    """
    # TODO- try to derive the exact formula for n_smaller_left_neighbours = 1:
    #  (2^n - n - 1) * (26 choose n)
    # and generalise to any n_smaller_left_neighbours- look into Eulerian
    # numbers
    res = [0, 0] if n_smaller_left_neighbours else [1, n_chars]
    curr = [[1] * n_chars]
    for length in range(2, max_len + 1):
        prev = curr
        curr = []
        curr.append([0] * (n_chars - length + 1))
        curr[0][n_chars - length] = prev[0][n_chars - length + 1]
        for i in reversed(range(n_chars - length)):
            curr[0][i] = curr[0][i + 1] + prev[0][i + 1]
        for j in range(1, min(n_smaller_left_neighbours, len(prev)) + 1):
            curr.append([0] * (n_chars - length + 1))
            if j < len(prev):
                curr[j][n_chars - length] = prev[j][n_chars - length + 1]
                for i in reversed(range(n_chars - length)):
                    curr[j][i] = curr[j][i + 1] + prev[j][i + 1]
            # Transition from the previous with one fewer smaller neighbours
            cumu = 0
            for i in range(n_chars - length + 1):
                cumu += prev[j - 1][i]
                curr[j][i] += cumu
        #print(curr)
        res.append(sum(curr[-1]))
    return res

def maximumDifferentLetterStringsWithNSmallerLeftNeighbours(n_chars: int=26, max_len: int=26, n_smaller_left_neighbours: int=1) -> List[int]:
    """
    Solution to Project Euler #158

    For strings constructed from an ordered alphabet consisting
    of n_chars characters such that no character is used more
    than once, calculates the largest number of such strings of a 
    given length not exceeding max_len that can be formed where
    exactly n_smaller_left_neighbours of the letters in the string
    have the letter directly to its left be a character that
    appears earlier in the alphabet, for all string.

    Args:
        Optional named:
        n_chars (int): The number of characters in the alphabet
                being considered.
            Default: 26
        max_len (int): The largest length string considered.
                Should be between 1 and n_chars inclusive.
            Default: 26
        n_smaller_left_neighbours (int): The exact number of
                letters in the strings counted for which the
                letter to its left in the string is a character
                that appears earlier in the alphabet that that
                of the letter in question.
            Default: 1
    
    Returns:
    Integers (int) giving the largest count of strings of a given
    length no greater than max_len satisfying the specified
    conditions.
    """
    since = time.time()
    res = max(countAllDifferentLetterStringsWithNSmallerLeftNeighbours(n_chars, max_len, n_smaller_left_neighbours))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 159
def digitalRoot(num: int, base: int=10) -> int:
    """
    For a non-negative integer num, finds the digital
    root of that integer in a chosen base.

    The digital root of a non-negative integer in a
    given base is defined recursively as follows:
     1) If the integer is strictly less than base, then
        it is equal to the value of that integer.
     2) Otherwise, it is equal to the digital root of the
        integer calculated by summing the value of the
        digits of the representation of the intger in the
        given base.
    For any base greater than 1, this definition uniquely
    defines the digital root, given that for any integer
    no less than base, the sum of its digits in the chosen
    base has an integer value no less than 1 and strictly
    less than the value of the integer. This guarantees
    that step 2 will only need to be applied a finite number
    of times before reaching a number between 1 and
    (base - 1) inclusive for any integer (with an upper
    bound on the number of applications being the value of
    the integer itself).

    Args:
        Required positional:
        num (int): The strictly positive integer whose
                digital root is to be calculated.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the
                base in which the integers are to be represented
                when finding the sum of digits.
            Default: 10
    
    Returns:
    Integer (int) between 1 and (base - 1) inclusive giving
    the digital root of num in the chosen base, as defined
    above.
    """
    #num0 = num
    while num >= base:
        num2 = 0
        while num:
            num, r = divmod(num, base)
            num2 += r
        num = num2
    #print(num0, num)
    return num

def maximalDigitalRootFactorisations(n_max: int, base: int=10) -> List[int]:
    """
    Calculates the maximal digital root sum in the chosen
    base for every non-negative integer no greater than
    n_max.

    The digital root of a strictly positive integer in a
    chosen base is defined recursively as follows:
     1) If the integer is strictly less than base, then
        it is equal to the value of that integer.
     2) Otherwise, it is equal to the digital root of the
        integer calculated by summing the value of the
        digits of the representation of the intger in the
        chosen base.
    For any base greater than 1, this definition uniquely
    defines the digital root, given that for any integer
    no less than base, the sum of its digits in the chosen
    base has an integer value no less than 1 and strictly
    less than the value of the integer. This guarantees
    that step 2 will only need to be applied a finite number
    of times before reaching a number between 1 and
    (base - 1) inclusive for any integer (with an upper
    bound on the number of applications being the value of
    the integer itself).

    The digital root sum of a postive integer factorisation is
    the sum of the digital roots of the terms in the
    factorisation, where repeated factors are included in the
    sum as many times as they appear in the factorisation.

    The maximal digital root sum of a non-negative integer is
    defined to be 0 for the integers 0 and 1 and the largest
    digital root sum of its possible integer factorisations
    in which 1 does not appear as a factor.

    Args:
        Required positional:
        n_max (int): The largest non-negative integer for
                which the maximal digit root sum is to be
                calculated.
        
        Optional named:
        base (int): Integer strictly greater than 1 giving the
                base in which the integers are to be represented
                when finding the digital roots.
            Default: 10
    
    Returns:
    List of integers (int) with length (n_max + 1), for which
    the i:th index (using 0-indexing) gives the maximal digit
    root sum of the integer i in the chosen base.
    
    Outline of rationale:
    For any integer strictly greater than 1, all possible
    positive integer factorisations with no 1 factors except
    for the factorisation consisting of the integer itself
    (with no other factors and the integer appearing only once)
    include an integer between 1 and the original integer
    exclusive. For any of the latter factorisations, consider
    the result obtained by partitioning the factors into two
    non-empty partitions. The result is two factorisations of 
    positive integers greater than 1 and strictly less than the
    original integer.
    Suppose the original factorisation gives rise to a maximal
    digital root sum for the original integer. By the definition
    of the digital root sum of a positive integer factorisation,
    the value of this maximal digital root sum is equal to the
    sum of the digital root sums of the two new factorisations.
    Suppose the digital root sum of one of these new factorisations
    is not a maximal digital root sum for the integer for which it
    is a factorisation (which, as previously observed, is strictly
    smaller than the original integer). If we now consider a
    positive integer factorisation of this integer with no 1 factors
    whose digital root sum is a maximal digital root sum for that
    integer, its digital root sum is by definition larger than that
    of the previous factorisation. Re-inserting the integers in the
    other partition into this factorisation, we obtain a new
    factorisation of the original integer, whose digital root
    sum is equal to that digital root sum plus the digital root sum
    of the factorisation in the other partition, giving a digital
    root sum strictly greater than the original factorisation of the
    original integer, which we supposed was the maximal digital root
    sum of the integer. Given that this new factorisation contains
    only positive integers and no 1s, by the definition of the
    maximal digital root sum of the integer its digital root sum
    cannot exceed that of the maximal digital root sum, so we have a
    contradiction.
    Consequently, the maximal digital root sum of an integer strictly
    greater than 1 is either the digital sum of the integer itself
    or the sum of the maximal digit sum of two smaller integers
    whose product is equal to that integer.
    Thus, once all the maximal digital root sums of smaller integers
    strictly greater than 1 are known, we can calculate the maximal
    digit root sum of an integer by taking the maximum values of
    the digit sum of the integer itself and for each non-unit factor
    pair (i.e. pairs of integers strictly greater than 1 whose product
    is the integer in question), the sum of the two factors' maximal
    digital root sums. This enables iterative calculation of all the
    maximal digital root sums up to an arbitrarily large value.
    We further optimise this process by using a sieve approach
    (similar to the sieve of Eratosthenes in prime searching)
    to circumvent the need to find factors (which would involve the
    expensive integer division operation), whereby after each maximal
    digital root sum is calculated, the multiples up to the maximum
    value or the square of the current integer (whichever is smaller)
    are updated to take the maximum value of their current value
    and the sum of the maximal digital root sum just calculated and
    that of the integer being multiplied by (which, given that it
    is no larger than the integer being considered will have already
    been calculated). Thus, when reaching a new integer, all that
    needs to be calculated to find the maximal digital root sum is
    the digital sum of that integer, to check whether that exceeds
    the existing maximum value.
    TODO- revise wording of rationale for clarity
    """
    res = [0] * (n_max + 1)
    for n in range(2, n_max + 1):
        res[n] = max(res[n], digitalRoot(n, base=base))
        for n2 in range(2, min(n, n_max // n) + 1):
            m = n2 * n
            res[m] = max(res[m], res[n] + res[n2])
    return res

def maximalDigitalRootFactorisationsSum(n_min: int=2, n_max: int=10 ** 6 - 1, base: int=10) -> int:
    """
    Solution to Project Euler #159

    Calculates the sum of the maximal digital root sums in
    the chosen base over every non-negative integer between
    n_min and n_max inclusive.

    The digital root of a strictly positive integer in a
    chosen base is defined recursively as follows:
     1) If the integer is strictly less than base, then
        it is equal to the value of that integer.
     2) Otherwise, it is equal to the digital root of the
        integer calculated by summing the value of the
        digits of the representation of the intger in the
        chosen base.
    For any base greater than 1, this definition uniquely
    defines the digital root, given that for any integer
    no less than base, the sum of its digits in the chosen
    base has an integer value no less than 1 and strictly
    less than the value of the integer. This guarantees
    that step 2 will only need to be applied a finite number
    of times before reaching a number between 1 and
    (base - 1) inclusive for any integer (with an upper
    bound on the number of applications being the value of
    the integer itself).

    The digital root sum of a postive integer factorisation is
    the sum of the digital roots of the terms in the
    factorisation, where repeated factors are included in the
    sum as many times as they appear in the factorisation.

    The maximal digital root sum of a non-negative integer is
    defined to be 0 for the integers 0 and 1 and the largest
    digital root sum of its possible integer factorisations
    in which 1 does not appear as a factor.

    Args:
        Optional named:
        n_min (int): The smallest strictly positive integer for
                which the maximal digit root sum is to be
                included in the sum.
            Default: 2
        n_max (int): The largest strictly positive integer for
                which the maximal digit root sum is to be
                included in the sum.
            Default: 10 ** 6 - 1
        base (int): Integer strictly greater than 1 giving the
                base in which the integers are to be represented
                when finding the digital roots.
            Default: 10
    
    Returns:
    Integer (int) giving the sum of the maximal digital root sums
    in the chosen base over every non-negative integer between
    n_min and n_max inclusive.

    Outline of rationale:
    See documentation for maximalDigitalRootFactorisations().
    """
    since = time.time()
    arr = maximalDigitalRootFactorisations(n_max, base=10)
    #print(arr)
    res = sum(arr[n_min:])
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 160


# Problem 161
def nextTriominoStates(state: Tuple[Tuple[int, bool]], rows_remain: int=-1, nxt_insert: Optional[int]=None) -> List[Tuple[List[Tuple[int, bool]], int, int]]:
    # Boolean represents whether there is an overhang caused by
    # an L-piece
    if rows_remain < 0:
        rows_remain = float("inf")
    n = len(state)
    lvl_delta0 = 0
    if nxt_insert is None or state[nxt_insert][0]:
        mn_state = min(x[0] for x in state)
        if mn_state:
            lvl_delta0 = mn_state
            state = [(x[0] - mn_state, x[1]) for x in state]
        for i in range(n):
            if not state[i][0]:
                nxt_insert = i
                break
    
    res = []
    def addState(state2: List[Tuple[int, bool]], piece_mn: Tuple[int, int], piece_rng: Tuple[int, int]) -> None:
        
        mn = piece_mn
        for i in range(piece_rng[1] + 1, n):
            mn = min(mn, (state2[i][0], i))
            if not mn: break
        if mn[0] >= 1:
            for i in range(piece_rng[0]):
                mn = min(mn, (state2[i][0], i))
                if state2[i][0] <= 1: break
        state2 = tuple((x[0] - mn[0], x[1]) for x in state2)
        #if state2 == ((1, False), (1, False), (0, False), (0, False), (0, False), (1, False)):
        #    print(piece_rng, piece_mn, mn)
        res.append((state2, lvl_delta0 + mn[0], mn[1]))
    
    # Pieces that can fill in an overhang space
    #if state == ((1, False), (1, False), (0, False), (0, False), (0, False), (1, False)):
    #    print("hello", nxt_insert)
    # Flat line piece
    if rows_remain >= 1 and nxt_insert + 2 < n and not state[nxt_insert + 1][0] and not state[nxt_insert + 2][0] and\
            not (nxt_insert + 3 < n and state[nxt_insert + 3] == (0, True)):
        state2 = list(state)
        mn = (float("inf"), 0)
        for i in range(nxt_insert, nxt_insert + 3):
            state2[i] = (1 + state[i][1], False)
            mn = min(mn, (state2[i][0], i))
        addState(state2, mn, (nxt_insert, nxt_insert + 2))
        #print(state)
        """
        for i in range(nxt_insert + 3, n):
            mn = min(mn, state2[i][0])
            if not mn: break
        if mn > 1:
            for i in range(nxt_insert):
                mn = min(mn, state2[i][0])
                if mn <= 1: break
        if mn: state2 = [(x[0] - mn, x[1]) for x in state2]
        res.append((state2, lvl_delta0 + mn))
        """
    if rows_remain <= 1: return res

    # Backward R-piece
    if nxt_insert + 1 < n and not state[nxt_insert + 1][0] and not state[nxt_insert + 1][1] and\
            not (nxt_insert + 2 < n and state[nxt_insert + 2] == (0, True)):
        state2 = list(state)
        state2[nxt_insert] = (1 + state[nxt_insert][1], False)
        state2[nxt_insert + 1] = (2, False)
        mn = min((state2[nxt_insert][0], nxt_insert), (2, nxt_insert + 1))
        addState(state2, mn, (nxt_insert, nxt_insert + 1))

    if state[nxt_insert][1]: return res

    # The other pieces

    # Upright line piece
    if rows_remain >= 3 and not (nxt_insert + 1 < n and state[nxt_insert + 1] == (0, True)):
        state2 = list(state)
        state2[nxt_insert] = (3, False)
        mn = (3, nxt_insert)
        addState(state2, mn, (nxt_insert, nxt_insert))
    
    # R piece
    if nxt_insert + 1 < n and not state[nxt_insert + 1][0] and\
            not (nxt_insert + 2 < n and state[nxt_insert + 2] == (0, True)):
        state2 = list(state)
        state2[nxt_insert] = (2, False)
        state2[nxt_insert + 1] = (1 + state[nxt_insert + 1][1], False)
        mn = min((2, nxt_insert), (state2[nxt_insert + 1][0], nxt_insert + 1))
        addState(state2, mn, (nxt_insert, nxt_insert + 1))
    
    # L piece
    if nxt_insert + 1 < n and (state[nxt_insert + 1][0] <= 1 and state[nxt_insert + 1] != (0, True)) and\
            not (nxt_insert + 2 < n and state[nxt_insert + 2] == (0, True)) and\
            (state[nxt_insert + 1][0] == 1 or (nxt_insert + 2 < n and state[nxt_insert + 2] == (0, False))):
        state2 = list(state)
        state2[nxt_insert] = (2, False)
        state2[nxt_insert + 1] = (2, False) if state[nxt_insert + 1][0] == 1 else (0, True)
        mn = min((2, nxt_insert), (state2[nxt_insert + 1][0], nxt_insert + 1))
        addState(state2, mn, (nxt_insert, nxt_insert + 1))
    
    # Backwards L piece
    if nxt_insert > 0 and (state[nxt_insert - 1][0] == 1):
        state2 = list(state)
        state2[nxt_insert - 1] = (2, False)
        state2[nxt_insert] = (2, False)
        mn = (2, nxt_insert - 1)
        addState(state2, mn, (nxt_insert - 1, nxt_insert))
    
    return res

def triominoStateComplementRotated(state: Tuple[Tuple[int, bool]]) -> Tuple[Tuple[Tuple[int, bool]], int]:
    # Assumes state1 has at least one entry at level 0
    h = max(x[0] for x in state)
    state2 = []
    for x, b in state:
        if not b: state2.append((h - x, False))
        else: state2.append((h - x - 2, True))
    return (tuple(state2[::-1]), h)

def triominoAreaFillCombinations(n_rows: int=9, n_cols: int=12) -> int:
    """
    Solution to Project Euler #161
    """
    since = time.time()
    n_triominos, r = divmod(n_rows * n_cols, 3)
    if r: return 0
    #print(n_triominos)
    n_cols, n_rows = sorted([n_rows, n_cols])
    #n_crow, n_rows = sorted([n_rows, n_cols])
    states_counts = {(tuple((0, False) for _ in range(n_cols)), 0, 0): 1}
    
    for _ in range(n_triominos):#(n_triominos) >> 1):
        new_states_counts = {}
        for (state, lvl, nxt_insert), cnt in states_counts.items():
            if lvl > n_rows - 3 and lvl + max(x[0] + 2 * x[1] for x in state) > n_rows:
                continue
            for (state2, lvl_delta, nxt_insert2) in nextTriominoStates(state, n_rows - lvl, nxt_insert=nxt_insert):
                lvl2 = lvl + lvl_delta
                x = (state2, lvl2, nxt_insert2)
                new_states_counts[x] = new_states_counts.get(x, 0) + cnt
        states_counts = new_states_counts
        #print(states_counts)
    res = sum(states_counts.values())
    """
    print("states:")
    for (state, lvl, nxt_insert), cnt in states_counts.items():
        print(state, lvl, nxt_insert, cnt)
    #print(states_counts)
    res = 0
    if n_triominos & 1:
        states_complement_counts = {}
        complements_map = {}
        for (state, lvl, nxt_insert), cnt in states_counts.items():
            state_c, h = triominoStateComplementRotated(state)
            complements_map[state] = state_c
            print(f"state = {state}")
            print(f"state_c = {state_c}")
            #states_complement_counts[(state_c, n_rows - lvl - h)] = cnt
            states_complement_counts[state_c] = cnt
        for (state, lvl, nxt_insert), cnt in states_counts.items():
            #state_c = complements_map[state]
            for (state2, lvl_delta, nxt_insert2) in nextTriominoStates(state, nxt_insert=nxt_insert):
                print(f"centre state = {state}")
                #if states_complement_counts.get(state2, 0) != cnt or state_c != state2:
                res += states_complement_counts.get(state2, 0) * cnt
                #else: res += cnt * (cnt - 1)
    else:
        #print("middle states:")
        #states_complement_counts = {}
        for (state, lvl, nxt_insert), cnt in states_counts.items():
            state_c, h = triominoStateComplementRotated(state)
            print(state, state_c)
            if state_c == state:
                print(f"self-match: {state}, {cnt}")
                res += cnt ** 2
            elif state_c in states_counts.keys():
                print(f"match: {state}, {cnt}, {state_c}, {states_counts.get(state_c, 0)}")
                res += states_counts.get(state_c, 0) * cnt
    """
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem 162
def intAsHexadecimal(num: int) -> str:
    """
    For a non-negative integer, finds its representation as a
    hexadecimal number, where the digits with value between 10
    and 15 inclusive are represented by "A", "B", "C", "D", "E"
    and "F" in that order. The answer is given without leading
    zeros (except in the case of zero itself which is returned
    as "0").

    Args:
        Required positional:
        num (int): Non-negative integer giving the integer whose
                representation as a hexadecimal number is being
                sought.
    
    Returns:
    String (str) giving the hexadecimal representation of num
    without leading zeros.
    """
    conv_dict = {x: str(x) for x in range(10)}
    for i, l in enumerate("ABCDEF", start=10):
        conv_dict[i] = l
    res = []
    while num:
        num, d = divmod(num, 16)
        res.append(conv_dict[d])
    res = res[::-1]
    return "".join(res) if res else "0"

def countIntegersContainGivenDigits(max_n_dig: int, n_contained_digs: int, contained_includes_zero: bool, base: int=10) -> int:
    """
    For a given base, calculates the number of strictly positive
    integers which when expressed in the chosen base with no
    leading zeros contain at most max_n_dig digits and contain each
    of n_contained_digs selected, distinct digits (with zero one of
    these digits if and only if contained_includes_zero is given
    as True) at least once.

    Args:
        Required positional:
        max_n_dig (int): Strictly positive integer giving the
                maximum number of digits the numbers considered
                can have when expressed in the chosen base
                without leading zeros.
        n_contained_digs (int): Non-negative integer giving the
                number of selected  distinct digits that must each
                be present in the representation in the chosen
                base of the integers counted without leading zeros.
        contained_includes_zero (bool): If given as True then
                zero is one of the digits in the n_contained_digs
                selected that must be present at least once in each
                number counted when expressed in the chosen base,
                otherwise zero is not one of these digits.
        
        Optional named:
        base (int): The base in which integers are to be expressed
                when judging how many digits it has and whether it
                contains all of the selected distinct digits.
    
    Returns:
    Integer (int) giving the number of strictly positive integers
    satisfying the requirements described above.
    """

    if n_contained_digs > base - (not contained_includes_zero):
        return 0
    memo = {}
    def recur(n_dig: int, n_contained_digs: int, contained_includes_zero: bool, first: bool=True) -> int:
        if n_contained_digs > n_dig or (not n_contained_digs and contained_includes_zero): return 0
        elif not n_dig: return 1
        args = (n_dig, n_contained_digs, contained_includes_zero, first)
        if args in memo.keys(): return memo[args]
        n_opts = base - first
        n_other_opts = n_opts - n_contained_digs + (first and contained_includes_zero)
        n_nonzero_contained_opts = n_contained_digs - contained_includes_zero
        res = n_other_opts * recur(n_dig - 1, n_contained_digs, contained_includes_zero, first=False)
        res += n_nonzero_contained_opts * recur(n_dig - 1, n_contained_digs - 1, contained_includes_zero, first=False)
        if not first and contained_includes_zero:
            res += recur(n_dig - 1, n_contained_digs - 1, False, first=False)
        memo[args] = res
        return res

    res = sum(recur(n_dig, n_contained_digs, contained_includes_zero, first=True) for n_dig in range(n_contained_digs, max_n_dig + 1))
    return res
    
def countHexadecimalIntegersContainGivenDigits(max_n_dig: int=16, n_contained_digs: int=3, contained_includes_zero: bool=True) -> str:
    """
    Solution to Project Euler #162

    Calculates the number of strictly positive integers which when
    represented as a hexadecimal number with no leading zeros contain
    at most max_n_dig digits and contain each of n_contained_digs
    selected, distinct digits (with zero one of these digits if and
    only if contained_includes_zero is given as True) at least once.
    The answer is given as a string giving the hexadecimal
    representation of this total (see documentation of
    intAsHexadecimal() for more detail about this representation).

    Args:
        Required positional:
        max_n_dig (int): Strictly positive integer giving the
                maximum number of digits the numbers considered
                can have when represented as a h
                without leading zeros.
        n_contained_digs (int): Non-negative integer giving the
                number of selected  distinct digits that must each
                be present in the hexadecimal representation of the
                integers counted without leading zeros.
        contained_includes_zero (bool): If given as True then
                zero is one of the digits in the n_contained_digs
                selected that must be present at least once in each
                number counted when expressed as a hexadecimal
                number, otherwise zero is not one of these digits.
    
    Returns:
    String (str) giving hexadecimal representation of the number of
    strictly positive integers satisfying the requirements described
    above.
    """
    since = time.time()
    res1 = countIntegersContainGivenDigits(max_n_dig, n_contained_digs, contained_includes_zero, base=16)
    print(res1)
    res2 = intAsHexadecimal(res1)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res2

# Problem 164
def countIntegersConsecutiveDigitSumCapped(
        n_digs: int=20,
        n_consec: int=3,
        consec_sum_cap: int=9,
        base: int=10,
) -> int:
    """
    Solution to Project Euler #164

    For a given base, finds the number of strictly positive integers
    which when represented in that base, contains exactly n_digs digits
    and the sum of any n_consec consecutive digits in that representation
    has a sum no greater than consec_sum_cap.

    Args:
        Optional named:
        n_digs (int): The number of digits (without leading zeros) in
                the representation in the chosen base of integers
                considered.
            Default: 20
        n_consec (int): The number of consecutive digits in the
                representation in the chosen base of any of the
                integers considered that must never exceed consec_sum_cap
                in order for the integer to be counted.
            Default: 3
        consec_sum_cap (int): The maximum value any n_consec consecutive
                digits in the representation in the chosen base of
                any of the integers considered can have in order for the
                integer to be counted.
            Default: 9
        base (int): Integer strictly greater than 1 giving the base
                in which the integers are to be represented when selecting
                which integers to be considered and which of these are
                to be counted.
            Default: 10

    Returns:
    Integer (int) giving the number of strictly positive integers which
    when represented in that base, contains exactly n_digs digits and the
    sum of any n_consec consecutive digits in that representation has a sum
    no greater than consec_sum_cap.
            
    Brief outline of rationale:
    This is calculated using bottom-up dynamic programming over the
    digits in the representation of the integer in the chosen base from
    left to right, keeping and updating the record of the number of valid
    integers with each combination of the latest (n_consec - 2) digits
    (including the digit under current consideration) and whose digit
    (n_consec - 2) places to the left of the current digit is no greater
    than each possible digit, as we consider each digit in turn from left
    to right.
    """
    since = time.time()
    if consec_sum_cap >= (base - 1) * n_consec:
        return (base - 1) * base ** (n_digs - 1)
    if n_consec == 1:
        return consec_sum_cap * (consec_sum_cap + 1) ** (n_digs - 1)

    def buildNDimensionalArrayIndexSumCapped(n_dim: int, index_max: int, index_sum_max: int) -> list:
        index_max = min(index_max, index_sum_max)
        #if n_dim == 1:
        #    return [0] * (min(index_max, index_sum_max) + 1)

        def recur(dim_idx: int, idx_sum: int) -> Union[list, int]:
            if dim_idx == n_dim - 1:
                return [0] * (min(index_max, index_sum_max - idx_sum) + 1)
            res = []
            for idx in range(min(index_max, index_sum_max - idx_sum) + 1):
                res.append(recur(dim_idx + 1, idx_sum + idx))
            return res
        return recur(0, 0)
    
    def arraySliceGenerator(arr: list, max_n_indices: Optional[int]) -> Generator[Tuple[Union[list, int], Tuple[int]], None, None]:
        #print("hi")
        if max_n_indices is None: max_n_indices = float("inf")
        #print(max_n_indices)
        curr = []
        def recur(arr_slice: Union[int, list], dim_idx: int) -> Generator[Tuple[Union[list, int], Tuple[int]], None, None]:
            #print("hello")
            #print(arr_slice, dim_idx)
            if isinstance(arr_slice, int) or dim_idx == max_n_indices:
                yield (arr_slice, tuple(curr))
                return
            res = []
            curr.append(0)
            for idx in range(len(arr_slice)):
                #print(f"idx = {idx}")
                curr[-1] = idx
                yield from recur(arr_slice[idx], dim_idx + 1)
            curr.pop()
            return
        yield from recur(arr, 0)
    
    def getNDimensionalArraySlice(arr: list, inds: Tuple[int]) -> Union[list, int]:
        arr2 = arr
        for idx in inds:
            arr2 = arr2[idx]
        return arr2
    
    def accumulateLastDimension(arr: list, n_dim: int) -> None:
        for arr_slice, _ in arraySliceGenerator(arr, n_dim - 1):
            for idx in range(1, len(arr_slice)):
                arr_slice[idx] += arr_slice[idx - 1]
        return
    
    arr = buildNDimensionalArrayIndexSumCapped(n_consec - 1, base - 1, consec_sum_cap)
    #print(arr)
    #for arr_slice, idx in arraySliceGenerator(arr, n_consec - 1):
    #    print(idx, arr_slice)
    base_eff = min(base, consec_sum_cap + 1)
    curr = buildNDimensionalArrayIndexSumCapped(n_consec - 1, base - 1, consec_sum_cap)#[[0] * min(base_eff, consec_sum_cap - d1 + 1) for d1 in range(base_eff)]
    if n_consec == 2:
        for d in range(len(curr)):
            curr[d] = d
    else:
        inds = [0] * (n_consec - 2)
        #for curr_slice, curr_inds in arraySliceGenerator(curr, n_consec - 2):
        #arr_slice = getNDimensionalArraySlice(curr, inds)
        for d1 in range(1, len(curr)):
            inds[0] = d1
            arr_slice = getNDimensionalArraySlice(curr, inds)
            for d2 in range(len(arr_slice)):
                arr_slice[d2] = 1
    #for curr_slice, curr_inds in arraySliceGenerator(curr, n_consec - 2):
        #inds[0] = d1
        #arr_slice = getNDimensionalArraySlice(curr, inds)
        #for d2 in range(1, len(curr_slice)):
        #    curr_slice[d2] = 1
    #for d1 in range(1, base_eff):
    #    for d2 in range(len(curr[d1])):
    #        curr[d1][d2] = 1
    #print(curr)
    for _ in range(n_digs - 1):
        prev = curr
        curr = buildNDimensionalArrayIndexSumCapped(n_consec - 1, base - 1, consec_sum_cap)#[[0] * min(base_eff, consec_sum_cap - d1 + 1) for d1 in range(base_eff)]
        for curr_slice, curr_inds in arraySliceGenerator(curr, n_consec - 2):
            curr_inds_sum = sum(curr_inds)
            for d2 in range(min(base_eff, consec_sum_cap - curr_inds_sum + 1)):
                prev_inds = (*curr_inds[1:], d2) if n_consec > 2 else ()
                prev_slice = getNDimensionalArraySlice(prev, prev_inds)
                curr_slice[d2] += prev_slice[consec_sum_cap - curr_inds_sum - d2]
        #for d1 in range(base_eff):
        #    for d2 in range(min(base_eff, consec_sum_cap - d1 + 1)):
        #        curr[d1][d2] += prev[d2][consec_sum_cap - d1 - d2]
        #print(curr)
        accumulateLastDimension(curr, n_consec - 1)
        #for d1 in range(base_eff):
        #    for d2 in range(1, len(curr[d1])):
        #        curr[d1][d2] += curr[d1][d2 - 1]
        #print(curr)
    #res = sum(curr[d][-1] for d in range(base_eff))
    res = sum(x[0][-1] for x in arraySliceGenerator(curr, n_consec - 2))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res


    """
    consec_sum_counts = [[0] * (min((base - 1) * (i + 1), consec_sum_cap) + 1) for i in range(n_consec - 1)]
    for d in range(1, min(consec_sum_cap + 1, base)):
        consec_sum_counts[0][d] = 1
    res = 0
    for _ in range(n_digs - 1):
        #prev_consec_sum_counts = consec_sum_counts
        #consec_sum_counts = [[0] * (min((base - 1) * i, consec_sum_cap) + 1) for i in range(n_consec - 1)]
        for j in range(len(consec_sum_counts[n_consec - 2])):
            res += min(base, consec_sum_cap) * consec_sum_counts[n_consec - 2][j]
        #for d in range(min(consec_sum_cap + 1, base)):
        #    res += consec_sum_counts[n_consec - 1][consec_sum_cap - d]
        for i in reversed(range(1, n_consec - 1)):
            consec_sum_counts[i] = [0] * (min((base - 1) * (i + 1), consec_sum_cap) + 1)
            for j in range(len(consec_sum_counts[i])):
                for d in range(min(base, j + 1)):
                    consec_sum_counts[i][j] += consec_sum_counts[i - 1][j - d]
        consec_sum_counts[0][0] = 1
        print(consec_sum_counts, res)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    
    return res
    """


# Problem 169
def sumOfPowersOfTwo(num: int=10 ** 25, max_rpt: int=2) -> int:
    """
    Solution to Project Euler #169

    For a non-negative integer, finds the number of ways that number
    can be expressed as the sum of powers of two, where each power
    of two can appear in the sum at most max_rpt times.

    Args:
        Optional named:
        num (int): The non-negative integer for which the number of
                ways it can be expressed as the sum of powers of two
                as described above is to be found.
            Default: 10 ** 25
        max_rpt (int): The maximum number of times each power of two
                can appear in any of the sums counted.
            Default: 2
    
    Returns:
    Integer (int) giving the number of ways that num can be expressed
    as the sum of powers of two, where each power of two can appear in
    the sum at most max_rpt times.

    Outline of rationale:
    We use top-down dynamic programming with memoisation going through
    the digits of num when expressed in binary from left to right (i.e.
    from most to least significant), with the parameters idx,
    representing position of the digit in question when read from right
    to left starting at 0 (i.e. the power to which 2 is taken for that
    digit) and higher_req, representing the number of multiples of the
    next higher power of 2 that needs to be accounted for by the current
    and smaller powers of 2.
    The recurrence consists of adding together the results when accounting
    for the quantity passed to the current stage (multiplied by 2 as the
    current power  of 2 is half the size of that from which it was passed
    on), plus one in the case that the current digit is 1 with the
    different possible multiples of the current power of 2 (the multiples
    being in the integer range 0 to max_rpt inclusive) and passing the
    remainder to the smaller powers of 2. The base case is after the units
    digit (from which no value can be passed on but also has no
    corresponding digit meaning it does not add any value to be accounted
    for), so this gives 1 (representing the empty sum) if higher_req is
    0, otherwise 0.
    We exclude some unnecessary calls by noting that the sum of
    all powers of 2 up to but not including a given power of 2
    is exactly one less than that power of 2. Therefore, an upper
    bound on the number of multiples of a higher power that can
    be accounted for by the smaller powers of 2 is:
        floor((max_rpt * (2 ** idx - 1)) / 2 ** idx)
    """
    # Review- Try to implement bottom up dynamic programming solution
    since = time.time()
    digs = []
    while num:
        digs.append(num & 1)
        num >>= 1
    n_dig = len(digs)
    memo = {}
    def recur(idx: int, higher_req: int=0) -> int:
        if idx == -1:
            # This cannot contribute anything, so is only non-zero
            # if higher_req is zero, in which case there is only
            # one option, no powers of 2 (giving an empty sum with
            # value 0).
            return 1 if not higher_req else 0
        
        #if (max_rpt * (exp - 1)) // exp < higher_req: return 0
        args = (idx, higher_req)
        if args in memo.keys(): return memo[args]
        #res = 0
        exp = 1 << (idx)
        higher_req_min = max(0, 2 * higher_req + digs[idx] - max_rpt)
        higher_req_max = min(2 * higher_req + digs[idx], (max_rpt * (exp - 1)) // exp)
        #for i in range(min(2 * higher_req + digs[idx], max_rpt) + 1):
        #    res += recur(idx - 1, higher_req=2 * higher_req + digs[idx] - i)
        res = sum(recur(idx - 1, higher_req=i) for i in range(higher_req_min, higher_req_max + 1))

        """
        if digs[idx]:
            #res = recur(idx - 1, higher_req=True) if higher_req else recur(idx - 1, higher_req=False) + recur(idx - 1, higher_req=True)
            res = recur(idx - 1, higher_req=higher_req * 2) + 
            if not higher_req: res = recur(idx - 1, higher_req=0) + recur(idx - 1, higher_req=2)
            elif higher_req == 1: res = recur(idx - 1, higher_req=0) + recur(idx - 1, higher_req=3)
            #print(f"hi1")
        else:
            #if not higher_req: res = recur(idx - 1, higher_req=0)
            #elif higher_req == 1: res = recur(idx - 1, higher_req=0) + recur(idx - 1, higher_req=2)
            #else: res = recur(idx - 1, higher_req=2)
            #print(f"hi2")
            #res = recur(idx - 1, higher_req=False) + recur(idx - 1, higher_req=True) if higher_req else recur(idx - 1, higher_req=False) + recur(idx - 1, higher_req=True)
        """
        memo[args] = res
        return res
    
    res = recur(n_dig - 1, higher_req=0)
    #print(memo)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res
    """
    print(format(num, "b"))
    dp = [[0, 1]]
    while num:
        if not num & 1:
            curr.append(curr[-1])
        else:
            curr.append(sum(curr))
        #curr = [curr[0], sum(curr)] if num & 1 else [sum(curr), 0]
        num >>= 1
        print(curr)
    return curr[-1]
    """

if __name__ == "__main__":
    to_evaluate = {164}

    if not to_evaluate or 151 in to_evaluate:
        res = singleSheetCountExpectedValueFloat(n_halvings=4)
        print(f"Solution to Project Euler #151 = {res}")

    if not to_evaluate or 152 in to_evaluate:
        res = sumsOfSquareReciprocalsCount(target=(1, 2), denom_min=2, denom_max=80)
        print(f"Solution to Project Euler #152 = {res}")
    
    if not to_evaluate or 153 in to_evaluate:
        res = findRealPartSumOverGaussianIntegerDivisors(n_max=5)
        print(f"Solution to Project Euler #153 = {res}")
    
    if not to_evaluate or 154 in to_evaluate:
        res = multinomialCoefficientMultiplesCount2(n=2 * 10 ** 5, n_k=3, factor_p_factorisation={2: 12, 5: 12})
        print(f"Solution to Project Euler #154 = {res}")

    if not to_evaluate or 155 in to_evaluate:
        res = countDistinctCapacitorCombinationValues(max_n_capacitors=18)
        print(f"Solution to Project Euler #155 = {res}")

    if not to_evaluate or 156 in to_evaluate:
        res = cumulativeNonZeroDigitCountEqualsNumberSum(base=10)
        print(f"Solution to Project Euler #156 = {res}")

    if not to_evaluate or 157 in to_evaluate:
        res = countReciprocalPairSumsMultipleOfReciprocalPower(reciprocal_factorisation={2: 1, 5: 1}, min_power=1, max_power=9)
        print(f"Solution to Project Euler #157 = {res}")

    if not to_evaluate or 158 in to_evaluate:
        res = maximumDifferentLetterStringsWithNSmallerLeftNeighbours(n_chars=26, max_len=26, n_smaller_left_neighbours=1)
        print(f"Solution to Project Euler #158 = {res}")

    if not to_evaluate or 159 in to_evaluate:
        res = maximalDigitalRootFactorisationsSum(n_min=2, n_max=10 ** 6 - 1, base=10)
        print(f"Solution to Project Euler #159 = {res}")
    

    if not to_evaluate or 161 in to_evaluate:
        res = triominoAreaFillCombinations(n_rows=9, n_cols=12)
        print(f"Solution to Project Euler #161 = {res}")

    if not to_evaluate or 162 in to_evaluate:
        res = countHexadecimalIntegersContainGivenDigits(max_n_dig=16, n_contained_digs=3, contained_includes_zero=True)
        print(f"Solution to Project Euler #162 = {res}")

    if not to_evaluate or 164 in to_evaluate:
        res = countIntegersConsecutiveDigitSumCapped(n_digs=20, n_consec=3, consec_sum_cap=9, base=10)
        print(f"Solution to Project Euler #164 = {res}")

    if not to_evaluate or 169 in to_evaluate:
        res = sumOfPowersOfTwo(num=10 ** 25, max_rpt=1)
        print(f"Solution to Project Euler #169 = {res}")