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
from misc_mathematical_algorithms import CustomFraction, gcd, lcm
from prime_sieves import PrimeSPFsieve, SimplePrimeSieve

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


if __name__ == "__main__":
    to_evaluate = {204}
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

    print(f"Total time taken = {time.time() - since0:.4f} seconds")