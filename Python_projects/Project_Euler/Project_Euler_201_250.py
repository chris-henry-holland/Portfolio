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

if __name__ == "__main__":
    to_evaluate = {201}
    since0 = time.time()

    if not to_evaluate or 201 in to_evaluate:
        since = time.time()
        res = subsetsOfSquaresWithUniqueSumTotal(n_max=100, k=50)
        #res = subsetsWithUniqueSumTotal({1, 3, 6, 8, 10, 11}, 3)
        print(f"Solution to Project Euler #151 = {res}, calculated in {time.time() - since:.4f} seconds")


    print(f"Total time taken = {time.time() - since0:.4f} seconds")