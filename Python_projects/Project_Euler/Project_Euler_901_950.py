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
from pseudorandom_number_generators import blumBlumShubPseudoRandomGenerator

def redBlackCardGameLastCardBlackProbabilityFraction(n_red_init: int, n_black_init: int) -> CustomFraction:

    prev = [0] * (n_red_init + 1)
    row = [0] * (n_red_init + 1)
    row[0] = 1
    row[1] = 0
    for n_red in range(2, n_red_init + 1):
        tot = n_red + 1
        denom = (n_red + 1) * n_red
        #p_2red = CustomFraction(n_red - 1, n_red + 1)
        #p_redblack = CustomFraction(2, n_red + 1)
        row[n_red] = CustomFraction(n_red - 1, n_red + 1) * row[n_red - 2]# + CustomFraction(2, n_red + 1) * prev[n_red]


    for n_black in range(2, n_black_init + 1):
        if not n_black % 100:
            print(f"n_black = {n_black} of {n_black_init}")
        prev = row
        row = [0] * (n_red_init + 1)
        row[0] = 1
        m = 1 - CustomFraction((n_black - 1), n_black + 1)
        k = CustomFraction(2, n_black + 1) * prev[1]
        row[1] = k / m
        #denom = n_black + 1
        for n_red in range(2, n_red_init + 1):
            tot = n_red + n_black
            denom = tot * (tot - 1)
            p_2red = CustomFraction(n_red * (n_red - 1), denom)
            p_2black = CustomFraction(n_black * (n_black - 1), denom)
            p_redblack = CustomFraction(2 * n_red * n_black, denom)
            m = 1 - p_2black
            k = p_2red * row[n_red - 2] + p_redblack * prev[n_red]
            row[n_red] = k / m
    return row[-1]


    """
    memo = {}
    def recur(n_red: int, n_black: int) -> CustomFraction:
        if not n_red: return CustomFraction(1, 1)
        elif not n_black: return CustomFraction(0, 1)
        args = (n_red, n_black)
        if args in memo.keys(): return memo[args]
        tot = n_red + n_black
        denom = tot * (tot - 1)
        p_2red = CustomFraction(n_red * (n_red - 1), denom)
        p_2black = CustomFraction(n_black * (n_black - 1), denom)
        p_redblack = CustomFraction(2 * n_red * n_black, denom)
        #print(f"total probability = {p_2red + p_2black + p_redblack}")
        slf_mult = CustomFraction(1, 1)
        res = CustomFraction(0, 1)
        if p_2red != 0:
            res += p_2red * recur(n_red - 2, n_black)
        if p_2black != 0:
            slf_mult -= p_2black
        if p_redblack != 0:
            res += p_redblack * recur(n_red, n_black - 1)
        res /= slf_mult
        memo[args] = res
        return res
    res = recur(n_red_init, n_black_init)
    #print(memo)
    return res
    """
    
def redBlackCardGameLastCardBlackProbabilityFloat(n_red_init: int=24690, n_black_init: int=12345) -> float:
    """
    Solution to Project Euler #938
    """
    # Look into closed form solution
    prev = [0] * (n_red_init + 1)
    row = [0] * (n_red_init + 1)
    row[0] = 1
    row[1] = 0
    for n_red in range(2, n_red_init + 1):
        tot = n_red + 1
        denom = (n_red + 1) * n_red
        #p_2red = CustomFraction(n_red - 1, n_red + 1)
        #p_redblack = CustomFraction(2, n_red + 1)
        row[n_red] = ((n_red - 1) / (n_red + 1)) * row[n_red - 2]# + CustomFraction(2, n_red + 1) * prev[n_red]


    for n_black in range(2, n_black_init + 1):
        if not n_black % 10:
            print(f"n_black = {n_black} of {n_black_init}")
        prev = row
        row = [0] * (n_red_init + 1)
        row[0] = 1
        m = 1 - ((n_black - 1) / (n_black + 1))
        k = (2 / (n_black + 1)) * prev[1]
        row[1] = k / m
        #denom = n_black + 1
        for n_red in range(2, n_red_init + 1):
            tot = n_red + n_black
            denom = tot * (tot - 1)
            p_2red = (n_red * (n_red - 1) / denom)
            p_2black = (n_black * (n_black - 1) / denom)
            p_redblack = (2 * n_red * n_black / denom)
            m = 1 - p_2black
            k = p_2red * row[n_red - 2] + p_redblack * prev[n_red]
            row[n_red] = k / m
    return row[-1]
    """
    memo = {}
    def recur(n_red: int, n_black: int) -> float:
        if not n_red: return 1
        elif not n_black: return 0
        args = (n_red, n_black)
        if args in memo.keys(): return memo[args]
        tot = n_red + n_black
        denom = tot * (tot - 1)
        p_2red = n_red * (n_red - 1) / denom
        p_2black = n_black * (n_black - 1) / denom
        p_redblack = 2 * n_red * n_black / denom
        #print(f"total probability = {p_2red + p_2black + p_redblack}")
        slf_mult = 1
        res = 0
        if p_2red != 0:
            res += p_2red * recur(n_red - 2, n_black)
        if p_2black != 0:
            slf_mult -= p_2black
        if p_redblack != 0:
            res += p_redblack * recur(n_red, n_black - 1)
        res /= slf_mult
        memo[args] = res
        return res
    res = recur(n_red_init, n_black_init)
    #print(memo)
    return res
    """
    #res = redBlackCardGameLastCardBlackProbabilityFraction(n_red_init, n_black_init)
    #print(res)
    #return res.numerator / res.denominator

def sumOfSubsetElevisorsBruteForce(n_max: int) -> int:
    res = 0
    for bm in range(1, 1 << n_max):
        ss = set()
        for i in reversed(range(1, n_max + 1)):
            if bm & 1:
                for num in range(i << 1, n_max + 1, i):
                    if num in ss:
                        res += i
                        break
                ss.add(i)
            bm >>= 1
    return res

def sumOfSubsetElevisorsBruteForce2(n_max: int) -> int:
    res = 0
    for num in range(1, n_max + 1):
        if not num % 10 ** 5:
            print(f"phase 1, num = {num} of {n_max - 1}")
        n_mults = n_max // num
        res = res + num * (pow(2, n_max - 1) - pow(2, n_max - n_mults))
    return res

def sumOfSubsetElevisors(n_max: int=10 ** 14, md: Optional[int]=1234567891) -> int:
    """
    Solution to Project Euler #944
    """
    rt = isqrt(n_max)
    res = 0
    mx = rt + 1
    if rt * rt == n_max:
        res = res + rt * (pow(2, n_max - 1, mod=md) - pow(2, n_max - rt, mod=md))
        if md is not None: res %= md
        mx -= 1
    for num in range(1, mx):
        if not num % 10 ** 5:
            print(f"phase 1, num = {num} of {mx - 1}")
        n_mults = n_max // num
        res = res + num * (pow(2, n_max - 1, mod=md) - pow(2, n_max - n_mults, mod=md))
        if md is not None: res %= md
    #print(res)
    for num in range(2, mx):
        
        if not num % 10 ** 5:
            print(f"phase 2, num = {num} of {mx - 1}")
        rgt = (n_max) // num
        lft = max((n_max) // (num + 1), rt) + 1
        #print(num, lft, rgt)
        if rgt < lft: break
        mult = (rgt * (rgt + 1) - lft * (lft - 1)) >> 1
        if md is not None: mult %= md
        ans = mult * (pow(2, n_max - 1, mod=md) - pow(2, n_max - num, mod=md))
        res = res + ans
        #print(num, mn, mx, mult, res)
        if md is not None: res %= md
    return res

if __name__ == "__main__":
    to_evaluate = {944}
    since0 = time.time()

    if not to_evaluate or 938 in to_evaluate:
        since = time.time()
        res = redBlackCardGameLastCardBlackProbabilityFloat(n_red_init=24690, n_black_init=12345)
        print(f"Solution to Project Euler #938 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 944 in to_evaluate:
        since = time.time()
        res = sumOfSubsetElevisors(n_max=10 ** 14, md=1234567891)
        print(f"Solution to Project Euler #944 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

"""
for num in range(1, 17):
    ans1 = sumOfSubsetElevisorsBruteForce(num)
    ans2 = sumOfSubsetElevisorsBruteForce2(num)
    ans = sumOfSubsetElevisors(n_max=num, md=None)
    print(f"num = {num}, brute force 1 = {ans1}, brute force 2 = {ans2}, func = {ans}")
"""