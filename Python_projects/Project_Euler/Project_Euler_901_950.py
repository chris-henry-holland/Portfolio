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

if __name__ == "__main__":
    to_evaluate = {938}
    since0 = time.time()

    if not to_evaluate or 938 in to_evaluate:
        since = time.time()
        res = redBlackCardGameLastCardBlackProbabilityFloat(n_red_init=24690, n_black_init=12345)
        print(f"Solution to Project Euler #938 = {res}, calculated in {time.time() - since:.4f} seconds")


    print(f"Total time taken = {time.time() - since0:.4f} seconds")


"""
n_max = 1000
for n in range(1, n_max + 1):
    res = func(n)
    res2 = func2(n)
    if res != res2:
        print(n, res, res2)
"""
"""
for k in range(1, 101):
    #num = 8 * a ** 3 + 15 * a ** 2 + 6 * a - 1
    #if not num % 27:
    #    print(a, num // 27)
    num = 8 * k - 3
    print(k, 3 * k - 1, num, k ** 2 * num)
"""
"""
def upperBoundDigitSumCoarse(max_dig_count: int, base: int=10) -> Tuple[int, int]:
    num_max = math.factorial(base - 1) * (max_dig_count + 1) - 1
    n_dig = 0
    num2 = num_max
    while num2 >= base:
        num2 //= base
        n_dig += 1
    #n_dig = max(n_dig, mx_non_max_dig_n_dig)
    return num_max, n_dig * (base - 1) + num2

prev = -1
for i in range(1, 10 ** 9):
    n_dig = upperBoundDigitSumCoarse(i, base=10)[1]
    if n_dig > prev:
        print(i, n_dig)
        prev = n_dig
"""