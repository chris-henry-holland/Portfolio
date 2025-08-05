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

# Problem 938
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


# Problem 940
def twoDimensionalRecurrenceFibonacciSum(k_min: int=2, k_max: int=50, md: Optional[int]=1123581313) -> int:

    addMod = (lambda a, b: a + b) if md is None else (lambda a, b: (a + b) % md)
    

    # Assumes, if given, that md is prime (uses FLT)
    inv13 = 1 / 13 if md is None else pow(13, md - 2, md)

    fib_arr = [0, 1]
    for _ in range(2, k_max + 1):
        fib_arr.append(fib_arr[-2] + fib_arr[-1])
    print(f"maximum argument = {fib_arr[-1]}")
    
    a1 = (3 + math.sqrt(13)) / 2
    a2 = (3 - math.sqrt(13)) / 2
    b1 = (1 + math.sqrt(13)) / 2
    b2 = (1 - math.sqrt(13)) / 2
    print(f"a1 = {a1}, a2 = {a2}, b1 = {b1}, b2 = {b2}")

    def floatPowerMod(base: float, exp: int, md: int, mult: float=1) -> float:
        if abs(base) <= 1: return (mult * base ** exp) % md
        base_neg = base < 0
        base = abs(base)
        mult_neg = mult < 0
        mult = abs(mult)
        res_neg = mult_neg
        if exp & 1 and base_neg: res_neg = not res_neg
        #print(base, exp, md)
        res = mult % md
        print(f"mult = {mult}, res = {res}")
        p = base
        exp2 = exp
        while exp2:
            if exp2 & 1:
                print(res, p)
                res = (res * p) % md
                #print(res, p)
                if exp2 == 1:
                    break
            p = (p * p) % md
            exp2 >>= 1
        print(base, exp, md, mult, res)
        print(f"res_neg = {res_neg}")
        return (md - res) if res_neg else res
    #print(floatPowerMod(2, 13, md))
    #print(f"inv13 = {inv13}, multiplied by 13 modulo md = {(inv13 * 13) % md}")

    def Apure(m: int, n: int) -> int:
        res = ((a1 - 1) * a1 ** m - (a2 - 1) * a2 ** m) * (b1 ** n - b2 ** n)
        #print(res)
        res += 3 * (a1 ** m - a2 ** m) * (b1 ** (n - 1) - b2 ** (n - 1))
        #print(res)
        res = round(res / 13)
        return res % md

    def inverseMod(num: int, md: int) -> int:
        # Using FLT
        return pow(num, md - 2, md)
    
    divideMod = (lambda a, b: a // b) if md is None else (lambda a, b: (a * inverseMod(b, md)) % md)

    def binomialOddCoefficientsGeneratorMod(n: int, md: int) -> Generator[Tuple[int, int], None, None]:
        if n < 1: return
        num = n % md
        i = -1
        for i in range(1, n - 1, 2):
            yield (i, num)
            num = (num * inverseMod((i + 1) * (i + 2), md)) % md
            num = (num * (n - i) * (n - i - 1)) % md
        yield (i + 2, num)
        return

    #inv3 = inverseMod(3, md)

    def addRoots(rt1: Tuple[int, int], rt2: Tuple[int, int], md: Optional[int]=None) -> Tuple[int, int]:
        res = (rt1[0] + rt2[0], rt1[1] + rt2[1])
        if md is not None:
            res = tuple(x % md for x in res)
        return res

    def subtractRoots(rt1: Tuple[int, int], rt2: Tuple[int, int], md: Optional[int]=None) -> Tuple[int, int]:
        res = (rt1[0] - rt2[0], rt1[1] - rt2[1])
        if md is not None:
            res = tuple(x % md for x in res)
        return res

    def multiplyRoots(rt1: Tuple[int, int], rt2: Tuple[int, int], m: int, md: Optional[int]=None) -> Tuple[int, int]:
        res = (rt1[0] * rt2[0] + m * rt1[1] * rt2[1], rt1[0] * rt2[1] + rt1[1] * rt2[0])
        if md is not None:
            res = tuple(x % md for x in res)
        return res

    def rootPower(rt: Tuple[int, int], m: int, exp: int, md: Optional[int]=None) -> int:
        res = (1, 0)
        p = rt
        exp2 = exp
        while exp2:
            if exp2 & 1:
                res = multiplyRoots(res, p, m, md=md)
                if exp2 == 1: break
            p = multiplyRoots(p, p, m, md=md)
            exp2 >>= 1
        return res

    a_div = 2
    b_div = 2
    a = (3, 1)
    b = (1, 1)
    ab_m = 13
    a1_pows = {}
    a2_pows = {}
    a_div_pows = {}
    b1_pows = {}
    b2_pows = {}
    b_div_pows = {}
    for idx in range(k_min, k_max + 1):
        exp = fib_arr[idx]
        if exp in a1_pows.keys(): continue
        a1_pows[exp] = rootPower(rt=a, m=ab_m, exp=exp, md=md)
        a2_pows[exp] = rootPower(rt=(a[0], -a[1]), m=ab_m, exp=exp, md=md)
        a_div_pows[exp] = pow(a_div, exp, mod=md)
        b1_pows[exp] = rootPower(rt=b, m=ab_m, exp=exp, md=md)
        b2_pows[exp] = rootPower(rt=(b[0], -b[1]), m=ab_m, exp=exp, md=md)
        b_div_pows[exp] = pow(b_div, exp, mod=md)
    
    res = 0
    for idx1 in range(k_min, k_max + 1):
        exp1 = fib_arr[idx1]
        for idx2 in range(k_min, k_max + 1):
            exp2 = fib_arr[idx2]
            t1 = multiplyRoots(a1_pows[exp1], b1_pows[exp2], ab_m, md=md)
            t2 = multiplyRoots(a2_pows[exp1], b2_pows[exp2], ab_m, md=md)
            rt_pow = subtractRoots(t1, t2, md=md)
            ans = rt_pow[1]
            div = b_div_pows[exp1] * b_div_pows[exp2]
            res = addMod(res, divideMod(ans, div))
    return res
    """
    if md is not None:
        m_term1_dict = {}
        m_term2_dict = {}
        n_term1_dict = {}
        n_term2_dict = {}
        inv9 = inverseMod(9, md)
        print("Precalculating m terms")
        for idx in range(k_min, k_max + 1):
            m = fib_arr[idx]
            print(f"m = {m}, Fibonacci number {idx} of {k_max}")
            if m in n_term1_dict.keys(): continue
            m_mult = (inv9 * 13) % md
            
            n_mult = 13 % md
            m_term1 = 0
            inv_mod_pow2_m = inverseMod(pow(2, m, md), md)
            curr = (pow(3, m, md) * inv_mod_pow2_m) % md
            m_term1_dict[m] = 0
            for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(m + 1, md)):
                m_term1_dict[m] = (m_term1_dict[m] + coef * curr) % md
                curr = (curr * m_mult) % md
            curr = (pow(3, m - 1, md) * inv_mod_pow2_m * 2) % md
            m_term2_dict[m] = 0
            for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(m, md)):
                m_term2_dict[m] = (m_term2_dict[m] + coef * curr) % md
                curr = (curr * m_mult) % md
        print("Precalculating n terms")
        for idx in range(k_min - 1, k_max + 1):
            n = fib_arr[idx]
            print(f"n = {n}, Fibonacci number {idx} of {k_max}")
            if n in n_term1_dict.keys(): continue
            inv_mod_pow2_n = inverseMod(pow(2, n - 1, md), md)
            curr = inv_mod_pow2_n
            n_term1_dict[n] = 0
            for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(n, md)):
                n_term1_dict[n] = (n_term1_dict[n] + coef * curr) % md
                curr = (curr * n_mult) % md
            
            curr = (inv_mod_pow2_n * 6) % md
            n_term2_dict[n] = 0
            for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(n - 1, md)):
                n_term2_dict[n] = (n_term2_dict[n] + coef * curr) % md
                curr = (curr * n_mult) % md
            
        #print(f"m_term1_dict = {m_term1_dict}")
        #print(f"m_term2_dict = {m_term2_dict}")
        #print(f"n_term1_lst = {n_term1_dict}")
        #print(f"n_term2_lst = {n_term2_dict}")
    """
    #def Amod(m: int, n: int, md: int) -> int:
        #ans1 = ((m_term1_dict[m] - m_term2_dict[m]) * n_term1_dict[n]) % md
        #ans2 = (m_term2_dict[m] * n_term2_dict[n]) % md
        #return (ans1 + ans2) % md
    """
        inv9 = inverseMod(9, md)
        m_mult = (inv9 * 13) % md
        n_mult = 13
        m_term1 = 0
        curr = pow(3, m, md)
        for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(m + 1, md)):
            m_term1 = (m_term1 + coef * curr) % md
            curr = (curr * m_mult) % md
        m_term2 = 0
        curr = pow(3, m - 1, md)
        for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(m, md)):
            m_term2 = (m_term2 + coef * curr) % md
            curr = (curr * m_mult) % md
        n_term1 = 0
        curr = 1
        for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(n, md)):
            n_term1 = (n_term1 + coef * curr) % md
            curr = (curr * n_mult) % md
        n_term2 = 0
        curr = 1
        for i, (j, coef) in enumerate(binomialOddCoefficientsGeneratorMod(n - 1, md)):
            n_term2 = (n_term2 + coef * curr) % md
            curr = (curr * n_mult) % md
        inv_pow2 = inverseMod(pow(2, m + n - 1, md), md)
        ans1 = ((m_term1 - 2 * m_term2) * n_term1 * inv_pow2) % md
        ans2 = (3 * m_term2 * n_term2 * inv_pow2 * 4) % md
        return (ans1 + ans2) % md
    """
        
    """
        #print(m, n)
        #print(a1, m, md, a1 - 1, floatPowerMod(a1, m, md, mult=a1 - 1))
        #print(a2, m, md, a2 - 1, floatPowerMod(a2, m, md, mult=a2 - 1))
        print(m, n)
        res = (floatPowerMod(a1, m, md, mult=(a1 - 1) / 13) - (((a2 - 1) * a2 ** m) / 13)) % md#floatPowerMod(a2, m, md, mult=(a2 - 1) / 13)) % md
        print(res)
        res = (floatPowerMod(b1, n, md, mult=res) - (res * b2 ** n)) % md# - floatPowerMod(b2, n, md, mult=res)) % md
        print(res)
        res2 = (floatPowerMod(a1, m, md, mult=3 / 13) - ((3 * a2 ** m) / 13)) % md# - floatPowerMod(a2, m, md, mult=3 / 13)) % md
        res2 = (floatPowerMod(b1, (n - 1), md, mult=res2) - (res2 * b2 ** (n - 1))) % md# - floatPowerMod(b2, (n - 1), md, mult=res2)) % md
        print(res2)
        #res = (round((res + res2) % md) * inv13) % md
        res = round((res + res2) % md)
        return res
    """

    #A = Apure if md is None else lambda m, n: Amod(m, n, md=md)
    #A = Apure
    """
    res = 0
    res2 = 0
    for i in range(k_min, k_max + 1):
        print(f"i = {i} of {k_max}")
        f1 = fib_arr[i]
        for j in range(k_min, k_max + 1):
            f2 = fib_arr[j]
            ans = A(f1, f2)
            #print(f"A({f1}, {f2}) = {ans}")
            res += ans
            #ans2 = Apure(f1, f2)
            #print(f"Apure({f1}, {f2}) = {ans2}")
            #res2 = addMod(res2, ans2)
    #print(res, res2)
    return res
    """
    """
    appendMod = (lambda lst, val: lst.append(val)) if md is None else (lambda lst, val: lst.append(val % md))
    addMod = (lambda a, b: a + b) if md is None else (lambda a, b: (a + b) % md)

    fib_arr = [0, 1]
    for i in range(2, k_max + 1):
        appendMod(fib_arr, fib_arr[-2] + fib_arr[-1])
    print(fib_arr[-1])
    res = 0
    n_max = fib_arr[-1]
    row = [0, 1]
    nxt_idx = k_min
    for _ in range(2, n_max + 1):
        appendMod(row, 3 * row[-2] + row[-1])
        if row[-1] == 1 and row[-2] == 0: print("here")
    while nxt_idx < len(fib_arr) and fib_arr[nxt_idx] < 0:
        nxt_idx += 1
    if nxt_idx < len(fib_arr) and fib_arr[nxt_idx] == 0:
        lst = []
        for k_ in range(k_min, k_max + 1):
            res = addMod(res, row[fib_arr[k_]])
            lst.append(row[fib_arr[k_]])
        #print(0, lst)
    print(0, row)
    #print(nxt_idx, row)
    for m in range(1, n_max + 1):
        prev = row
        row = []
        for i in range(n_max):
            appendMod(row, prev[i + 1] + prev[i])
        appendMod(row, 2 * row[-1] + prev[-2])
        while nxt_idx < len(fib_arr) and fib_arr[nxt_idx] < m:
            nxt_idx += 1
        if nxt_idx < len(fib_arr) and fib_arr[nxt_idx] == m:
            lst = []
            for k_ in range(k_min, k_max + 1):
                res = addMod(res, row[fib_arr[k_]])
                lst.append(row[fib_arr[k_]])
            #print(m, lst)
        print(m, row)
    return res
    """

# Problem 944
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

# Problem 945
def xorMultiply(num1: int, num2: int) -> int:
    if num1.bit_count() < num2.bit_count(): num1, num2 = num2, num1
    res = 0
    while num2:
        if num2 & 1:
            res ^= num1
        num2 >>= 1
        num1 <<= 1
    return res

def xorEquationNontrivialPrimitiveSolutionsGenerator(a_b_max: int) -> Generator[Tuple[int, int, int], None, None]:

    sq_lst = [0]

    sq_nxt = 1
    for b in range(1, a_b_max + 1):
        if b >= len(sq_lst):
            sq_lst.append(xorMultiply(sq_nxt, sq_nxt))
            sq_nxt += 1
        b_sq = sq_lst[b]
        for a in range(1 + (b & 1), b + 1, 2):
            #if gcd(a, b) > 1: continue
            a_sq = sq_lst[a]
            num = a_sq ^ xorMultiply(2, xorMultiply(a, b)) ^ b_sq
            while num > sq_lst[-1]:
                sq_lst.append(xorMultiply(sq_nxt, sq_nxt))
                sq_nxt += 1
            c = bisect.bisect_left(sq_lst, num)
            if c < len(sq_lst) and sq_lst[c] == num:
                yield (a, b, c)
    print(sq_lst)
    return

def xorEquationSolutionsCount(a_b_max: int=10 ** 7) -> int:
    def solutionCheck(a, b, c) -> bool:
        return xorMultiply(a, a) ^ xorMultiply(2, xorMultiply(a, b)) ^ xorMultiply(b, b) == xorMultiply(c, c)

    res = a_b_max + 1
    ab_pairs = {}
    for triple in xorEquationNontrivialPrimitiveSolutionsGenerator(a_b_max):
        print(triple)
        a, b = triple[0], triple[1]
        mult = a_b_max // b
        while mult:
            res += 1
            mult >>= 1
        ab_pairs.setdefault(a, [])
        ab_pairs[a].append(b)
        ab_pairs.setdefault(b, [])
        ab_pairs[b].append(a)
        #res += a_b_max // b
        #for mult in range(2, (a_b_max // b) + 1):
        #    triple2 = tuple(x * mult for x in triple)
        #    if not solutionCheck(*triple2):
        #        print(f"Multiple of solution is not a solution: {triple2} = {mult} * {triple}")
    print("Pairs:")
    for odd in sorted(ab_pairs.keys()):
        if not odd & 1: continue
        print(f"{format(odd, 'b')}: {[format(x, 'b') for x in sorted(ab_pairs[odd])]}")
    return res

if __name__ == "__main__":
    to_evaluate = {945}
    since0 = time.time()

    if not to_evaluate or 938 in to_evaluate:
        since = time.time()
        res = redBlackCardGameLastCardBlackProbabilityFloat(n_red_init=24690, n_black_init=12345)
        print(f"Solution to Project Euler #938 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 940 in to_evaluate:
        since = time.time()
        res = twoDimensionalRecurrenceFibonacciSum(k_min=2, k_max=50, md=1123581313)
        print(f"Solution to Project Euler #940 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 944 in to_evaluate:
        since = time.time()
        res = sumOfSubsetElevisors(n_max=10 ** 14, md=1234567891)
        print(f"Solution to Project Euler #944 = {res}, calculated in {time.time() - since:.4f} seconds")

    #if not to_evaluate or 945 in to_evaluate:
    #    since = time.time()
    #    res = xorEquationSolutionsCount(a_b_max=127)
    #    print(f"Solution to Project Euler #945 = {res}, calculated in {time.time() - since:.4f} seconds")

    print(f"Total time taken = {time.time() - since0:.4f} seconds")

"""
for num in range(1, 17):
    ans1 = sumOfSubsetElevisorsBruteForce(num)
    ans2 = sumOfSubsetElevisorsBruteForce2(num)
    ans = sumOfSubsetElevisors(n_max=num, md=None)
    print(f"num = {num}, brute force 1 = {ans1}, brute force 2 = {ans2}, func = {ans}")
"""
#a, b, c = 1, 8, 13
#print(xorMultiply(a, a) ^ xorMultiply(2, xorMultiply(a, b)) ^ xorMultiply(b, b), xorMultiply(c, c))
#for num in range(21):
#    print(num, xorMultiply(num, num))