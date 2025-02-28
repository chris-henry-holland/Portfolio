#!/usr/bin/env python

import bisect
import gmpy2
import heapq
import itertools
import math
import os
from PIL import Image
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

def integerNthRoot(m: int, n: int) -> int:
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

# Problem root 13
def rootExpansionDigits(num: int, n_digs: int, base: int=10) -> List[int]:

    num *= base ** (2 * n_digs)
    num_sqrt = isqrt(num)
    res = []
    for _ in range(n_digs):
        num_sqrt, r = divmod(num_sqrt, base)
        res.append(r)
    res = res[::-1]
    #print(res)
    return res

def rootExpansionDigitSum(num: int=13, n_digs: int=1_000, base: int=10) -> int:
    """
    Solution to Project Euler #root 13
    """
    since = time.time()
    res = sum(rootExpansionDigits(num, n_digs, base=base))
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res


# Problem Heegner
def closestCosPiSqrtInteger(abs_n_max: int=1000) -> int:
    """
    Solution to Project Euler Bonus Problem Heegner

    For integers n with absolute value no greater than
    abs_n_max, finds the value of n such that the value
    of:
        x = cos(pi * sqrt(n))
    is closest to an integer, i.e. the value of n for
    which (abs(x - round(x))) is smallest.

    Args:
        Optional named:
        abs_n_max (int): The largest absolute value of the
                integers considered.
            Default: 1000
    
    Returns:
    Integer (int) giving the value of the integer n with
    absolute value no greater than abs_n_max such that
    cos(pi * sqrt(n)) is closest to an integer.

    Outline of rationale:
    Given that:
        cos(pi * sqrt(n)) = cosh(i * pi * sqrt(n))
    for negative n, we get:
        cos(pi * sqrt(n)) = cosh(i * pi * sqrt(abs(n)))
    This becomes very large very quickly as n becomes
    more negative, and with normal float operations the
    fractional part will rapidly not be calculated accurately
    (if at all).
    We therefore use the gmpy2 package to perform the
    calculations with arbitrary precision, and with each
    caluclation we ensure that the current precision is
    sufficient to get sufficient precision in the fractional
    part to compare between answers, increasing the
    precision if not.

    Note that, given that for x >> 1:
        cosh(x) approximately equals exp(x)
    it is to be expected that for relatively small values of
    abs_n_max the solution is the negative of a Heegner number,
    specifically the largest Heegner number no greater than
    n_max, as a property of these numbers m is that
    exp(pi * sqrt(m)) is extremely close to an integer, with
    the larger the Heeneger number the closer to an integer.
    The Heegner numbers are:
        1, 2, 3, 7, 11, 19, 43, 67, and 163
    """
    since = time.time()
    squares = set(i ** 2 for i in range(math.isqrt(abs_n_max) + 1))
    res = (float("inf"), -1)
    for func, filter, mult in ((gmpy2.cos, lambda x: x not in squares, 1), (gmpy2.cosh, lambda x: True, -1)):
        for i in range(1, abs_n_max + 1):
            if not filter(i): continue
            while True:
                val = func(gmpy2.const_pi() * gmpy2.sqrt(gmpy2.mpc(i))).real
                val_str = str(val)
                increase_precision = False
                if "." not in set(val_str[-10:]):
                    zero_run = 0
                    nine_run = 0
                    for j in range(len(val_str)):
                        d = val_str[~j]
                        if d == ".":
                            increase_precision = True
                            break
                        if d == "0": zero_run += 1
                        else: zero_run = 0
                        if d == "9": nine_run += 1
                        else: nine_run = 0
                        #print(j, zero_run, nine_run, j - max(zero_run, nine_run))
                        if j - max(zero_run, nine_run) >= 10: break
                    if not increase_precision: break
                #print(val)
                gmpy2.get_context().precision += 10
            #print(i, val)
            v2 = abs(val - round(val))
            if v2 < res[0]:
                res = (v2, -i)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res[1]


# Problem 18i
def polynomialPrimeProductRemainder(p_min: int=10 ** 9, p_max: int=11 * 10 ** 8) -> int:
    since = time.time()
    ps = PrimeSPFsieve(p_max)
    print("found prime sieve")
    poly_func = lambda x: x ** 3 - 3 * x + 4
    res = 0
    p_i_mn = bisect.bisect_left(ps.p_lst, p_min)
    seen_primes = set()
    largest_poly_arg = -1
    poly_past_range = False
    for p_i in range(p_i_mn, len(ps.p_lst)):
        p = ps.p_lst[p_i]
        if not poly_past_range:
            for largest_poly_arg in range(largest_poly_arg + 1, p):
                val = poly_func(largest_poly_arg)
                #print(f"arg = {largest_poly_arg}, val = {val}")
                if val > p_max:
                    poly_past_range = True
                    break
                p_facts = ps.primeFactors(val)
                seen_primes |= set(p_facts)
                #if ps.isPrime(val):
                #    seen_primes.add(val)
        if p in seen_primes: continue
        ans = 1
        for i in range(p):
            ans = (ans * poly_func(i)) % p
            if not ans: break
        #print(f"p = {p}, product = {ans}")
        res += ans
    #print(seen_primes)
    print(f"Time taken = {time.time() - since:.4f} seconds")
    return res

# Problem Secret
def loadBlackAndWhitePNGImage(filename: str, relative_to_program_file_directory: bool=False) -> List[Tuple[int]]:
    """
        Optional named:
        relative_to_program_file_directory (bool): If True then
                if doc is specified as a relative path, that
                path is relative to the directory containing
                the program file, otherwise relative to the
                current working directory.
            Default: False
    """
    if relative_to_program_file_directory and not filename.startswith("/"):
        filename = os.path.join(os.path.dirname(__file__), filename)
    image_np = np.array(Image.open(filename))
    #shape = image_np.shape[:2]
    res = image_np[:, :, 0]

    print(res)
    print(res.shape)
    return res

#def 

if __name__ == "__main__":
    to_evaluate = {"secret"}

    if not to_evaluate or "root_13" in to_evaluate:
        res = rootExpansionDigitSum(num=13, n_digs=1_000, base=10)
        print(f"Solution to Project Euler #root 13 = {res}")

    if not to_evaluate or "heegner" in to_evaluate:
        res = closestCosPiSqrtInteger(abs_n_max=1000)
        print(f"Solution to Project Euler #heegner = {res}")
    
    #if not to_evaluate or "18i" in to_evaluate:
    #    res = polynomialPrimeProductRemainder(p_min=100_000, p_max=110_000)
    #    print(f"Solution to Project Euler #18i = {res}")

    #if not to_evaluate or "secret" in to_evaluate:
    #    res = loadBlackAndWhitePNGImage(filename="bonus_secret_statement.png", relative_to_program_file_directory=True)
    #    print(f"Solution to Project Euler #secret = {res}")
