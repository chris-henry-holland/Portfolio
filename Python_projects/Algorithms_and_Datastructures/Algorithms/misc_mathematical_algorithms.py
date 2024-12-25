#! /usr/bin/env python
import scipy.special as sp

from typing import List, Union

Real = Union[int, float]

def nthRoot(a: Real, b: int, eps: Real=1e-5) -> Real:
    """
    Finds the non-negative real b:th root of a (a^(1/b)) to a given
    accuracy using the Newton-Raphson method.

    Args:
        Required positional:
        a (real numeric value): The number whose root is sought.
                This number should be real and non-negative.
        b (int): The root of a to be found. Must be non-zero.
                If this is negative, then a must be non-zero.
        
        Optional named:
        eps (small positive real numeric value): The maximum
                permitted error (i.e. the absolute difference
                between the actual value and the returned value
                must be no larger than this)
    
    Returns:
    Real numeric value giving a value within eps of the non-negative
    b:th root of a.

    Examples:
        >>> nthRoot(2, 2, eps=1e-5)
        1.4142156862745097

        This is indeed within 0.00001 of the square root of 2, which
        is 1.4142135623730950 (to 16 decimal places)

        >>> nthRoot(589, 5, eps=1e-5)
        3.5811555709280753

        >>> nthRoot(2, -2, eps=1e-5)
        0.7071078431372548

        Again, this is indeed within 0.00001 of the square root of a
        half, which is 0.7071067811865476 (to 16 decimal places)
    """
    if a < 0:
        raise ValueError("a should be non-negative")
    if not a: return 0
    elif b < 0:
        if not a:
            raise ValueError("If b is negative, a must be non-zero")
        b = -b
        a = 1 / a
    if b == 1:
        raise ValueError("b must be non-zero")
    x2 = float("inf")
    x1 = a
    while abs(x2 - x1) >= eps:
        x2 = x1
        x1 = ((b - 1) * x2 + a / x2 ** (b - 1)) / b
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

def isqrt(n: int) -> int:
    """
    For a non-negative integer n, finds the largest integer m
    such that m ** 2 <= n (or equivalently, the floor of the
    positive square root of n)
    
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
    if n < 0:
        raise ValueError("n must be non-negative")
    return integerNthRoot(n, 2)
    """
    # Newton's method
    x2 = n
    x1 = (n + 1) >> 1
    while x1 < x2:
        x2 = x1
        x1 = (x2 + n // x2) >> 1
    return x2
    """

def factorialPrimeFactorExponent(n: int, p: int) -> int:
    # The exponent of the prime p in the prime
    # factorisation of n!
    # Assumes p is prime
    res = 0
    while n:
        n //= p
        res += n
    return res

class PrimeModuloCalculator:
    # Can be used to solve Leetcode #2954
    def __init__(self, p: int):
        self.p = p

    factorials_div_ppow = [1, 1]
    #inv_factorials = [1, 1]

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.p

    def mult(self, a: int, b: int) -> int:
        return (a * b) % self.p
    
    def pow(self, a: int, n: int) -> int:
        if not n: return 1
        elif n > 0:
            return pow(a, n, self.p)
        if not a % self.p:
            raise ValueError("a may not be a multiple of p for negative exponents")
        return pow(a, self.p - n - 1, self.p)

    def inv(self, a: int) -> int:
        return pow(a, self.p - 2, self.p)
    
    def extendFactorialsDivPPow(self, a: int) -> None:
        a0 = len(self.factorials_div_ppow) - 1
        if a <= a0: return
        q0, r0 = divmod(a0, self.p)
        q1, r1 = divmod(a, self.p)
        if q0 == q1:
            for i in range(r0 + 1, r1 + 1):
                self.factorials_div_ppow.append(self.mult(self.factorials_div_ppow[-1], i))
            return self.factorials_div_ppow[a]
        
        for i in range(r0 + 1, self.p):
            self.factorials_div_ppow.append(self.mult(self.factorials_div_ppow[-1], i))
        
        def interPMultExtension(q: int, r_max: int) -> None:
            while not q % self.p:
                q //= self.p
            self.factorials_div_ppow.append(self.mult(self.factorials_div_ppow[-1], q))
            for i in range(1, r_max):
                self.factorials_div_ppow.append(self.mult(self.factorials_div_ppow[-1], i))
            return
        
        for q in range(q0 + 1, q1):
            interPMultExtension(q, self.p)
        interPMultExtension(q1, r1 + 1)
        return
    
    def factorialDivPPow(self, a: int) -> int:
        self.extendFactorialsDivPPow(a)
        return self.factorials_div_ppow[a]
    
    def factorial(self, a: int) -> int:
        if a >= self.p:
            return 0
        return self.factorialDivPPow(a)
    
    def inverseFactorialDivPPow(self, a: int) -> int:
        return self.inv(self.factorialDivPPow(a))
    
    def inverseFactorial(self, a: int) -> int:
        if not a % self.p:
            raise ValueError("a may not be a multiple of p")
        return self.inverseFactorialDivPPow(a)

    def binomial(self, n: int, k: int) -> int:
        if k > n or k < 0:
            return 0
        if n >= self.p:
            p_exp_numer = factorialPrimeFactorExponent(n, self.p)
            p_exp_denom = factorialPrimeFactorExponent(k, self.p) + factorialPrimeFactorExponent(n - k, self.p)
            if p_exp_numer > p_exp_denom:
                return 0
        return self.mult(self.factorialDivPPow(n), self.mult(self.inverseFactorialDivPPow(k), self.inverseFactorialDivPPow(n - k)))

    def multinomial(self, k_lst: List[int]) -> int:
        n = sum(k_lst)
        if n >= self.p:
            p_exp_numer = factorialPrimeFactorExponent(n, self.p)
            p_exp_denom = sum(factorialPrimeFactorExponent(k, self.p) for k in k_lst)
            if p_exp_numer > p_exp_denom:
                return 0
        res = self.factorialDivPPow(n)
        for k in k_lst:
            res = self.mult(res, self.inverseFactorialDivPPow(k))
        return res

if __name__ == "__main__":
    res = nthRoot(2, 2, eps=1e-5)
    print(f"nthRoot(2, 2, eps=1e-5) = {res}")

    res = nthRoot(589, 5, eps=1e-5)
    print(f"nthRoot(589, 5, eps=1e-5) = {res}")

    res = nthRoot(2, -2, eps=1e-5)
    print(f"nthRoot(2, -2, eps=1e-5) = {res}")