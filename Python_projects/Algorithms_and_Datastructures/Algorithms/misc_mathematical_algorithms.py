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
    """
    For a positive integer n and a prime p, calculates the exponent
    of p in the prime factorisation of n! (n factorial).

    Args:
        n (int): The positive integer whose factorial is being
                assessed for the exponent of the chosen prime in
                the prime factorisation.
        p (int): The prime number whose exponent in n! is being
                calculated. This is assumed to be a prime and is
                not checked, so specifying a non-prime will give
                unexpected behaviour.
    
    Returns:
    Non-negative integer (int) giving the exponent of p in the prime
    factorisation of n! (n factorial).
    """
    res = 0
    while n:
        n //= p
        res += n
    return res

class PrimeModuloCalculator:
    """
    Class for making integer calculations modulo a given prime (i.e.
    calculations as a remainder when divided by p).

    All results modulo p are integers between 0 and (p - 1) inclusive.

    Intended to be extended to accommodate further common calculation
    types.

    Initialization args:
        Required positional:
        p (int): The prime to be used as the modulus for all calculations.
    
    Attributes:
        p (int): The prime to be used as the modulus for all calculations.
    
    Methods:
        (For more detail about a specific method, see that method's
        documentation)

        add(): Finds the sum of two integers modulo p.
        mult(): Finds the product of two integers modulo p.
        pow(): Finds a given integer to a given integer power modulo
                p. For numbers that are not multiples of p, allows
                negative exponents.
        multiplicativeInverse(): Finds the multiplicative inverse of a
                given integer modulo p (i.e. the integer between 1 and
                (p - 1) inclusive that multiplies with the chosen number
                modulo p to give 1). Requires that the chosen integer
                is not a multiple of p.
        factorial(): Finds the factorial of a non-negative integer
                modulo p.
        multiplicativeInverseFactorial(): Finds the multiplicative inverse
                of the factorial of a non-negative integer modulo p.
        binomial(): For an ordered pair of two non-negative integers
                finds the corresponding binomial coefficient modulo
                p.
        multinomial(): For a list of non-negative integers finds the
                multinomial coefficient modulo p.

    Can be used to solve Leetcode #2954
    """
    def __init__(self, p: int):
        self.p = p

    _factorials_div_ppow = [1, 1]
    #inv_factorials = [1, 1]

    def add(self, a: int, b: int) -> int:
        """
        Calculates the sum of the integers a and b modulo the attribute
        p (i.e. (a + b) % self.p).

        Args:
            Required positional:
            a (int): One of the two integers to sum modulo p.
            b (int): The other of the two integers to sum modulo p.
        
        Returns:
        An integer (int) between 0 and (p - 1) inclusive giving the
        sum of a and b modulo the attribute p.
        """
        return (a + b) % self.p

    def mult(self, a: int, b: int) -> int:
        """
        Calculates the product of the integers a and b modulo the attribute
        p (i.e. (a * b) % self.p).

        Args:
            Required positional:
            a (int): One of the two integers to multiply modulo p.
            b (int): The other of the two integers to multiply modulo p.
        
        Returns:
        An integer (int) between 0 and (p - 1) inclusive giving the
        product of a and b modulo the attribute p.
        """
        return (a * b) % self.p
    
    def pow(self, a: int, n: int) -> int:
        """
        Calculates the a to the power of n modulo the attribute p for
        integers a and n (i.e. (a ^ n) % self.p).
        For negative n, calculates the modulo p multiplicative inverse of
        a to the power of the absolute value of n (using Fermat's Little
        Theorem). This case requires that a is not a multiple of the
        attribute p.

        Args:
            Required positional:
            a (int): The integer whose exponent modulo p is to be calculated.
            n (int): The integer giving the exponent a is to be taken to.
        
        Returns:
        An integer (int) between 0 and (p - 1) inclusive giving a to the power
        of n modulo p for non-negative n or the modulo p multiplicative inverse
        of a to the power of (-n) modulo p for negative n.
        """
        if not n: return 1
        elif n > 0:
            return pow(a, n, self.p)
        if not a % self.p:
            raise ValueError("a may not be a multiple of p for negative exponents")
        return pow(a, self.p - n - 1, self.p)

    def multiplicativeInverse(self, a: int) -> int:
        """
        Calculates the modulo p multiplicative inverse of the integer a (where
        p is the attribute p), i.e. the integer b such that (a * b) = 1 mod p.
        a must not be a multiple of the attribute p.

        Args:
            Required positional:
            a (int): The integer whose modulo p multiplicative inverse is to be
                    calculated
        
        Returns:
        The modulo p multiplicative inverse of a, i.e. the unique integer b between
        1 and (p - 1) inclusive for which (a - b) = 1 mod p.
        """
        if not a % self.p: raise ValueError("a cannot be a multiple of the attribute p")
        return pow(a, self.p - 2, self.p)
    
    def _extendFactorialsDivPPow(self, a: int) -> None:
        a0 = len(self._factorials_div_ppow) - 1
        if a <= a0: return
        q0, r0 = divmod(a0, self.p)
        q1, r1 = divmod(a, self.p)
        if q0 == q1:
            for i in range(r0 + 1, r1 + 1):
                self._factorials_div_ppow.append(self.mult(self._factorials_div_ppow[-1], i))
            return self._factorials_div_ppow[a]
        
        for i in range(r0 + 1, self.p):
            self._factorials_div_ppow.append(self.mult(self._factorials_div_ppow[-1], i))
        
        def interPMultExtension(q: int, r_max: int) -> None:
            while not q % self.p:
                q //= self.p
            self._factorials_div_ppow.append(self.mult(self._factorials_div_ppow[-1], q))
            for i in range(1, r_max):
                self._factorials_div_ppow.append(self.mult(self._factorials_div_ppow[-1], i))
            return
        
        for q in range(q0 + 1, q1):
            interPMultExtension(q, self.p)
        interPMultExtension(q1, r1 + 1)
        return
    
    def _factorialDivPPow(self, a: int) -> int:
        self._extendFactorialsDivPPow(a)
        return self._factorials_div_ppow[a]
    
    def factorial(self, a: int) -> int:
        """
        TODO
        """
        if a >= self.p:
            return 0
        return self._factorialDivPPow(a)
    
    def _multiplicativeInverseFactorialDivPPow(self, a: int) -> int:
        return self.multiplicativeInverse(self._factorialDivPPow(a))
    
    def multiplicativeInverseFactorial(self, a: int) -> int:
        """
        TODO
        """
        if not a % self.p:
            raise ValueError("a may not be a multiple of p")
        return self._multiplicativeInverseFactorialDivPPow(a)

    def binomial(self, n: int, k: int) -> int:
        """
        TODO
        """
        if k > n or k < 0:
            return 0
        if n >= self.p:
            p_exp_numer = factorialPrimeFactorExponent(n, self.p)
            p_exp_denom = factorialPrimeFactorExponent(k, self.p) + factorialPrimeFactorExponent(n - k, self.p)
            if p_exp_numer > p_exp_denom:
                return 0
        return self.mult(
            self._factorialDivPPow(n),
            self.mult(self._multiplicativeInverseFactorialDivPPow(k),
            self._multiplicativeInverseFactorialDivPPow(n - k))
        )

    def multinomial(self, k_lst: List[int]) -> int:
        """
        TODO
        """
        n = sum(k_lst)
        if n >= self.p:
            p_exp_numer = factorialPrimeFactorExponent(n, self.p)
            p_exp_denom = sum(factorialPrimeFactorExponent(k, self.p) for k in k_lst)
            if p_exp_numer > p_exp_denom:
                return 0
        res = self._factorialDivPPow(n)
        for k in k_lst:
            res = self.mult(res, self._multiplicativeInverseFactorialDivPPow(k))
        return res

if __name__ == "__main__":
    res = nthRoot(2, 2, eps=1e-5)
    print(f"nthRoot(2, 2, eps=1e-5) = {res}")

    res = nthRoot(589, 5, eps=1e-5)
    print(f"nthRoot(589, 5, eps=1e-5) = {res}")

    res = nthRoot(2, -2, eps=1e-5)
    print(f"nthRoot(2, -2, eps=1e-5) = {res}")