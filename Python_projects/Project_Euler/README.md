<h1>Project Euler</h1>

Contains the solutions to <a href="https://projecteuler.net/" target="_blank">Project Euler</a> problems.

The solutions are written to give a solution to as general a collections of problems (of which the question posed in the specific Project Euler problem is a special case) as possible. In particular, the solutions also work with different inputs than those requested by Project Euler problems (which in general only request computation for a single case), and solutions to problems involving the digits of integers or concatenation of integers are where possible written so that the base used can be specified by the user, and not just limited to representation in base 10.

The solutions are given by the computation of one of the functions in the Project_Euler_m1_m2.py file (where m1 and m2 are integers and [m1, m2] is the range of the Project Euler problem labels whose solutions the file contains), with the specific function solving each problem outlined below.

The functions representing the solution are generally configured so that the default arguments result in the function giving the solution to the corresponding Project Euler problem. In other words, say that func is the function that calculates the solution to Project Euler problem n, where n is an integer. Then the solution to problem n is given by the return value of func(). The exception to this is for problems where the problem is the same just for different inputs (e.g. Project Euler 18 and 67, Maximum Path Sum I & II), for which the second and any other subsequent use of the function requires at least one argument to be explicitly specified.

Each function and class defined has its own documentation, which can be accessed using the dunder method __doc__() or if looking at the code directly is between the triple double quotation marks at the beginning of the function/class definition. You can refer to these regarding using these functions and classes to solve the same problem with different parameters or a related probem.

This directory also contains the .txt files conatining data that is provided by certain Project Euler problems on which the calculation for that problem should be based.

All solutions written using Python 3.6.

<h2>Project_Euler_1_50.py</h2>
This contains the functions that calculate solutions to Project Euler problems 1-50 and 67. The functions corresponding to each solution (including any required arguments where the function solving a previous problem is used) are as follows:

- <a href="https://projecteuler.net/problem=1" target="_blank">Problem 1</a>: multipleSum()
- <a href="https://projecteuler.net/problem=2" target="_blank">Problem 2</a>: multFibonacciSum()
- <a href="https://projecteuler.net/problem=3" target="_blank">Problem 3</a>: largestPrimeFactor()
- <a href="https://projecteuler.net/problem=4" target="_blank">Problem 4</a>: largestPalindrome()
- <a href="https://projecteuler.net/problem=5" target="_blank">Problem 5</a>: smallestMultiple()
- <a href="https://projecteuler.net/problem=6" target="_blank">Problem 6</a>: sumSquareDiff()
- <a href="https://projecteuler.net/problem=7" target="_blank">Problem 7</a>: findPrime()
- <a href="https://projecteuler.net/problem=8" target="_blank">Problem 8</a>: largestProduct()
- <a href="https://projecteuler.net/problem=9" target="_blank">Problem 9</a>: specialPythagoreanTriplet()
- <a href="https://projecteuler.net/problem=10" target="_blank">Problem 10</a>: sumPrimes()
- <a href="https://projecteuler.net/problem=11" target="_blank">Problem 11</a>: largestLineProduct()
- <a href="https://projecteuler.net/problem=12" target="_blank">Problem 12</a>: triangleNDiv()
- <a href="https://projecteuler.net/problem=13" target="_blank">Problem 13</a>: sumNumbers()
- <a href="https://projecteuler.net/problem=14" target="_blank">Problem 14</a>: longestCollatzChain()
- <a href="https://projecteuler.net/problem=15" target="_blank">Problem 15</a>: countLatticePaths()
- <a href="https://projecteuler.net/problem=16" target="_blank">Problem 16</a>: digitSum()
- <a href="https://projecteuler.net/problem=17" target="_blank">Problem 17</a>: numberLetterCount()
- <a href="https://projecteuler.net/problem=18" target="_blank">Problem 18</a>: triangleMaxSum()
- <a href="https://projecteuler.net/problem=19" target="_blank">Problem 19</a>: countMonthsStartingDoW()
- <a href="https://projecteuler.net/problem=20" target="_blank">Problem 20</a>: digitSum(math.factorial(100)) (uses solution to question 16)
- <a href="https://projecteuler.net/problem=21" target="_blank">Problem 21</a>: amicableNumbersSum()
- <a href="https://projecteuler.net/problem=22" target="_blank">Problem 22</a>: nameListScoreFromFile()
- <a href="https://projecteuler.net/problem=23" target="_blank">Problem 23</a>: notExpressibleAsSumOfTwoAbundantNumbersSum()
- <a href="https://projecteuler.net/problem=24" target="_blank">Problem 24</a>: nthPermutation()
- <a href="https://projecteuler.net/problem=25" target="_blank">Problem 25</a>: firstFibonacciGEn()
- <a href="https://projecteuler.net/problem=26" target="_blank">Problem 26</a>: maxBasimalCycleLength()
- <a href="https://projecteuler.net/problem=27" target="_blank">Problem 27</a>: maxConsecutiveQuadraticPrimesProduct()
- <a href="https://projecteuler.net/problem=28" target="_blank">Problem 28</a>: numSpiralDiagonalsSum()
- <a href="https://projecteuler.net/problem=29" target="_blank">Problem 29</a>: distinctPowersNum()
- <a href="https://projecteuler.net/problem=30" target="_blank">Problem 30</a>: digitPowSumEqualsSelfSum()
- <a href="https://projecteuler.net/problem=31" target="_blank">Problem 31</a>: coinCombinations()
- <a href="https://projecteuler.net/problem=32" target="_blank">Problem 32</a>: pandigitalProductsSum()
- <a href="https://projecteuler.net/problem=33" target="_blank">Problem 33</a>: digitCancellationsEqualToSelfProdDenom()
- <a href="https://projecteuler.net/problem=34" target="_blank">Problem 34</a>: digitFactorialSumEqualsSelfSum()
- <a href="https://projecteuler.net/problem=35" target="_blank">Problem 35</a>: circularPrimesCount()
- <a href="https://projecteuler.net/problem=36" target="_blank">Problem 36</a>: multiBasePalindromesSum()
- <a href="https://projecteuler.net/problem=37" target="_blank">Problem 37</a>: truncatablePrimesSum()
- <a href="https://projecteuler.net/problem=38" target="_blank">Problem 38</a>: multiplesConcatenatedPandigitalMax()
- <a href="https://projecteuler.net/problem=39" target="_blank">Problem 39</a>: pythagTripleMaxSolsPerim()
- <a href="https://projecteuler.net/problem=40" target="_blank">Problem 40</a>: champernowneConstantProduct()
- <a href="https://projecteuler.net/problem=41" target="_blank">Problem 41</a>: largestPandigitalPrime()
- <a href="https://projecteuler.net/problem=42" target="_blank">Problem 42</a>: countTriangleWordsInTxtDoc()
- <a href="https://projecteuler.net/problem=43" target="_blank">Problem 43</a>: pandigitalDivPropsSum()
- <a href="https://projecteuler.net/problem=44" target="_blank">Problem 44</a>: kPolygonalMinKPolyDiff()
- <a href="https://projecteuler.net/problem=45" target="_blank">Problem 45</a>: triangularPentagonalAndHexagonal()
- <a href="https://projecteuler.net/problem=46" target="_blank">Problem 46</a>: goldbachOtherChk()
- <a href="https://projecteuler.net/problem=47" target="_blank">Problem 47</a>: smallestnConsecutiveMDistinctPrimeFactorsUnlimited()
- <a href="https://projecteuler.net/problem=48" target="_blank">Problem 48</a>: selfExpIntSumLastDigits()
- <a href="https://projecteuler.net/problem=49" target="_blank">Problem 49</a>: primePermutArithmeticProgressionConcat()
- <a href="https://projecteuler.net/problem=50" target="_blank">Problem 50</a>: primeSumOfMostConsecutivePrimes()

Problems from outside the range 1-50 whose solution uses one of the above functions:
- <a href="https://projecteuler.net/problem=67" target="_blank">Problem 67</a>: triangleMaxSum(triangle="p067_triangle.txt") (uses solution to Problem 18)
