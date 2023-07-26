<h1>Project Euler</h1>

Contains the solutions to <a href="https://projecteuler.net/" target="_blank">Project Euler</a> problems.

The solutions are written to give a solution to as general a collections of problems (of which the question posed in the specific Project Euler problem is a special case) as possible. In particular, the solutions also work with different inputs than those requested by Project Euler problems (which in general only request computation for a single case), and solutions to problems involving the digits of integers are where possible written so that the base used can be specified by the user, and not just limited to representation in base 10.

The solutions are given by the computation of one of the functions in the Project_Euler_m1_m2.py file (where m1 and m2 are integers and [m1, m2] is the range of the Project Euler problem labels whose solutions the file contains), with the specific function solving each problem outlined below.

The functions representing the solution generally configured so that the default arguments result in the function giving the solution to the corresponding Project Euler problem. In other words, say that func is the function that calculates the solution to Project Euler problem n, where n is an integer. Then the solution to problem n is given by the return value of func(). The exception to this is for problems where the problem is the same just for different inputs (e.g. Project Euler 18 and 67, Maximum Path Sum I & II), for which the second and any other subsequent use of the function requires at least one argument to be explicitly specified.

This directory also contains the .txt files conatining data that is provided by certain Project Euler problems on which the calculation for that problem should be based.

All solutions written using Python 3.6

<h2>Project_Euler_1_50.py</h2>
This contains the functions that calculate solutions to Project Euler problems 1-50 and 67. The function corresponding to each solution (including any required arguments where the function solving a previous problem is used) is as follows:

- <a href="https://projecteuler.net/problem=1" target="_blank">Problem 1</a>: multipleSum()
- Problem 2: multFibonacciSum()
- Problem 3: largestPrimeFactor()
- Problem 4: largestPalindrome()
- Problem 5: smallestMultiple()
- Problem 6: sumSquareDiff()
- Problem 7: findPrime()
- Problem 8: largestProduct()
- Problem 9: specialPythagoreanTriplet()
- Problem 10: sumPrimes()
- Problem 11: largestLineProduct()
- Problem 12: triangleNDiv()
- Problem 13: sumNumbers()
- Problem 14: longestCollatzChain()
- Problem 15: countLatticePaths()
- Problem 16: digitSum()
- Problem 17: numberLetterCount()
- Problem 18: triangleMaxSum()
- Problem 19: countMonthsStartingDoW()
- Problem 20: digitSum(math.factorial(100)) (uses solution to question 16)
- Problem 21: amicableNumbersSum()
- Problem 22: nameListScoreFromFile()
- Problem 23: notExpressibleAsSumOfTwoAbundantNumbersSum()
- Problem 24: nthPermutation()
- Problem 25: firstFibonacciGEn()
- Problem 26: maxBasimalCycleLength()
- Problem 27: maxConsecutiveQuadraticPrimesProduct()
- Problem 28: numSpiralDiagonalsSum()
- Problem 29: distinctPowersNum()
- Problem 30: digitPowSumEqualsSelfSum()
- Problem 31: coinCombinations()
- Problem 32: pandigitalProductsSum()
- Problem 33: digitCancellationsEqualToSelfProdDenom()
- Problem 34: digitFactorialSumEqualsSelfSum()
- Problem 35: circularPrimesCount()
- Problem 36: multiBasePalindromesSum()
- Problem 37: truncatablePrimesSum()
- Problem 38: multiplesConcatenatedPandigitalMax()
- Problem 39: pythagTripleMaxSolsPerim()
- Problem 40: champernowneConstantProduct()
- Problem 41: largestPandigitalPrime()
- Problem 42: countTriangleWordsInTxtDoc()
- Problem 43: pandigitalDivPropsSum()
- Problem 44: kPolygonalMinKPolyDiff()
- Problem 45: triangularPentagonalAndHexagonal()
- Problem 46: goldbachOtherChk()
- Problem 47: smallestnConsecutiveMDistinctPrimeFactorsUnlimited()
- Problem 48: selfExpIntSumLastDigits()
- Problem 49: primePermutArithmeticProgressionConcat()
- Problem 50: primeSumOfMostConsecutivePrimes()

Problems from outside the range 1-50 whose solution uses one of the above functions:
- Problem 67: triangleMaxSum(triangle="p067_triangle.txt") (uses solution to Problem 18)
