#!/usr/bin/env python

from sortedcontainers import SortedList
import random

from typing import List

def uniformRandomDistinctIntegers(n: int, mn: int, mx: int) -> List[int]:
    """
    Time complexity O(n * (log(n)) ** 2)
    """
    sz = mx - mn + 1
    if sz < n:
        raise ValueError(f"Fewer than {n} integers between {mn} and {mx} inclusive")
    elif not n: return []
    elif sz == n: return list(range(mn, mx + 1))
    lst = SortedList()
    
    def countLT(num: int) -> int:
        return num - lst.bisect_left(num)
    
    def insertNum() -> None:
        num0 = random.randrange(0, sz - len(lst))
        lft, rgt = num0, num0 + len(lst)
        while lft < rgt:
            mid = lft - ((lft - rgt) >> 1)
            if countLT(mid) <= num0: lft = mid
            else: rgt = mid - 1
        lst.add(lft)
        return lft
    if 2 * n <= sz:
        for _ in range(n):
            insertNum()
        return [num + mn for num in lst]
    for _ in range(sz - n):
        insertNum()
    j = 0
    res = []
    for num in range(sz):
        if num == lst[j]:
            j += 1
            if j == len(lst): break
            continue
        res.append(num + mn)
    else: num = sz
    for num in range(num + 1, sz):
        res.append(num + mn)
    return res
