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

# Problem 251
def cardanoTripletGeneratorBySum(sum_max: Optional[int]=None) -> Generator[Tuple[int, Tuple[int, int, int]], None, None]:
    ps = PrimeSPFsieve()
    h = []
    for k in itertools.count(1):
        #rt1 = isqrt(8 * k - 3)
        #rt2 = 2 * isqrt(2 * k - 1) + 1
        
        a = 3 * k - 1
        b_lb = integerNthRoot(2 * k ** 2 * (8 * k - 3), 3)#k * (3 + min(rt1, rt2))
        sm_lb = a + b_lb + k ** 2 * (8 * k - 3) // b_lb ** 2
        if sum_max is not None and sm_lb > sum_max: break
        #print(a, lb)
        while h and h[0][0] <= sm_lb:
            yield heapq.heappop(h)
        #b_sq_c = 27 * k ** 2
        
        pf = ps.primeFactorisation(k)
        c0 = 1

        non_sq = (8 * k - 3)
        num = non_sq
        for p in ps.endlessPrimeGenerator():
            p_sq = p * p
            if p_sq > num: break
            num2, r = divmod(num, p_sq)
            while not r:
                num = num2
                pf[p] = pf.get(p, 0) + 1
                num2, r = divmod(num, p_sq)
            num2, r = divmod(num, p)
            if not r:
                num = num2
                c0 *= p
        c0 *= num
        p_lst = sorted(pf.keys())
        n_p = len(p_lst)
        f_lst = [pf[p] for p in p_lst]
        c_mx = sum_max - a - 1 if sum_max is not None else float("inf")
        #print(k, non_sq, pf, c0)

        def recur(idx: int, b_curr: int=1, c_curr: int=c0) -> Generator[Tuple[int, int], None, None]:
            if idx == n_p:
                if a + b_curr + c_curr > sum_max: return
                yield (b_curr, c_curr)
                return
            c = c_curr
            b = b_curr * p_lst[idx] ** f_lst[idx]
            for i in range(f_lst[idx] + 1):
                yield from recur(idx + 1, b_curr=b, c_curr=c)
                c *= p_lst[idx] ** 2
                if c > c_mx: break
                b //= p_lst[idx]
            return

        for b, c in recur(0, b_curr=1, c_curr=c0):
            sm = a + b + c
            heapq.heappush(h, (sm, (a, b, c)))


    while h: yield heapq.heappop(h)
    return


def cardanoTripletCount(sum_max: int=11 * 10 ** 7) -> int:
    """
    Solution to Project Euler #251
    """
    # Review- Try to make faster
    ps = PrimeSPFsieve()
    h = []
    res = 0
    for k in itertools.count(1):
        #rt1 = isqrt(8 * k - 3)
        #rt2 = 2 * isqrt(2 * k - 1) + 1
        a = 3 * k - 1
        b_lb = integerNthRoot(2 * k ** 2 * (8 * k - 3), 3)#k * (3 + min(rt1, rt2))
        sm_lb = a + b_lb + k ** 2 * (8 * k - 3) // b_lb ** 2
        if sum_max is not None and sm_lb > sum_max: break
        
        if not k % 10 ** 4: print(f"a = {a}")
        
        pf = ps.primeFactorisation(k)
        c0 = 1

        non_sq = (8 * k - 3)
        num = non_sq
        for p in ps.endlessPrimeGenerator():
            p_sq = p * p
            if p_sq > num: break
            num2, r = divmod(num, p_sq)
            while not r:
                num = num2
                pf[p] = pf.get(p, 0) + 1
                num2, r = divmod(num, p_sq)
            num2, r = divmod(num, p)
            if not r:
                num = num2
                c0 *= p
        c0 *= num
        p_lst = sorted(pf.keys())
        n_p = len(p_lst)
        f_lst = [pf[p] for p in p_lst]
        c_mx = sum_max - a - 1 if sum_max is not None else float("inf")
        if a + max(k * isqrt((8 * k - 3) // c0) + c0, 1 + (k ** 2 * (8 * k - 3))) <= sum_max:
            ans = 1
            for f in f_lst:
                ans *= (f + 1)
            res += ans
            continue
        #print(k, non_sq, pf, c0)

        def recur(idx: int, b_curr: int=1, c_curr: int=c0) -> int:
            if idx == n_p:
                return a + b_curr + c_curr <= sum_max
            c = c_curr
            b = b_curr * p_lst[idx] ** f_lst[idx]
            res = 0
            for i in range(f_lst[idx] + 1):
                res += recur(idx + 1, b_curr=b, c_curr=c)
                c *= p_lst[idx] ** 2
                if c > c_mx: break
                b //= p_lst[idx]
            return res
        res += recur(0, b_curr=1, c_curr=c0)

    return res

# Problem 253
def constructingLinearPuzzleMaxSegmentCountDistribution0(n_pieces: int) -> List[int]:
    
    
    if n_pieces == 1: return [0, 1]

    def addDistribution(distr: List[int], distr_add: List[int], mn_val: int=0) -> None:
        for _ in range(max(mn_val + 1, len(distr_add)) - len(distr)):
            distr.append(distr[-1])
        for j in range(mn_val, len(distr)):
            #print(j)
            distr[j] += distr_add[min(j, len(distr_add) - 1)]
        while len(distr) > 1 and distr[-1] == distr[-2]:
            distr.pop()
        return

    def distributionProduct(distr1: List[int], distr2: List[int], mn_val: int=0) -> None:
        if mn_val >= len(distr1) + len(distr2) - 2:
            res = [0] * (mn_val + 1)
            res[-1] = distr1[-1] * distr2[-1]
            return res
        res = [0] * (len(distr1) + len(distr2) - 1)
        for i1 in range(len(distr1)):
            if not distr1[i1]: continue
            for i2 in range(max(0, mn_val - i1), len(distr2)):
                res[i1 + i2] += distr1[i1] * distr2[i2]
        return res
    """
    memo = {}
    def recur(size: int, first: bool, last: bool) -> List[int]:
        if size == 1:
            return [0, 1] if not first and not last else [1]
        if first < last: first, last = last, first
        args = (size, first, last)
        if args in memo.keys(): return memo[args]
        res = recur(size - 1, True, last) if first else [0, *recur(size - 1, True, last)]
        addDistribution(res, recur(size - 1, first, True) if last else [0, *recur(size - 1, first, True)])
        mn_val = first + last + 1
        for i in range(1, size - 1):
            distr1 = recur(i, first, True)
            distr2 = recur(size - i - 1, True, last)
            mult = math.comb(size - 1, i)
            print(distr1, distr2)
            print(f"mult = {mult}")
            prod = [x * mult for x in distributionProduct(distr1, distr2, mn_val=mn_val)]
            print(f"prod = {prod}")
            addDistribution(res, prod)
        memo[args] = res
        return res


    """
    memo = {}
    def recur(gaps: Dict[int, int], ends: List[int]) -> List[int]:
        #print(gaps, ends)
        if gaps and min(gaps.keys()) < 0: return []
        if not gaps and not max(ends):
            return [0, 1]
        #elif max(gaps) < 3 and (first or gaps[0] < 2) and (last or gaps[-1] < 2):
        #    m = len(gaps) + first + last - 1
        #    tot = sum(gaps)
        #    res = [0] * (m + 1)
        #    res[m] = math.factorial(tot)
        #    return res
        #gaps, first, last = min((gaps, first, last), (gaps[::-1], last, first))
        gaps2 = tuple(sorted((k, v) for k, v in gaps.items()))
        args = (gaps2, tuple(sorted(ends)))
        if args in memo.keys(): return memo[args]
        n_pieces = (sum(gaps.values()) if gaps else 0) + 1
        res = [0] * (n_pieces + 1)
        
        for j in range(2):
            end0 = ends[j]
            if not end0: continue
            elif end0 == 1:
                ends[j] = 0
                distr = recur(gaps, ends)
                addDistribution(res, distr, mn_val=n_pieces)
                ends[j] = end0
                continue
            for i in range(end0 - 1):
                i2 = end0 - i - 1
                ends[j] = i
                gaps[i2] = gaps.get(i2, 0) + 1
                distr = recur(gaps, ends)
                addDistribution(res, distr, mn_val=n_pieces)
                gaps[i2] -= 1
                if not gaps[i2]: gaps.pop(i2)
            ends[j] = end0 - 1
            distr = recur(gaps, ends)
            addDistribution(res, distr, mn_val=n_pieces)
            ends[j] = end0
        
        for gap in list(gaps.keys()):
            f = gaps[gap]
            gaps[gap] -= 1
            if not gaps[gap]: gaps.pop(gap)
            if gap == 1:
                distr = [f * x for x in recur(gaps, ends)]
                addDistribution(res, distr, mn_val=n_pieces)
                gaps[gap] = gaps.get(gap, 0) + 1
                continue
            gaps[gap - 1] = gaps.get(gap - 1, 0) + 1
            distr = [2 * f * x for x in recur(gaps, ends)]
            addDistribution(res, distr, mn_val=n_pieces)
            gaps[gap - 1] -= 1
            if not gaps[gap - 1]: gaps.pop(gap - 1)
            for i in range(1, (gap >> 1)):
                i2 = gap - i - 1
                gaps[i] = gaps.get(i, 0) + 1
                gaps[i2] = gaps.get(i2, 0) + 1
                distr = [2 * f * x for x in recur(gaps, ends)]
                addDistribution(res, distr, mn_val=n_pieces)
                gaps[i] -= 1
                if not gaps[i]: gaps.pop(i)
                gaps[i2] -= 1
                if not gaps[i2]: gaps.pop(i2)
            if gap & 1:
                i = gap >> 1
                gaps[i] = gaps.get(i, 0) + 2
                distr = [f * x for x in recur(gaps, ends)]
                addDistribution(res, distr, mn_val=n_pieces)
                gaps[i] -= 2
                if not gaps[i]: gaps.pop(i)
            
            gaps[gap] = gaps.get(gap, 0) + 1
        memo[args] = res
        return res
        """
        for i in range(len(gaps)):
            num = gaps[i]
            if num == 1:
                addDistribution(res, recur([*gaps[:i], *gaps[i + 1:]], first or not i, last or (i == len(gaps) - 1)), mn_val=n_pieces)
                continue
            gaps[i] -= 1
            addDistribution(res, recur(gaps, first or not i, last), mn_val=n_pieces)
            addDistribution(res, recur(gaps, first, last or (i == len(gaps) - 1)), mn_val=n_pieces)
            gaps[i] += 1
            if num <= 2: continue
            gaps2 = [*gaps[:i], 1, num - 2, *gaps[i + 1:]]
            for j in range(1, num - 2):
                addDistribution(res, recur(gaps2, first, last), mn_val=n_pieces)
                gaps2[i] += 1
                gaps2[i + 1] -= 1
            addDistribution(res, recur(gaps2, first, last), mn_val=n_pieces)
        
        memo[args] = res
        return res
        """
        """
        for i in range(n_pieces):
            #print(f"i = {i}")
            bm2 = 1 << i
            if bm2 & bm: continue
            n_segs2 = n_segs + 1
            if i > 0 and bm & (1 << (i - 1)):
                n_segs2 -= 1
            if i < n_pieces - 1 and bm & (1 << (i + 1)):
                n_segs2 -= 1
            ans = recur(bm | bm2, n_segs2)
            for _ in range(len(ans) - len(res)):
                res.append(res[-1])
            for j in range(n_segs, len(res)):
                #print(j)
                res[j] += ans[min(j, len(ans) - 1)]
        while len(res) > 1 and res[-1] == res[-2]:
            res.pop()
        memo[args] = res
        return res
        """

    res_cumu = recur({}, [0, n_pieces - 1])
    for i in range(1, n_pieces >> 1):
        distr = recur({}, [i, n_pieces - i - 1])
        addDistribution(res_cumu, distr)
    res_cumu = [x << 1 for x in res_cumu]
    if n_pieces & 1:
        x = n_pieces >> 1
        addDistribution(res_cumu, recur({}, [x, x]))
    #res_cumu = recur([n_pieces], False, False)
    print(f"len(memo) = {len(memo)}")
    #print(f"res_cumu = {res_cumu}")
    #print(memo)
    res = [res_cumu[0]]
    for i in range(1, len(res_cumu)):
        res.append(res_cumu[i] - res_cumu[i - 1])
    return res

def constructingLinearPuzzleMaxSegmentCountDistribution(n_pieces: int) -> List[int]:
    
    # Review- Look into the binary tree solution of Lucy_Hedgehog
    if n_pieces == 1: return [0, 1]

    def addDistribution(distr: List[int], distr_add: List[int], mn_val: int=0) -> None:
        for _ in range(max(mn_val + 1, len(distr_add)) - len(distr)):
            distr.append(distr[-1])
        for j in range(mn_val, len(distr)):
            #print(j)
            distr[j] += distr_add[min(j, len(distr_add) - 1)]
        while len(distr) > 1 and distr[-1] == distr[-2]:
            distr.pop()
        return

    def distributionProduct(distr1: List[int], distr2: List[int], mn_val: int=0) -> None:
        if mn_val >= len(distr1) + len(distr2) - 2:
            res = [0] * (mn_val + 1)
            res[-1] = distr1[-1] * distr2[-1]
            return res
        res = [0] * (len(distr1) + len(distr2) - 1)
        for i1 in range(len(distr1)):
            if not distr1[i1]: continue
            for i2 in range(max(0, mn_val - i1), len(distr2)):
                res[i1 + i2] += distr1[i1] * distr2[i2]
        return res
    
    memo = {}
    def recur(seg_lens: Dict[int, int]) -> List[int]:
        if not seg_lens:
            return [1]
        seg_lens2 = tuple(sorted((k, v) for k, v in seg_lens.items()))
        args = seg_lens2
        if args in memo.keys(): return memo[args]
        n_pieces = (sum(seg_lens.values()) if seg_lens else 0)
        res = [0] * (n_pieces + 1)
        
        for seg_len in list(seg_lens.keys()):
            f = seg_lens[seg_len]
            seg_lens[seg_len] -= 1
            if not seg_lens[seg_len]: seg_lens.pop(seg_len)
            if seg_len == 1:
                distr = [f * x for x in recur(seg_lens)]
                addDistribution(res, distr, mn_val=n_pieces)
                seg_lens[seg_len] = seg_lens.get(seg_len, 0) + 1
                continue
            seg_lens[seg_len - 1] = seg_lens.get(seg_len - 1, 0) + 1
            distr = [2 * f * x for x in recur(seg_lens)]
            addDistribution(res, distr, mn_val=n_pieces)
            seg_lens[seg_len - 1] -= 1
            if not seg_lens[seg_len - 1]: seg_lens.pop(seg_len - 1)
            for i in range(1, (seg_len >> 1)):
                i2 = seg_len - i - 1
                seg_lens[i] = seg_lens.get(i, 0) + 1
                seg_lens[i2] = seg_lens.get(i2, 0) + 1
                distr = [2 * f * x for x in recur(seg_lens)]
                addDistribution(res, distr, mn_val=n_pieces)
                seg_lens[i] -= 1
                if not seg_lens[i]: seg_lens.pop(i)
                seg_lens[i2] -= 1
                if not seg_lens[i2]: seg_lens.pop(i2)
            if seg_len & 1:
                i = seg_len >> 1
                seg_lens[i] = seg_lens.get(i, 0) + 2
                distr = [f * x for x in recur(seg_lens)]
                addDistribution(res, distr, mn_val=n_pieces)
                seg_lens[i] -= 2
                if not seg_lens[i]: seg_lens.pop(i)
            
            seg_lens[seg_len] = seg_lens.get(seg_len, 0) + 1
        memo[args] = res
        return res

    res_cumu = recur({n_pieces: 1})
    #res_cumu = recur([n_pieces], False, False)
    print(f"len(memo) = {len(memo)}")
    #print(f"res_cumu = {res_cumu}")
    #print(memo)
    res = [res_cumu[0]]
    for i in range(1, len(res_cumu)):
        res.append(res_cumu[i] - res_cumu[i - 1])
    return res

def constructingLinearPuzzleMaxSegmentCountMeanFraction(n_pieces: int) -> CustomFraction:
    denom = 0
    numer = 0
    distr = constructingLinearPuzzleMaxSegmentCountDistribution(n_pieces)
    print(distr)
    for i, num in enumerate(distr):
        numer += i * num
        denom += num
    return CustomFraction(numer, denom)

def constructingLinearPuzzleMaxSegmentCountMeanFloat(n_pieces: int=40) -> float:
    """
    Solution to Project Euler #253
    """
    frac = constructingLinearPuzzleMaxSegmentCountMeanFraction(n_pieces)
    print(frac)
    return frac.numerator / frac.denominator

# Problem 260
def stoneGamePlayerTwoWinningConfigurationsGenerator(n_piles: int, pile_size_max: int) -> Generator[Tuple[int], None, None]:
    # Using Sprague-Grundy

    memo = {}
    def winning(state: List[int]) -> bool:
        state = tuple(sorted(state))
        if state[-1] == 0: return False
        args = state
        if args in memo.keys(): return memo[args]
        #res = 0
        #seen = SortedSet()
        res = False
        for bm in range(1, 1 << n_piles):
            idx_lst = []
            mx = float("inf")
            for i in range(n_piles):
                if bm & 1:
                    mx = min(mx, state[i])
                    idx_lst.append(i)
                    if bm == 1: break
                bm >>= 1
            state2 = list(state)
            for sub in range(1, mx + 1):
                for idx in idx_lst:
                    state2[idx] -= 1
                if not winning(state2):
                    res = True
                    break
            else: continue
            break
        memo[args] = res
        return res
    
    curr = []
    def recur(idx: int) -> Generator[Tuple[int], None, None]:
        if idx == n_piles - 1:
            #print(curr)
            mn = curr[-1] if curr else 0
            curr.append(mn)
            for _ in range(mn, pile_size_max + 1):
                if not winning(curr):
                    yield tuple(curr)
                    break
                curr[-1] += 1
            curr.pop()
            return
        mn = curr[-1] if curr else 0
        curr.append(mn)
        for _ in range(mn, pile_size_max + 1):
            yield from recur(idx + 1)
            curr[-1] += 1
        curr.pop()
        return
    
    yield from recur(0)

def stoneGamePlayerTwoWinningConfigurationsSum(n_piles: int=3, pile_size_max: int=1000) -> int:
    res = 0
    cnt = 0
    for state in stoneGamePlayerTwoWinningConfigurationsGenerator(n_piles=n_piles, pile_size_max=pile_size_max):
        print(state)
        cnt += 1
        res += sum(state)
    print(f"number of states where player 2 is winning = {cnt}")
    return res

# Problem 265
def findAllBinaryCircles(n: int) -> List[int]:
    if n == 1: return [1]
    s_len = 1 << n
    Trie = lambda: defaultdict(Trie)
    init_lst = [1] + ([0] * n) + [1]

    full_trie = Trie()
    trie_lst = []
    for i0 in range(len(init_lst)):
        t = full_trie
        t["tot"] = t.get("tot", 0) + 1
        length = min(n, len(init_lst) - i0)
        for j in range(length):
            t = t[init_lst[i0 + j]]
            t["tot"] = t.get("tot", 0) + 1
        if length < n:
            #print(i0, init_lst[i0:i0 + length], t)
            trie_lst.append(t)
    
    """
    for num in init_lst:
        t["tot"] = 1
        t = t[num]
    
    t = full_trie
    t["tot"] = t.get("tot", 0) + n + 1
    for i0 in range(n):
        t2 = t[1]
        t2["tot"] = 1
        trie_lst.append(t2)
        t = t[0]
        t["tot"] = n - i0
    trie_lst = trie_lst[::-1]
    """
    dig_lst = list(init_lst)
    #print(full_trie)
    #print(trie_lst)
    res = []

    def removeSubs(trie: "Trie") -> None:

        def recur2(t: "Trie", num: int, t0: Optional["Trie"]) -> None:
            if t0 is not None and "sub" not in t.keys(): return
            for num2 in range(2):
                if not num2 in t.keys(): continue
                recur2(t[num2], num2, t)
            sub = t.pop("sub", 0)
            if not sub: return
            t["tot"] -= sub
            if t["tot"] or t0 is None: return
            #print(t0, t, num)
            t0.pop(num)
            return
        recur2(trie, -1, None)
        return

    def recur(idx: int, trie_lst: List["Trie"]) -> None:
        if idx == s_len:
            #print("hi")
            n_exp0 = 1
            #print(full_trie)
            for i, t in enumerate(trie_lst):
                n_exp = n_exp0
                for j in range(i + 1):
                    num = dig_lst[j]
                    t = t[num]
                    t["sub"] = t.get("sub", 0) + 1
                    t["tot"] = t.get("tot", 0) + 1
                    if t["tot"] > n_exp: break
                    n_exp >>= 1
                else:
                    n_exp0 <<= 1
                    continue
                break
            else:
                ans = 0
                for d in dig_lst[1:]:
                    ans = (ans << 1) + d
                ans <<= 1
                if dig_lst[0]: ans += 1
                res.append(ans)
                #print(ans, format(ans, "b"), len(trie_lst))
                #print(full_trie)
            for t in trie_lst:
                removeSubs(t)
            return
        
        dig_lst.append(0)
        full_trie["tot"] = full_trie.get("tot", 0) + 1
        for num in range(2):
            n_exp = 1
            to_stop = False
            for t in trie_lst:
                t2 = t[num]
                if t2.get("tot", 0) >= n_exp:
                    to_stop = True
                    break
                n_exp <<= 1
            else:
                t = full_trie[num]
                if t.get("tot", 0) >= n_exp:
                    to_stop = True
            if to_stop: continue
            dig_lst[-1] = num
            trie_lst2 = []
            t2 = trie_lst[0][num]
            t2["tot"] = t2.get("num", 0) + 1
            for t in trie_lst[1:]:
                t2 = t[num]
                t2["tot"] = t2.get("tot", 0) + 1
                trie_lst2.append(t2)
            
            t2 = full_trie[num]
            t2["tot"] = t2.get("tot", 0) + 1
            trie_lst2.append(t2)
            recur(idx + 1, trie_lst2)
            for t2 in trie_lst2:
                t2["tot"] -= 1
            t2 = trie_lst[0][num]
            t2["tot"] -= 1
        full_trie["tot"] -= 1
        dig_lst.pop()
    
    recur(n + 2, trie_lst)

    return res

def allBinaryCirclesSum(n: int=5) -> List[int]:
    """
    Solution to Project Euler #265
    """
    return sum(findAllBinaryCircles(n))

if __name__ == "__main__":
    to_evaluate = {260}
    since0 = time.time()

    if not to_evaluate or 251 in to_evaluate:
        since = time.time()
        res = cardanoTripletCount(sum_max=11 * 10 ** 7)
        print(f"Solution to Project Euler #251 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 253 in to_evaluate:
        since = time.time()
        res = constructingLinearPuzzleMaxSegmentCountMeanFloat(n_pieces=40)
        print(f"Solution to Project Euler #253 = {res}, calculated in {time.time() - since:.4f} seconds")
    
    if not to_evaluate or 260 in to_evaluate:
        since = time.time()
        res = stoneGamePlayerTwoWinningConfigurationsSum(n_piles=3, pile_size_max=100)
        print(f"Solution to Project Euler #260 = {res}, calculated in {time.time() - since:.4f} seconds")

    if not to_evaluate or 265 in to_evaluate:
        since = time.time()
        res = allBinaryCirclesSum(n=5)
        print(f"Solution to Project Euler #265 = {res}, calculated in {time.time() - since:.4f} seconds")

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