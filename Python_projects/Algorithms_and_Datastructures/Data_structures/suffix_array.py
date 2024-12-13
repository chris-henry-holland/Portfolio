#!/usr/bin/env python

from collections import deque
from typing import Generator, Dict, List, Set, Tuple, Optional, Union

class SuffixArray:
    # Using SA-IS algorithm constructed based on https://zork.net/~st/jottings/sais.html
    """
    Class implementing a suffix array for a given string, including
    methods allowing it to be used to search the string.

    A suffix array for a given string is an array with length one
    greater than that of the string, storing the start indices of
    all the suffixes of the string (including the whole string and
    the empty suffix) in alphabetical order of the suffixes.
    Construction of this array allows for efficient searching of
    the string for matching patterns.

    Initialization args:
        Required positional:
        s (str): The string for which the suffix array is to be
                constructed.
    
    Attributes:
        s (str): The string on which the suffix array is based.
        n (int): The length of the string s.
        suff_arr (list of ints): The suffix array of s, an array
                of length one greater than that of s (i.e. n + 1).
        lcp (list of ints): The longest common prefix array of
                s, an array of length equal to that of suff_arr.
                Assists with the searching of the string. TODO
        lcp_lr (list of ints): The LCP-LR array of s. An array whose
                length is the largest power of 2 not exceeding
                the length of s. This further accelerates
                the string searching process. TODO
    
    Methods:
        (For details of a specific method, see the documentation of
        that method)
        encodeChars(): TODO
        buildFrequencyArrays(): TODO
        buildSuffixArray(): TODO
        buildLongestCommonPrefixArray(): TODO
        checkLCP(): TODO
        buildLCPLR(): TODO
        search(): TODO
    """
    def __init__(self, s: str):
        self.s = s
        self.n = len(s)
    
    @property
    def suff_arr(self):
        res = getattr(self, "_suff_arr", None)
        if res is None:
            res = self.buildSuffixArray()
            self._suff_arr = res
        return res
    
    @property
    def lcp(self):
        res = getattr(self, "_lcp", None)
        if res is None:
            res = self.buildLongestCommonPrefixArray()
            self._lcp = res
        return res
    
    @property
    def lcp_lr(self):
        res = getattr(self, "_lcp_lr", None)
        if res is None:
            res = self.buildLCPLR(self.lcp)
            self._lcp_lr = res
        return res

    def encodeChars(self, s: str) -> Dict[str, int]:
        chars = sorted(set(s))
        res = {}
        for i, l in enumerate(chars, start=1):
            res[l] = i
        return res
    
    def buildFrequencyArrays(self, nums: List[int], nums_mx: int) -> Tuple[List[int]]:
        f_arr = [0] * (nums_mx + 1)
        for num in nums:
            f_arr[num] += 1
        cumu_arr = [0] * (len(f_arr) + 1)
        for i, f in enumerate(f_arr):
            cumu_arr[i + 1] = cumu_arr[i] + f
        return (f_arr, cumu_arr)
    """
    def buildFrequencyArrays(self, nums: List[int]) -> Tuple[Union[List[bool], List[int]]]:
        n = len(nums)
        arr = [True] * (n + 1)
        lms = []
        num1 = 0
        for i in reversed(range(n)):
            num1, num2 = nums[i], num1
            if num1 > num2 or (num1 == num2 and not arr[i + 1]):
                arr[i] = False
                if arr[i + 1]: lms.append(i + 1)
        return (arr, lms[::-1])
    """
    def buildSuffixArray(self) -> List[int]:

        def induceSortLS(nums: List[int], suff_arr: List[int], ls_arr: List[bool], cumu_arr: List[int]) -> None:
            for (curr_inds, iter_obj, incr) in\
                    (([cumu_arr[i] for i in range(len(cumu_arr) - 1)],\
                    range(len(suff_arr)), 1),\
                    ([cumu_arr[i] - 1 for i in range(1, len(cumu_arr))],\
                    reversed(range(len(suff_arr))), -1)):
                b = (incr == 1)
                for i in iter_obj:
                    suff_idx = suff_arr[i] - 1
                    if suff_idx < 0 or ls_arr[suff_idx] is b:
                        continue
                    char_idx = nums[suff_idx]
                    suff_arr[curr_inds[char_idx]] = suff_idx
                    curr_inds[char_idx] += incr
            return

        encoding = self.encodeChars(self.s)
        nums = [encoding[l] for l in self.s]
        nums.append(0)

        def recur(nums: List[int], nums_mx: int) -> List[int]:
            n = len(nums) - 1
            if nums_mx == n:
                res = [n] * (n + 1)
                for idx, num in enumerate(nums):
                    res[num] = idx
                return res
            cumu_arr = self.buildFrequencyArrays(nums, nums_mx)[1]
            ls_arr, lms_inds = self.buildLSArrayAndLMS(nums)

            curr_inds = [cumu_arr[i] - 1 for i in range(1, len(cumu_arr))]
            suff_arr = [-1] * (n + 1)
            for i in lms_inds:
                char_idx = nums[i]
                suff_arr[curr_inds[char_idx]] = i
                curr_inds[char_idx] -= 1
            induceSortLS(nums, suff_arr, ls_arr, cumu_arr)
            lms_dict = {lms_inds[i]: i for i in range(len(lms_inds) - 1)}
            name_dict = {lms_inds[-1]: 1}
            curr_name = 1
            prev_inds = (n, n)
            for suff_arr_idx in range(1, n + 1):
                idx1 = suff_arr[suff_arr_idx]
                if idx1 not in lms_dict.keys(): continue
                j = lms_dict[idx1]
                inds = (idx1, lms_inds[j + 1])
                if inds[1] - inds[0] == prev_inds[1] - prev_inds[0]:
                    for (i1, i2) in zip(range(*prev_inds), range(*inds)):
                        if nums[i1] != nums[i2]:
                            curr_name += 1
                            break
                else:
                    curr_name += 1
                name_dict[idx1] = curr_name#len(name_lsts) - 1
                prev_inds = inds
            lms_summary = [name_dict[x] for x in lms_inds]
            lms_summary.append(0)
            lms_suff_arr = recur(lms_summary, curr_name)
            curr_inds = [cumu_arr[i] - 1 for i in range(1, len(cumu_arr))]
            suff_arr = [-1] * (n + 1)
            for j in reversed(lms_suff_arr[1:]):
                i = lms_inds[j]
                char_idx = nums[i]
                suff_arr[curr_inds[char_idx]] = i
                curr_inds[char_idx] -= 1
            induceSortLS(nums, suff_arr, ls_arr, cumu_arr)
            return suff_arr
        return recur(nums, len(encoding))
    
    def buildLongestCommonPrefixArray(self) -> List[int]:
        # Kasai algorithm
        # Based on https://leetcode.com/problems/number-of-distinct-substrings-in-a-string/solutions/1010936/python-suffix-array-lcp-o-n-logn/
        suff_arr_inv = [0] * (self.n + 1)
        for rnk, idx in enumerate(self.suff_arr):
            suff_arr_inv[idx] = rnk
        length = 0
        res = [0] * (self.n + 1)
        for idx, rnk in enumerate(suff_arr_inv):
            if rnk >= self.n:
                length = 0
                continue
            idx2 = self.suff_arr[rnk + 1]
            while idx + length < self.n and idx2 + length < self.n and self.s[idx + length] == self.s[idx2 + length]:
                length += 1
            res[rnk] = length
            length = max(length - 1, 0)
        return res
    
    def checkLCP(self) -> bool:
        nxt = self.s[self.suff_arr[0]:]
        for i in range(len(self.lcp) - 1):
            idx = self.suff_arr[i + 1]
            curr, nxt = nxt, self.s[idx:]
            mx_len = min(len(curr), len(nxt))
            for j in range(mx_len):
                if curr[j] != nxt[j]: break
            else: j = mx_len
            if j != self.lcp[i]:
                return False
        if self.lcp[-1]: return False
        return True
    
    def buildLCPLR(self, lcp: List[int]) -> List[int]:
        # Based on https://stackoverflow.com/questions/38128092/how-do-we-construct-lcp-lr-array-from-lcp-array

        # This array has length of the largest power of 2 that does not exceed
        # the length of s
        length = 1
        m = self.n + 1 # The length of self.lcp
        pow2 = 0
        while length <= m:
            length <<= 1
            pow2 += 1
        res = [0] * (length >> 1)
        idx0 = length >> 2
        for i in range(m >> 2):
            i2 = i << 2
            res[idx0 + i] = min(lcp[i2: i2 + 3])
        step2 = 4
        idx = idx0
        for _ in range(2, pow2):
            step, step2 = step2, step2 << 1
            for i in range(length - step - 1, -1, -step2):
                idx -= 1
                if i >= m: continue
                res[idx] = min(res[idx << 1], res[(idx << 1) + 1], lcp[i])
        return res
        """
        n = len(lcp)
        length = 1
        pow2 = 0
        while length < n:
            length <<= 1
            pow2 += 1
        res = [0] * length
        hlf_len = length >> 1
        for i in range(len(lcp) >> 1):
            res[i + hlf_len] = lcp[i << 1]
        step2 = 2
        idx = hlf_len
        for _ in range(1, pow2):
            step, step2 = step2, step2 << 1
            for i in range(length - step - 1, -1, -step2):
                idx -= 1
                if i >= len(lcp): continue
                res[idx] = min(res[idx << 1], res[(idx << 1) + 1], lcp[i])
        print(res[:hlf_len])
        return res[:hlf_len]
        """

    def search(self, p: str) -> List[int]:
        lcp = self.lcp
        lcp_lr = self.lcp_lr
        suff_arr = self.suff_arr

        l_idx = 1
        lft, rgt = 0, (len(self.lcp_lr) << 1) - 1
        length = 0

        def maximiseCommonLength(i: int, length: int) -> Tuple[Union[int, bool]]:
            end = min(self.n - i, len(p))
            is_ge_p = False
            for j in range(length, end):
                if self.s[i + j] == p[j]:
                    continue
                elif self.s[i + j] > p[j]:
                    is_ge_p = True
                break
            else:
                j = end
                is_ge_p = (end == len(p))
            return j, is_ge_p
        
        def binarySearchStep(lft: int, rgt: int, mid: int, length: int, length2: int) -> Tuple[Union[int, bool]]:
            if mid > self.n:
                return (lft, mid, length, True)
            length2 = lcp_lr[l_idx]
            last_lft = False
            if length > length2:
                last_lft = True
            elif length == length2:
                length3, last_lft = maximiseCommonLength(suff_arr[mid], length)
                if not last_lft: length = length3
            if last_lft:
                return (lft, mid, length, True)
            length2 = lcp[mid]
            if length != length2:
                length = min(length, length2)
            elif mid < self.n:
                length = maximiseCommonLength(suff_arr[mid + 1], length)[0]
            return (mid + 1, rgt, length, False)

        # Identifying a range of indices of length at most 4 in suffix array in which has the first
        # suffix of s in this array which has p as a prefix must be located if any such suffixes
        # exist
        while lft < rgt - 3 and length < len(p):
            mid = lft + ((rgt - lft) >> 1)
            l_idx <<= 1
            lft, rgt, length, last_lft = binarySearchStep(lft, rgt, mid, length, lcp_lr[l_idx])
            l_idx += (not last_lft)
        
        # Further restricting this range to length 2
        if lft == rgt - 3 and lft < self.n and length < len(p):
            lft, rgt, length, last_lft = binarySearchStep(lft, rgt, lft + 1, length, lcp[lft])
        
        # Finding the first index in the suffix array whose corresponding suffix of s has p as a
        # prefix if such a suffix exists, otherwise returning the empty list as this implies
        # that p is not a substring of s
        if length < len(p):
            if lft + 1 >= len(lcp) or lcp[lft] != length: return []
            i = suff_arr[lft + 1]
            if self.n - i < len(p): return []
            for j in range(length, len(p)):
                if self.s[i + j] != p[j]:
                    return []
            lft += 1
        # Finding all the suffixes in s which have p as a prefix, which by the definition
        # of the suffix array correspond to a contiguous subarray of the suffix array whose
        # first element is at the index identifies in the preceding steps. We use the LCP
        # array to identify when this contiguous subarray ends (which is when the LCP
        # entry is less than the length of p)
        res = []
        for sa_idx in range(lft, self.n + 1):
            res.append(suff_arr[sa_idx])
            if lcp[sa_idx] < len(p):
                break
        return sorted(res)

def strStr(haystack: str, needle: str) -> int:
    """
    
    
    An overkill solution to Leetcode #28 (Find the Index of the First
    Occurrence in a String) to illustrate and test the use of suffix
    array for pattern matching.
    
    Original problem description:
    
    Given two strings needle and haystack, return the index of the
    first occurrence of needle in haystack, or -1 if needle is not part
    of haystack.
    """
    sa = SuffixArray(haystack)
    print(len(haystack), len(sa.suff_arr))
    ind_lst = sa.search(needle)
    return ind_lst[0] if ind_lst else -1

def countDistinct(s: str) -> int:
    """
    
    Solution to (Premium) Leetcode #1698 (Number of Distinct Substrings
    in a String) to illustrate and test a possible use of the LCP
    array
    
    Original problem description:
    
    Given a string s, return the number of distinct substrings of s.

    A substring of a string is obtained by deleting any number of
    characters (possibly zero) from the front of the string and any
    number (possibly zero) from the back of the string. 
    """
    n = len(s)
    sa = SuffixArray(s)
    return ((n * (n + 1)) >> 1) - sum(sa.lcp)

if __name__ == "__main__":
    print(strStr("abcbacba", "ba"))