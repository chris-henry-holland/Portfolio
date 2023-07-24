#!/usr/bin/env python

from collections import deque
from collections.abc import Iterable
import itertools
from typing import Dict, Generator, List, Tuple, Optional, Union, Callable

def LPSArray(s: str) -> List[int]:
    """
    Constructs the Longest Prefix Suffix (LPS) array for a string s,
    using the Knuth-Morris-Pratt (KMP) algorithm. This is applied
    to the pattern string in the first step of the KMP
    pattern searching algorithm (see the KMPMatchGenerator() function
    below).
    For a string s of length n, the LPS array is a 1D integer array of
    length n, where the integer at a given index represents
    the longest non-prefix substring of s (i.e. a substring of s that
    does not begin at the start of s) ending at the corresponding
    index that matches a prefix of s. Alternatively, as the name
    suggests, the ith index represents the length of the longest
    proper prefix (i.e. a prefix that is not the whole string)
    of the string s[:i + 1] (i.e. the substring of s consisting of the
    first i + 1 characters of s) that is also a (proper) suffix of s[:i + 1].
    
    Args:
        Required positional:
        s (str): the string for which the LPS array is to
                be generated.
    
    Returns:
        List of integers (int) representing the LPS array
        of the input string s
       
    Example:
        >>> LPSArray("abacabcabacad")
        [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5, 0]
        
        This signifies for instance that the longest non-prefix
        substring ending at index 11 that matches a prefix of s
        is length 5 (namely the substring "abaca").
    """
    n = len(s)
    res = [0] * n
    j = 0
    for i in range(1, n):
        l = s[i]
        while j > 0 and l != s[j]:
            j = res[j - 1]
        if l != s[j]: continue
        j += 1
        res[i] = j
    return res

def KMPMatchGenerator(s: str, p: str, lps_arr: Optional[List[int]]=None)\
        -> Generator[int, None, None]:
    """
    Generator that iterates over each and every index in string s at
    which a substring matching the pattern string p starts, using the
    Knuth-Morris-Pratt (KMP) Algorithm
    
    Args:
        Required positional:
            s (str): The string in which the occurences of the
                    pattern string are to be located.
            p (str): The pattern string.
    
    Returns:
        Generator yielding integers (int) giving the indices of s at
        which every substring of s matching the pattern string p start,
        in increasing order.
    
    Example:
        >>> for i in KMPMatchGenerator("casbababbbbbabbceab", "bb"): print(i)
        7
        8
        9
        10
        13
        
        This signifies that a substrings of this string exactly
        matching "bb" begin at preciesly the indices 7, 8, 9, 10 and 13
        and nowhere else in the string.
    """
    if lps_arr is None: lps_arr = LPSArray(p)
    m = len(p)
    j = 0
    for i, l in enumerate(s):
        while j > 0 and (j == m or l != p[j]):
            j = lps_arr[j - 1]
        if l != p[j]: continue
        j += 1
        if j == m:
            yield i - m + 1
    return

class KMP:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.lps = self.generateLPS()
    
    def generateLPS(self):
        return LPSArray(self.pattern)
    
    def matchStartGenerator(self, s: str):
        yield from KMPMatchGenerator(s, self.pattern, lps_arr=self.lps)

def ZArray(s: str) -> List[int]:
    """
    Constructs the Z array for a string s using the Z Algorithm.
    For a string s of length n, the Z array is a 1D integer array of
    length n, where the integer at a given index represents
    the longest substring of s starting at the corresponding
    index that matches a prefix of s.
    
    Args:
        Required positional:
        s (str): the string for which the Z array is to
                be generated.
    
    Returns:
        List of integers (int) representing the Z array
        of the input string s
       
    Example:
        >>> ZArray("abacabcabacad")
        [12, 0, 1, 0, 2, 0, 0, 5, 0, 1, 0, 1, 0]
        
        This signifies for instance that the longest substring
        starting at index 7 that matches a prefix of s
        is length 5 (namely the substring "abaca"- note that
        the next character of the substring starting at index
        7 would be "d", which does not match the next character
        for any longer prefix, which would be "b").
    """
    n = len(s)
    res = [0] * n
    lt, rt = 0, 1
    for i in range(1, n):
        if res[i - lt] < rt - i:
            res[i] = res[i - lt]
            continue
        lt = i
        for rt in range(max(rt, i), n):
            if s[rt] != s[rt - lt]: break
        else: rt = n
        res[i] = rt - lt
    res[0] = n
    return res

def ZMatchGenerator(s: str, p: str, wild_char: str="$") -> List[int]:
    """
    Generator that iterates over each and every index in string s at
    which a substring matching the pattern string p starts, using the
    Z Algorithm
    
    Args:
        Required positional:
            s (str): The string in which the occurences of the
                    pattern string are to be located.
            p (str): The pattern string.
        
        Optional named:
            wild_char (str): A string character which does not appear
                    in the string s (used as a separator placed between
                    p and s when they are combined in a single string
                    during the implementation of the Z algorithm to
                    ensure that the start of s cannot be inappropriately
                    treated as being part of the pattern.
                Default: "$"
    
    Returns:
        Generator yielding integers (int) giving the indices of s at
        which every substring of s matching the pattern string p start,
        in increasing order.
    
    Example:
        >>> for i in ZMatchGenerator("casbababbbbbabbceab", "bb"): print(i)
        7
        8
        9
        10
        13
        
        This signifies that a substrings of this string exactly
        matching "bb" begin at preciesly the indices 7, 8, 9, 10 and 13
        and nowhere else in the string.
    """
    s2 = "".join([p, wild_char, s])
    m, n, n2 = len(p), len(s), len(s2)
    z_arr = ZArray(s2)
    pref_len = n2 - n
    res = []
    for i in range(pref_len, n):
        if z_arr[i] == m:
            yield i - pref_len
    return


def rollingHash(s: Iterable, length: int, p_lst: Union[Tuple[int], List[int]]=(37, 53),
        md: int=10 ** 9 + 7, func: Optional[Callable]=None) -> Generator[int, None, None]:
    """
    Generator that yields the rolling hash values of each contiguous subset of
    the iterable s with length elements in order of their first element. The hash
    is polynomial-based around prime numbers as specified in p_lst modulo md.
    The elements of s are passed through the function func which transforms
    each possible input value into a distinct integer (by default, the identity
    if the elements of s are integers, and the ord() function if they are
    string characters).
    
    Args:
        Required positional:
            
    """
    if hasattr(s, "__len__") and len(s) < length:
        return
    if func is None:
        try: val = func(next(iter_obj))
        except StopIteration: return
        if isinstance(next(iter(s), str)):
            func = lambda x: ord(x)
        else: func = lambda x: x
    iter_obj = iter(s)
    n_p = len(p_lst)
    hsh = [0] * n_p
    for i in range(length):
        try: val = func(next(iter_obj))
        except StopIteration: return
        for j, p in enumerate(p_lst):
            hsh[j] = (hsh[j] * p + lst[i]) % md
    yield tuple(hsh)
    mults[j] = [pow(p, length, md) for p in p_lst]
    for i in itertools.count(length):
        try: val = func(next(iter_obj))
        except StopIteration: return
        for j, p in enumerate(p_lst):
            hsh[j] = ((hsh[j] - mults[j] * lst[i - length]) * p + lst[i]) % md
        yield tuple(hsh)
    return

def rollingHashSearch(s: str, patterns: List[str],
        p_lst: Optional[Union[List[int], Tuple[int]]]=(31, 37),
        md: int=10 ** 9 + 7) -> Dict[str, List[int]]:
    
    ord_A = ord("A")
    def char2num(l: str) -> int:
        return ord(l) - ord_A
    
    pattern_dict = {}
    for pattern in patterns:
        length = len(pattern)
        pattern_dict.setdefault(l, {})
        hsh = next(rollingHash(pattern, length, p_lst=p_lst, md=md, func=char2num))
        pattern_dict[length].setdefault(hsh, set())
        pattern_dict[length][hsh].add(pattern)
    
    res = {}
    for length, hsh_dict in pattern_dict.items():
        for i, hsh in enumerate(rollingHash(s, length, p_lst=p_lst, md=md, func=char2num)):
            if hsh not in hsh_dict.keys(): continue
            pattern = s[i: i + length]
            if pattern not in hsh_dict[hsh]: continue
            res.setdefault(pattern, [])
            res[pattern].append(i)
    return res

class AhoCorasick:
    """
    Data structure used for simultaneous matching of multiple
    patterns in a text, with time complexity O(n + m + z) where
    n is the length of the string being searched, m is the sum
    of the lengths of the patterns and z is the total number of
    matches over all of the patterns in the string.
    """

    def __init__(self, words: List[str]):
        self.goto = [{}]
        self.failure = [-1]
        self.out = [0]
        self.out_lens = [0]
        self.words = words
        self.buildAutomaton()

    def buildAutomaton(self) -> None:
        for i, w in enumerate(self.words):
            j = 0
            for l in w:
                if l not in self.goto[j].keys():
                    self.goto[j][l] = len(self.goto)
                    self.goto.append({})
                    self.failure.append(0)
                    self.out.append(0)
                    self.out_lens.append(0)
                j = self.goto[j][l]
            self.out[j] |= 1 << i
            self.out_lens[j] |= 1 << len(w)
        
        queue = deque(self.goto[0].values())
        
        while queue:
            j = queue.popleft()
            for l, j2 in self.goto[j].items():
                j_f = self.failure[j]
                while j_f and l not in self.goto[j_f].keys():
                    j_f = self.failure[j_f]
                j_f = self.goto[j_f].get(l, 0)
                self.failure[j2] = j_f
                self.out[j2] |= self.out[j_f]
                self.out_lens[j2] |= self.out_lens[j_f]
                queue.append(j2)
        return
    
    def _findNext(self, j: int, l: str) -> int:
        while j and l not in self.goto[j].keys():
            j = self.failure[j]
        return self.goto[j].get(l, 0)
    
    def search(self, s: str) -> Dict[str, List[int]]:
        """
        Gives dictionary for the starting index of each occurrence
        of each of self.words in the string s.
        """
        j = 0
        res = {}
        for i, l in enumerate(s):
            j = self._findNext(j, l)
            bm = self.out[j]
            for idx, w in enumerate(self.words):
                if not bm: break
                if bm & 1:
                    res.setdefault(w, [])
                    res[w].append(i - len(w) + 1)
                bm >>= 1
        return res
    
    def searchEndIndices(self, s: str) -> Generator[Tuple[int, List[int]], None, None]:
        """
        Generator yielding a 2-tuple of each index of s (in ascending order)
        and a list of the corresponding indies of the patterns in self.words
        that have a match in s that ends exactly at that index of s.
        """
        j = 0
        for i, l in enumerate(s):
            j = self._findNext(j, l)
            bm = self.out[j]
            idx = 0
            res = []
            while bm:
                if bm & 1: res.append(idx)
                idx += 1
                bm >>= 1
            yield (i, res)
        return

    def searchLengths(self, s: str) -> Generator[Tuple[int], None, None]:
        j = 0
        for i, l in enumerate(s):
            j = self._findNext(j, l)
            bm = self.out_lens[j]
            length = 0
            res = []
            while bm:
                if bm & 1: res.append(length)
                length += 1
                bm >>= 1
            yield res
        return
    
    

def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    """
    Solution to Leetcode #139 using Aho Corasick
    """
    ac = AhoCorasick(wordDict)
    arr = [False] * len(s)
    for i, lengths in enumerate(ac.searchLengths(s)):
        for j in lengths:
            if j == i + 1: arr[i] = True
            elif j > i + 1: break
            elif arr[i - j]: arr[i] = True
            if arr[i]: break
    return arr[-1]

def wordBreak2(self, s: str, wordDict: List[str]) -> List[str]:
    """
    Solution to Leetcode #140 using Aho Corasick
    """
    n = len(s)
    ac = AhoCorasick(wordDict)
    dp = [[] for _ in range(n)]
    for (i, inds) in ac.searchEndIndices(s):
        for j in inds:
            w = wordDict[j]
            if len(w) == i + 1:
                dp[i].append(w)
                continue
            for s2 in dp[i - len(w)]:
                dp[i].append(" ".join([s2, w]))
    return dp[-1]

def Manacher(self, s: str) -> str:
    """
    Manacher's algorithm
    """
    s2 = ["#"]
    for l in s:
        s2.append(l)
        s2.append("#")
    length = len(s2)
    curr_centre = 0
    curr_right = 0
    max_len = 0
    max_i = 0
    LPS = [0] * length
    for i in range(length):
        if i < curr_right:
            mirror = 2 * curr_centre - i
            LPS[i] = min(LPS[mirror], curr_right - i)
            if i + LPS[i] != curr_right:
                continue
        while i - LPS[-1] - 1 >= 0 and i + LPS[-1] + 1 < length and\
                s2[i + LPS[-1] + 1] == s2[i - LPS[-1] - 1]:
            LPS[-1] += 1
        if LPS[-1] > max_len:
            max_len = LPS[-1]
            max_i = i
        if i + LPS[-1] > curr_right:
            curr_right = i + LPS[-1]
            curr_centre = i
    i = (max_i - max_len) >> 1
    return s[i: i + max_len]
