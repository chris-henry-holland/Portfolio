#!/usr/bin/env python

from typing import Callable, Dict, List, Optional, Tuple, Union

class SegmentTree(object):
    
    """
    Creates a segment tree for an integer sequence based around
    specified assoicative binary operations.
    
    Args:
        Required positional:
        mn (int): Lower bound for sequence index values
        mx (int): Upper bound for sequence index values
        
        Optional named:
        ops (dict): Specifies the binary operation(s) to be considered.
                Given as a dictionary whose keys are unique strings
                with corresponding value either a 2-tuple whose 0th
                index is the associative binary operation (as a
                function) and whose 1st index is the identity of that
                binary operation, or a string representing
                a standard binary operation. The standard binary
                operations implemented are "sum" (gives the interval
                sums- identity 0), "product" (gives the inteval
                product- identity 1), "max" (gives the interval
                maximum), "min" (gives the interval minimum) and
                "union" (gives the union of sets on the interval).
                Note- it is possible to use the same binary operation
                multiple times (as long as each has a different key),
                which can be useful in the case where different
                iterations through the sequence result in different
                values for the same element (see example below)
            Default: {"sum": "sum"}, or equivalently:
                    {"sum": (lambda x, y: x + y, 0)}
    
    Attributes:
        
        
    Methods:
        (See method documentation for specific details)
        query(): Finds the interval value for a specified subset
                of the binary operations.
        update(): Sets the number associated with a given sequence
                value for one of the binary operations.
                
    """
    
    std_ops = {"sum": (lambda x, y: x + y, 0), "product": (lambda x, y: x * y, 1),
              "max": (lambda x, y: max(x, y), -float("inf")),
              "min": (lambda x, y: min(x, y), float("inf")),
              "union": (lambda x, y: x.union(y), set()),
              }
    
    def __init__(self, mn: int, mx: int, ops: Dict[str, Union[str, tuple]]={"sum": "sum"}):
        self.min = mn
        self.max = mx
        self.size = mx - mn + 1
        self.offset = self.size - self.min
        self.ops = {k: self.std_ops[op] if isinstance(op, str)\
                    else op for k, op in ops.items()}
        self.ops_order = tuple(self.ops.keys())
        self.ops_inds = {v: i for i, v in enumerate(self.ops_order)}
        self.tree = [[self.ops[k][1] for k in self.ops_order]\
                        for _ in range(2 * self.size)]
    
    def __getitem__(self, i: int) -> Optional[Dict[str, Union[int, float]]]:
        if i < self.min or i > self.max:
            return None
        return {op: self.tree[i + self.offset][self.ops_inds[op]]\
                for op in self.ops_order}
                
    def add_ops(self, ops: Dict[str, Union[str, tuple]]) -> None:
        ops = {k: self.std_ops[op] if isinstance(op, str)\
                    else op for k, op in ops.items() if k not in self.ops.keys()}
        ops_order = tuple(ops.keys())
        self.ops_inds.update({v: i for i, v in enumerate(ops_order, start=len(self.ops))})
        for lst in self.tree:
            lst.extend([ops[k][1] for k in ops_order])
        self.ops.update(ops)
        self.ops_order = (*self.ops_order, *ops_order)
        return
    
    def query(self, l: int, r: int, ops: Optional[Union[List[str], str]]=None)\
            -> Dict[str, Union[int, float]]:
        if ops is None: ops = self.ops_order
        elif isinstance(ops, str): ops = [ops]
        op_funcs = [(self.ops_inds[op], self.ops[op][0]) for op in ops]
        
        l += self.offset
        r += self.offset
        res = [self.ops[op][1] for op in ops] # The identity of each operation
        while l < r:
            if l & 1:
                for j, (i, func) in enumerate(op_funcs):
                    res[j] = func(res[j], self.tree[l][i])
                l += 1
            if r & 1:
                r -= 1
                for j, (i, func) in enumerate(op_funcs):
                    res[j] = func(res[j], self.tree[r][i])
            l >>= 1
            r >>= 1
        return {ops[j]: res[j] for j in range(len(op_funcs))}
    
    def update(self, i: int, val, op: str) -> None:
        i += self.offset
        j = self.ops_inds[op]
        self.tree[i][j] = val
        while i > 1:
            self.tree[i >> 1][j] = self.ops[op][0](self.tree[i][j], self.tree[i ^ 1][j])
            i >>= 1
        return
    
    def populate(self, i0: int, arr: Union[list, tuple], op: str) -> None:
        j = self.ops_inds[op]
        if i0 < self.min:
            arr = arr[self.min - i0:]
        i0 += self.offset
        for i, num in enumerate(arr, start=i0):
            self.tree[i][j] = num
        l = i0
        r = l + len(arr)
        while l > 1:
            #print(f"l = {l}")
            #print(self.tree)
            for i in range(l, r):
                i2 = i >> 1
                self.tree[i2][j] = self.ops[op][0](self.tree[i][j], self.tree[i ^ 1][j])
                #print(f"i2 = {i2}")
            l >>= 1
            r = ((r + 1) >> 1)
        return
        
def LengthOfLIS(nums: List[int], k: Optional[int]=None) -> int:
    """
    Finds the length of the longest strictly increasing subsequence
    of the integer sequence nums for which no two successive elements
    in the subsequence differ by more than k (with no such restriction
    if k is given as None)
    
    Args:
        Required positional:
        nums (list/tuple of ints): the sequence
        k (int/None): the maximum difference between any two successive
                elements in the subsequences allowed to be considered
                (with None meaning there is no restriction on the
                difference between successive elements)
        
    Returns:
        Integer (int) giving the longest subsequence length for the
        specified restrictions. 
    
    Note- solution to Leetcode #2407: Longest Increasing Subsequence II
    and (for k=None) Leetcode #300: Longest Increasing Subsequence
    
    Example:
        >>> LengthOfLIS([4,2,1,4,3,4,5,8,15], k = 3)
        5
    """
    n = len(nums)
    if k is None: k = float("inf")
    if n <= 1: return n
    mn, mx = float("inf"), -float("inf")
    for num in nums:
        mn = min(mn, num)
        mx = max(mx, num)
    n_sum = 0
    ops = {"max": "max"}
    seg = SegmentTree(mn, mx, ops=ops)
    
    res = 1
    for num in nums:
        max_prov = seg.query(max(mn, num - k), num, ops="max")["max"] + 1
        res = max(res, max_prov)
        seg.update(num, max_prov, op="max")
    return res

def IncreasingSubsequenceDistribution(nums: List[int], k: Optional[int]=None)\
        -> Tuple[int]:
    """
    Finds the number of strictly increasing subsequences of each
    length there exists of the integer sequence nums for which no
    two successive elements in the subsequence differ by more than
    k (with no such restriction if k is given as None)
    
    Args:
        Required positional:
        nums (list/tuple of ints): the sequence
        k (int/None): the maximum difference between any two successive
                elements in the subsequences allowed to be considered
                (with None meaning there is no restriction on the
                difference between successive elements)
        
    Returns:
        Tuple (tuple) of integers (int) for which the ith index gives
        the number of strictly increasing subsequences (subject to
        the specified restrictions) with exactly i elements. The length
        of the tuple is one greater than the number of elements in the
        longest possible such subsequence.
    
    Example:
        >>> IncreasingSubsequenceDistribution([4,2,1,4,3,4,5,8,15], k = 3)
        [1, 9, 13, 14, 9, 2]
        
        This signifies for instance that there are 13 strictly increasing
        subsequences of the given sequence such that any difference
        between successive elements is at most 3 with length 2, and 2
        with length 5 (namely [1,3,4,5,8] and [2,3,4,5,8]).
    """
    n = len(nums)
    if k is None: k = float("inf")
    if n <= 1: return n
    mn, mx = float("inf"), -float("inf")
    for num in nums:
        mn = min(mn, num)
        mx = max(mx, num)
    seg = SegmentTree(mn, mx, ops={"sum1": "sum"})
    res = [1, 0]
    curr_max = 1
    for num in nums:
        prev = f"sum{curr_max}"
        for i in reversed(range(1, curr_max + 1)):
            curr = prev
            if i == 1:
                seg.update(num, seg.tree[num + seg.offset][seg.ops_inds[curr]] + 1,\
                           op=curr)
                res[1] += 1
                continue
            prev = f"sum{i - 1}"
            sum_prov = seg.query(max(mn, num - k), num, ops=prev)[prev]
            seg.update(num, seg.tree[num + seg.offset][seg.ops_inds[curr]]\
                        + sum_prov, op=curr)
            res[i] += sum_prov
        if res[-1] != 0:
            seg.add_ops({f"sum{curr_max + 1}": "sum"})
            res.append(0)
            curr_max += 1
    if res[-1] == 0: res.pop()
    return tuple(res)

if __name__ == "__main__":
    print("\nLengthOfLIS([4,2,1,4,3,4,5,8,15], k = 3) = "
            f"{LengthOfLIS([4,2,1,4,3,4,5,8,15], k = 3)}")
    print("\nIncreasingSubsequenceDistribution([4,2,1,4,3,4,5,8,15], k = 3) = "
            f"{IncreasingSubsequenceDistribution([4,2,1,4,3,4,5,8,15], k = 3)}")
