#!/usr/bin/env python

from typing import Callable, Dict, List, Optional, Tuple, Union, Any

class SegmentTree(object):
    
    """
    Creates a segment tree for an integer sequence based around
    a specified associative binary operation.

    On initialization, the sequence is entirely populated with the
    identity element of the given binary operation
    
    Initialization args:
        Required positional:
        mn (int): Lower bound for sequence index values
        mx (int): Upper bound for sequence index values
        
        Optional named:
        op (string 2-tuple): Specifies the associative binary
                operation to be applied and its identity, either as
                a string identifying a standard binary operation or
                a 2-tuple giving the binary operation at index 0
                and the identity at index 1.
                The standard binary operations implemented are:
                 "sum" (gives the interval sums for real numeric values-
                    identity 0)
                 "product" (gives the interval products for
                    real numeric values- identity 1)
                 "max" (gives the interval maxima for real numeric
                    values- identity -float("inf"))
                 "min" (gives the interval minima for real numeric
                    values- identity float("inf"))
                 "union" (gives the union of sets over the intervals
                    for sets- identity set(), the empty set)
                 "bitwise and" (gives the interval bitwise and for
                    integers- identity -1).
            Default: "sum", or equivalently:
                    {"sum": (lambda x, y: x + y, 0)}
    
    Attributes:
        min (int): The index of the first term in the sequence
        max (int): The index of the final term in the sequence
        op (2-tuple): Contains at index 0 the binary assiciative
                operation as a function that takes two ordered
                inputs and gives a single output of the same
                kind, and at index 1 the identity element of that
                binary operation.
        size (int): The number of terms of the sequence (equal to
                max - min + 1).
        tree (list): A list of length twice the size representing
                the segment tree. The second half of this list is
                the same as the original sequence (in the same order).
        offset (int): The difference between the position of the first
                term in the sequence in the attribute tree (i.e. the
                attribute size) and index of the first term in the
                sequence (i.e. the attribute min).
                Consequently this takes the value (size - min).
        
    Methods:
        (See method documentation for specific details)
        query(): Finds the interval value for a specified subset
                of the binary operations.
        update(): Sets the number associated with a given sequence
                value for one of the binary operations.
        populate(): Sets the values for the sequence starting at a
                given index.
    """
    
    std_ops = {"sum": (lambda x, y: x + y, 0),
               "product": (lambda x, y: x * y, 1),
                "max": (lambda x, y: max(x, y), -float("inf")),
                "min": (lambda x, y: min(x, y), float("inf")),
                "union": (lambda x, y: x.union(y), set()),
                "bitwise_and": (lambda x, y: x & y, -1),
    }
    
    def __init__(self, mn: int, mx: int, op: Union[str, Tuple[Callable[[Any, Any], Any]]]="sum"):
        self.min = mn
        self.max = mx
        self.size = mx - mn + 1
        self.offset = self.size - self.min
        self.op = self.std_ops[op] if isinstance(op, str) else op
        self.tree = [self.op[1] for _ in range(2 * self.size)]
    
    def __getitem__(self, i: int) -> Optional[Any]:
        """
        Returns the term in the sequence corresponding to the
        index i, if there is such a term, otherwise None.
        """
        if i < self.min or i > self.max:
            return None
        return self.tree[i + self.offset]

    def query(self, l: int, r: int) -> Any:
        """
        For the contiguous subsequence with indices no less than l
        and no greater than r, returns the result of repeatedly
        replacing the first and second terms of the subsequence with
        the result of the binary operation of the first term with the
        second term (in that order) until a single term remains.

        If the subsequence is empty (e.g. if l > r or the range [l, r]
        is outside the range of valid indices), then returns the
        identity of the binary operation.

        Args:
            Required positional:
            l (int): The smallest possible index of the subsequence.
            r (int): The largest possible index of the subsequence.
        
        Returns:
        Value of the same type as that of the terms in the sequence,
        representing result the application of the binary operation in
        the specified subsequence with indices between l and r inclusive
        as described above if the subsequence is not empty, otherwise
        the identity element of the binary operation.
        """
        l = max(l, self.min) + self.offset
        r = min(r, self.max) + self.offset
        
        res = self.op[1] # The identity of the operation
        while l < r:
            if l & 1:
                res = self.op[0](res, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = self.op[0](res, self.tree[r])
            l >>= 1
            r >>= 1
        return res
    
    def update(self, i: int, val: Any) -> None:
        """
        Changes the value of the term in the sequence with index i to
        the value val, updating the rest of the segment tree to
        reflect this change.

        Args:
            Required positional:
            i (int): The index of the sequence term to be changed.
                    In order for a change to be made, this must be
                    an integer between the attributes min and max
                    inclusive.
            val (any): A value of the same type as that of the terms
                    in the sequence (i.e. one that can be used as
                    either the first or second argument in the binary
                    operation alongside another term) that should
                    replace the existing term at index i in the
                    sequence.
        
        Returns:
        None
        """
        if i < self.min or i > self.max: return
        i += self.offset
        if self.tree[i] == val: return
        self.tree[i] = val
        while i > 1:
            self.tree[i >> 1] = self.op[0](self.tree[i], self.tree[i ^ 1])
            i >>= 1
        return
    
    def populate(self, i0: int, arr: List[Any]) -> None:
        """
        Overwrites part or all of the sequence in a contiguous block
        starting at index i0 with the sequence represented by the array
        arr, in such a way that the number of terms in the sequence does
        not change.
        Any terms in arr that end up corresponding to indices outside
        of the range [min, max] will not be used.

        Args:
            Required positional:
            i (int): The index of the first term in the sequence to be
                    replaced.
                    In order for a change to be made, this must be
                    an integer between the attributes min and max
                    inclusive.
            arr (list of any type): An ordered list of values of
                    the same type as that of the terms in the sequence
                    (i.e. values that can be used as either the first
                    or second argument in the binary operation alongside
                    one another or another term) that should replace the
                    terms in the sequence as a contiguous block starting
                    from the term with index i0.
                    I.e. arr[0] replaces the term with index i0, arr[1]
                    replaces the term with index i0 + 1, ... arr[k]
                    replaces the term with index i0 + k for integer k
                    between 0 and len(arr) - 1 inclusive.
        
        Returns:
        None
        """
        if i0 < self.min:
            arr = arr[self.min - i0:]
        i0 += self.offset
        for i, val in enumerate(arr, start=i0):
            self.tree[i] = val
        l = i0
        r = l + len(arr)
        while l > 1:
            for i in range(l, r):
                i2 = i >> 1
                self.tree[i2] = self.op[0](self.tree[i], self.tree[i ^ 1])
            l >>= 1
            r = ((r + 1) >> 1)
        return
        
def lengthOfLIS(nums: List[int], k: Optional[int]=None) -> int:
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
    
    Example:
        >>> lengthOfLIS([4,2,1,4,3,4,5,8,15], k = 3)
        5
    
    Solution to Leetcode #2407: Longest Increasing Subsequence II
    and (for k=None) Leetcode #300: Longest Increasing Subsequence.
    
    Original problem 2407 description:

    You are given an integer array nums and an integer k.

    Find the longest subsequence of nums that meets the following
    requirements:

    The subsequence is strictly increasing and
    The difference between adjacent elements in the subsequence is at most
    k.
    Return the length of the longest subsequence that meets the
    requirements.

    A subsequence is an array that can be derived from another array by
    deleting some or no elements without changing the order of the remaining
    elements.
    
    """
    n = len(nums)
    if k is None: k = float("inf")
    if n <= 1: return n
    mn, mx = float("inf"), -float("inf")
    for num in nums:
        mn = min(mn, num)
        mx = max(mx, num)
    n_sum = 0
    st = SegmentTree(mn, mx, op=(lambda x, y: max(x, y), 0))
    
    res = 1
    for num in nums:
        max_prov = st.query(max(mn, num - k), num) + 1
        res = max(res, max_prov)
        st.update(num, max_prov)
    return res

def minimumValueSum(nums: List[int], andValues: List[int]) -> int:
    """
    Consider all the ways in which to partition the sequence of non-negative
    integers nums into exactly len(andValues) contiguous subsequences, such
    that the bitwise and value in andValues. Assign to each such partitioning
    a value equal to the sum of the final element in each of the contiguous
    subsequences.

    This function computes and returns the minimum of these assigned values
    (among all possible such partitionings) if any exist, otherwise it
    returns -1

    Args:
        Required positional:
        nums (list/tuple of ints): the sequence of non-negative integers
                to be partitioned
        andValues (list/tuple of ints): the values of the bitwise and
                that the contiguous subsequences of nums (in order) should
                achieve.
        
    Returns:
        Integer (int) giving the smallest sum of the final elements in
        partitions possible to achieve for partitionings that fulfill
        the given restrictions, if any. If no such partitionings exist,
        returns -1.

    Example:
        >>> minimumValueSum([2,3,5,7,7,7,5], [0,7,5])
        17

    Solution to Leetcode #3117:  Minimum Sum of Values by Dividing Array.
    
    Original problem #3117 description:

    You are given two arrays nums and andValues of length n and m respectively.

    The value of an array is equal to the last element of that array.

    You have to divide nums into m disjoint contiguous 
    subarrays
    such that for the ith subarray [li, ri], the bitwise AND of the subarray
    elements is equal to andValues[i], in other words,
    nums[li] & nums[li + 1] & ... & nums[ri] == andValues[i] for all 1 <= i <= m,
    where & represents the bitwise AND operator.

    Return the minimum possible sum of the values of the m subarrays nums is
    divided into. If it is not possible to divide nums into m subarrays
    satisfying these conditions, return -1.
    """
    n = len(nums)
    m = len(andValues)
    if m > n: return -1

    if len(andValues) == 1:
        ba = -1
        for num in nums:
            if num & andValues[0] != andValues[0]: return -1
            ba &= num
        if ba != andValues[0]: return -1
        return nums[-1]

    row = []
    ba = -1
    idx0 = -1
    for i, num in enumerate(nums):
        if num & andValues[0] != andValues[0]: break
        ba &= num
        if ba == andValues[0]:
            if not row:
                idx0 = i
            row.append(num)
        elif row:
            row.append(float("inf"))
    if not row: return -1
    while row and not isinstance(row[-1], int):
        row.pop()
    mn_seg = SegmentTree(idx0, len(row) + idx0, "min")
    mn_seg.populate(idx0, row)
    for j in range(1, m - 1):
        rng = []
        idx0 += 1
        i0 = idx0
        excl_starts = {}
        row = []
        for i in range(idx0, n):
            av = andValues[j]
            if nums[i] & av != av:
                i0 = i + 1
                excl_starts = {}
                if row: row.append(float("inf"))
                continue
            num = nums[i]
            excl_starts2 = {}
            excl_starts_mn = i + 1
            b_i = 0
            while num:
                if num & 1 and not av & 1:
                    excl_starts2[b_i] = excl_starts.get(b_i, i)
                    excl_starts_mn = min(excl_starts_mn, excl_starts2[b_i])
                num >>= 1
                av >>= 1
                b_i += 1
            excl_starts = excl_starts2
            if excl_starts_mn <= i0 and row:
                row.append(float("inf"))
                continue
            val = mn_seg.query(i0 - 1, excl_starts_mn - 2)
            if not isinstance(val, int):
                if row: row.append(float("inf"))
                continue
            if not row:
                idx0 = i
            row.append(mn_seg.query(i0 - 1, excl_starts_mn - 2) + nums[i])
        if not row: return -1
        while row and not isinstance(row[-1], int):
            row.pop()
        mn_seg = SegmentTree(idx0, len(row) + idx0, "min")
        mn_seg.populate(idx0, row)
        
    res = float("inf")
    ba = -1
    for i in reversed(range(idx0 + 1, n)):
        ba &= nums[i]
        if ba & andValues[-1] != andValues[-1]: break
        elif ba != andValues[-1]: continue
        if i <= idx0 + len(row):
            res = min(res, row[i - 1 - idx0] + nums[-1])
    return res if isinstance(res, int) else -1

if __name__ == "__main__":
    print("\nlengthOfLIS([4,2,1,4,3,4,5,8,15], k = 3) = "
            f"{lengthOfLIS([4,2,1,4,3,4,5,8,15], k = 3)}")
    
    print("\nminimumValueSum([2,3,5,7,7,7,5], [0,7,5]) = "
            f"{minimumValueSum([2,3,5,7,7,7,5], [0,7,5])}")
