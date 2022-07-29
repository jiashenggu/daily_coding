# 890. Find and Replace Pattern
class Solution:
    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        def change(s):
            d = {}
            ret = ''
            index = 0
            for ch in s:
                if ch not in d:
                    index += 1
                    d[ch] = index
                ret = ret + chr(ord('a') + d[ch])
            # print(ret)
            return ret

        p = change(pattern)
        ans = []
        for word in words:
            if change(word) == p:
                ans.append(word)
        return ans

class Solution(object):
    def findAndReplacePattern(self, words, pattern):
        def match(word):
            m1, m2 = {}, {}
            for w, p in zip(word, pattern):
                if w not in m1: m1[w] = p
                if p not in m2: m2[p] = w
                if (m1[w], m2[p]) != (p, w):
                    return False
            return True

        return filter(match, words)
# 792. Number of Matching Subsequences
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        d = defaultdict(list)
        for word in words:
            d[word[0]].append(word)

        for ch in s:
            nd = len(d[ch])
            for _ in range(nd):
                word = d[ch].pop(0)
                _word = word[1:]
                if _word == "":
                    continue
                d[_word[0]].append(_word)
        ans = 0
        for k, v in d.items():
            ans += len(v)
        return len(words) - ans
# 128. Longest Consecutive Sequence
class Solution:
    def longestConsecutive(self, nums):
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)

        return longest_streak
# 41. First Missing Positive
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        i = 0
        n = len(nums)
        while i < n:
            j = nums[i] - 1
            if 0 <= j < n and nums[j] != nums[i]:
                nums[i], nums[j] = nums[j], nums[i]
            else:
                i += 1
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        return n + 1


class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        if 1 not in nums:
            return 1
        n = len(nums)
        for i in range(n):
            if nums[i] <= 0 or nums[i] > n:
                nums[i] = 1
        for i in range(n):
            a = abs(nums[i])
            if a == n:
                nums[0] = -abs(nums[0])
            else:
                nums[a] = -abs(nums[a])
        for i in range(1, n):
            if nums[i] > 0:
                return i
        if nums[0] > 0:
            return n
        return n + 1


# 380. Insert Delete GetRandom O(1)
class RandomizedSet:

    def __init__(self):
        self.dict = {}
        self.list = []

    def insert(self, val: int) -> bool:
        if val in self.dict:
            return False
        self.dict[val] = len(self.list)
        self.list.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.dict:
            return False
        idx = self.dict[val]
        self.dict[self.list[-1]] = idx
        del self.dict[val]
        self.list[-1], self.list[idx] = self.list[idx], self.list[-1]
        self.list.pop(-1)
        return True

    def getRandom(self) -> int:
        return random.choice(self.list)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

# 299. Bulls and Cows
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        h = defaultdict(int)
        bulls = 0
        cows = 0
        for idx, s in enumerate(secret):
            g = guess[idx]
            if g == s:
                bulls += 1
            else:
                cows += int(h[s] < 0) + int(h[g] > 0)
                h[s] += 1
                h[g] -= 1
        return "{}A{}B".format(bulls, cows)


# 1923. Longest Common Subpath
class RabinKarp:
    def __init__(self, s):
        self.mod = 10 ** 18
        self.pow = [1]
        self.roll = [0]

        p = 10 ** 9 + 7
        for x in s:
            self.pow.append(self.pow[-1] * p % self.mod)
            self.roll.append((self.roll[-1] * p + x) % self.mod)

    def query(self, i, j):
        return (self.roll[j] - self.roll[i] * self.pow[j - i]) % self.mod


class Solution:
    def longestCommonSubpath(self, n: int, paths: List[List[int]]) -> int:
        rks = [RabinKarp(path) for path in paths]

        def fn(x):
            seen = set()
            for rk, path in zip(rks, paths):
                vals = {rk.query(i, i + x) for i in range(len(path) - x + 1)}
                if not seen:
                    seen = vals
                seen &= vals
                if not seen:
                    return False
            return True

        lo, hi = 0, len(paths[0])
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fn(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


# 1153. String Transforms Into Another String
class Solution:
    def canConvert(self, str1: str, str2: str) -> bool:
        if str1==str2:
            return True
        uni = set()
        conversion = dict()
        for l1, l2 in zip(str1, str2):
            if l1 not in conversion:
                conversion[l1] = l2
                uni.add(l2)
            elif conversion[l1]!=l2:
                return False
        if len(uni)<26:
            return True
        return False
# 532. K-diff Pairs in an Array
from collections import Counter

class Solution:
    def findPairs(self, nums, k):
        result = 0

        counter = Counter(nums)

        for x in counter:
            if k > 0 and x + k in counter:
                result += 1
            elif k == 0 and counter[x] > 1:
                result += 1
        return result
# 447. Number of Boomerangs
class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:

        ans = 0
        for i, p1 in enumerate(points):
            m = collections.defaultdict(int)
            for j, p2 in enumerate(points):
                if i == j:
                    continue
                d = (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])
                m[d] += 1
            for v in m.values():
                ans += v * (v - 1)
        return ans


#!/bin/python
# -*- coding: utf8 -*-
import collections
import sys
import os
import re
import itertools


# 请完成下面这个函数，实现题目要求的功能
# ******************************开始写代码******************************

def findSubArrays(nums):
    accu = list(itertools.accumulate(nums))
    m = collections.defaultdict(list)
    ans = []
    n = len(nums)
    print(accu)
    m[0] = [0]
    for i in range(n):
        if accu[i] not in m:
            m[accu[i]].append(i)
        else:
            for j in m[accu[i]]:
                ans.append([j+1, i])
            m[accu[i]].append(i)
    print(m)
    ans.sort()
    return ans


# ******************************结束写代码******************************

arr = input()
nums = [int(item.group(0)) for item in re.finditer(r'-?\d+', arr)]

res = findSubArrays(nums)

res = ['[{},{}]'.format(item[0], item[1]) for item in res]
print('[{}]'.format(','.join(res)))