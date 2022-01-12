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