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