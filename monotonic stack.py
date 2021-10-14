# 739. Daily Temperatures
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        q = []
        ans = []
        for i in range(n - 1, -1, -1):
            while q and temperatures[q[-1]] <= temperatures[i]:
                q.pop(-1)
            if q:
                ans.append(q[-1] - i)
            else:
                ans.append(0)
            q.append(i)
        return ans[::-1]


# # # 84. Largest Rectangle in Histogram
# # class Solution:
# #     def largestRectangleArea(self, heights: List[int]) -> int:
# #         stk = []
# #         ans = 0
# #         n = len(heights)
# #         for i, h in enumerate(heights):
# #             while stk and heights[stk[-1]]>=h:
# #                 start = stk.pop()
# #                 width = i if not stk else i-stk[-1]-1
# #                 ans = max(ans, heights[start]*width)
# #             stk.append(i)
# #         while stk:
# #             start = stk.pop()
# #             width = n if not stk else n-stk[-1]-1
# #             ans = max(ans, heights[start]*width)
# #         return ans
#
# # !/bin/python
# # -*- coding: utf8 -*-
# import sys
# import os
# import re
#
#
# # 请完成下面这个函数，实现题目要求的功能
# # 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# # ******************************开始写代码******************************
#
#
# def nextGreaterElements(nums):
#     n = len(nums)
#     i=0
#     stk = []
#     ans = [-1]*n
#     while i<2*n:
#         ii = i%n
#         while stk and nums[stk[-1]]<nums[ii]:
#             last = stk.pop()
#             if ans[last]==-1:
#                 ans[last] = nums[ii]
#         stk.append(ii)
#         i+=1
#     return ans
#
#
#
#
# # ******************************结束写代码******************************
#
#
# _nums_cnt = 0
# _nums_cnt = int(input())
# _nums_i = 0
# _nums = []
# while _nums_i < _nums_cnt:
#     _nums_item = int(input())
#     _nums.append(_nums_item)
#     _nums_i += 1
#
# res = nextGreaterElements(_nums)
#
# for res_cur in res:
#     print(str(res_cur) + "\n")



# !/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re


# 请完成下面这个函数，实现题目要求的功能
# ******************************开始写代码******************************

def seeCows(cows):
    n = len(cows)
    ans = 0
    stk  = []
    cows.append(float("inf"))
    for i in range(n+1):
        while stk and cows[stk[-1]]<=cows[i]:
            top = stk.pop()
            # print(stk, top, i)
            ans += (i-top-1)
        stk.append(i)
    return ans


# ******************************结束写代码******************************

arr = input()
cows = [int(item.group(0)) for item in re.finditer(r'-?\d+', arr)]

res = seeCows(cows)

print(res)