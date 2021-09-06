# #!/bin/python
# # -*- coding: utf8 -*-
# import sys
# import os
# import re
# import collections
# #请完成下面这个函数，实现题目要求的功能
# #******************************开始写代码******************************
#
#
# def  isMaxHeap(nums):
#     q = collections.deque([0])
#     n = len(nums)
#     while q:
#         cur  = q.popleft()
#         if cur*2+1<n:
#             q.append(cur*2+1)
#             if nums[cur]<nums[cur*2+1]:
#                 return False
#         if cur*2+2<n:
#             q.append(cur*2+2)
#             if nums[cur]<nums[cur*2+2]:
#                 return False
#     return True
#
#
# #******************************结束写代码******************************
#
# arr = input()
# nums = [int(item.group(0)) for item in re.finditer(r'-?\d+', arr)]
#
# res = isMaxHeap(nums)
#
# print('true' if res else 'false')

