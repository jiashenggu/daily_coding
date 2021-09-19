# 650. 2 Keys Keyboard
class Solution:
    def minSteps(self, n: int) -> int:
        f = [float("inf")]*(1+n)
        f[0] = 0
        f[1] = 0
        for i in range(2, 1+n):
            j = 1
            while j*j<=i:
                if i%j==0:
                    f[i] = min(f[i], f[j]+i//j, f[i//j]+j)
                j+=1
        print(f)
        return f[n]
# prime factorization
class Solution:
    def minSteps(self, n: int) -> int:
        ans = 0
        d = 2
        while n>1:
            while n%d==0:
                ans+=d
                n//=d
            d+=1
        return ans
# # #!/bin/python
# # # -*- coding: utf8 -*-
# # import sys
# # import os
# # import re
# #
# # #请完成下面这个函数，实现题目要求的功能
# # #当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# # #******************************开始写代码******************************
# #
# # def maxProduct(nums):
# #     n = len(nums)
# #     if n == 1:
# #         return nums[0]
# #     dpMax = [0] * (1 + n)
# #     dpMin = [0] * (1 + n)
# #     for i in range(1, n + 1):
# #         dpMax[i] = max(dpMax[i - 1] * nums[i - 1], dpMin[i - 1] * nums[i - 1], nums[i - 1])
# #         dpMin[i] = min(dpMax[i - 1] * nums[i - 1], dpMin[i - 1] * nums[i - 1], nums[i - 1])
# #     return max(dpMax)
# #
# #
# #
# #
# # #******************************结束写代码******************************
# #
# # def getInput(str):
# #     if not str:
# #         return None
# #     str = str.split(",")
# #     res = []
# #     for num in str:
# #         res.append(int(num))
# #     return res
# #
# # nums = getInput(input()[1:-1])
# # res = maxProduct(nums)
# #
# # print(res)
#
# #!/bin/python
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
# def uniquePathsWithObstacles(obstacleGrid):
#     m, n = len(obstacleGrid), len(obstacleGrid[0])
#     dp = [ n*[0] for _ in range(m)]
#     for i in range(m):
#         if obstacleGrid[i][0]==1:
#             break
#         dp[i][0]=1
#
#     for j in range(n):
#         if obstacleGrid[0][j]==1:
#             break
#         dp[0][j]=1
#
#     for i in range(1, m):
#         for j in range(1, n):
#             if obstacleGrid[i][j]==1:
#                 dp[i][j]=0
#             else:
#                 dp[i][j] = dp[i-1][j] + dp[i][j-1]
#     return dp[m-1][n-1]
#
#
# # ******************************结束写代码******************************
#
# text = input()
# obstacleGrid = list()
# for row in re.findall(re.compile("\[[^\[\]]*\]"), text):
#     if row == "[]":
#         obstacleGrid.append(list())
#         continue
#     obstacleGrid.append([int(num) for num in row[1:-1].split(",")])
#
# res = uniquePathsWithObstacles(obstacleGrid)
#
# print(res)
#


# !/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re


# 请完成下面这个函数，实现题目要求的功能
# 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# ******************************开始写代码******************************


def getMatrixChain(p):
    n = len(p)
    dp = [ n*[float("inf")] for _ in range(n)]
    for i in range(n):
        dp[i][i]=0
    for r in range(1, n):
        for start in range(n-r):
            end = start + r - 1
            for k in range(start, end+1):
                dp[start][end] = min(dp[start][end], dp[start][k]+dp[k+1][end]+p[start]*p[k+1]*p[end+1])
    # print(dp)
    return dp[0][n-2]




# ******************************结束写代码******************************


_p_cnt = 0
_p_cnt = int(input())
_p_i = 0
_p = []
while _p_i < _p_cnt:
    _p_item = int(input())
    _p.append(_p_item)
    _p_i += 1

res = getMatrixChain(_p)

print(str(res) + "\n")
#
#
# # # #!/bin/python
# # # # -*- coding: utf8 -*-
# # # import sys
# # # import os
# # # import re
# # #
# # #
# # # # 请完成下面这个函数，实现题目要求的功能
# # # # 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# # # # ******************************开始写代码******************************
# # #
# # #
# # # def longestPalindromeSub(s):
# # #     n = len(s)
# # #     dp = [n * [0] for _ in range(n)]
# # #     for i in range(n):
# # #         dp[i][i] = 1
# # #     for i in range(n - 1, -1, -1):
# # #         for j in range(i + 1, n):
# # #             if s[i] == s[j]:
# # #                 dp[i][j] = dp[i + 1][j - 1] + 2
# # #             else:
# # #                 dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
# # #     return dp[0][n - 1]
# # #
# # #
# # # # ******************************结束写代码******************************
# # #
# # #
# # # try:
# # #     _s = input()
# # # except:
# # #     _s = None
# # #
# # # res = longestPalindromeSub(_s)
# # #
# # # print(str(res) + "\n")
# # #
# # # # !/bin/python
# # # # -*- coding: utf8 -*-
# # # import sys
# # # import os
# # # import re
# # #
# # #
# # # # 请完成下面这个函数，实现题目要求的功能
# # # # 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# # # # ******************************开始写代码******************************
# # #
# # #
# # # def minCostClimbingStair(cost):
# # #     n = len(cost)
# # #     dp = (1+n)*[0]
# # #
# # #     for i in range(2, n+1):
# # #         dp[i] = min(dp[i-2]+cost[i-2], dp[i-1]+cost[i-1])
# # #     return dp[n]
# # #
# # # # ******************************结束写代码******************************
# # #
# # #
# # # _cost_cnt = 0
# # # _cost_cnt = int(input())
# # # _cost_i = 0
# # # _cost = []
# # # while _cost_i < _cost_cnt:
# # #     _cost_item = int(input())
# # #     _cost.append(_cost_item)
# # #     _cost_i += 1
# # #
# # # res = minCostClimbingStair(_cost)
# # #
# # # print(str(res) + "\n")
# # #
# # # # !/bin/python
# # # # -*- coding: utf8 -*-
# # # import sys
# # # import os
# # # import re
# # #
# # #
# # # # 请完成下面这个函数，实现题目要求的功能
# # # # 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# # # # ******************************开始写代码******************************
# # #
# # #
# # # def minCost(costs):
# # #     n = len(costs)
# # #     dp = [ [0]*3 for _ in range(1+n)]
# # #     for i in range(1, n+1):
# # #         for j in range(3):
# # #             dp[i][0] = min(dp[i - 1][1], dp[i - 1][2]) + costs[i-1][0]
# # #             dp[i][1] = min(dp[i - 1][0], dp[i - 1][2]) + costs[i-1][1]
# # #             dp[i][2] = min(dp[i - 1][0], dp[i - 1][1]) + costs[i-1][2]
# # #     return min(dp[n])
# # #
# # # # ******************************结束写代码******************************
# # #
# # #
# # # _costs_rows = 0
# # # _costs_cols = 0
# # # _costs_rows = int(input())
# # # _costs_cols = int(input())
# # #
# # # _costs = []
# # # for _costs_i in range(_costs_rows):
# # #     _costs_temp = map(int, re.split(r'\s+', input().strip()))
# # #     _costs.append(list(_costs_temp))
# # #
# # # res = minCost(_costs)
# # #
# # # print(str(res) + "\n")