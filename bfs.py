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
# def onediff(s1, s2):
#     cnt = 0
#     i = 0
#     n = len(s1)
#     for i in range(n):
#         if s1[i]!=s2[i]:
#             cnt+=1
#     return cnt==1
#
# def ladderLength(beginWord, endWord, wordList):
#     if endWord not in wordList:
#         return 0
#     n = len(wordList)
#     q = [beginWord]
#     ans = 1
#     vis = set()
#     while q:
#         n = len(q)
#         for _ in range(n):
#             cur = q.pop(0)
#             if cur==endWord:
#                 return ans
#             for word in wordList:
#                 if onediff(word, cur) and word not in vis:
#                     q.append(word)
#                     vis.add(word)
#         ans += 1
#     return 0
#
#
# # ******************************结束写代码******************************
#
#
# try:
#     _beginWord = input()
# except:
#     _beginWord = None
#
# try:
#     _endWord = input()
# except:
#     _endWord = None
#
# _wordList_cnt = 0
# _wordList_cnt = int(input())
# _wordList_i = 0
# _wordList = []
# while _wordList_i < _wordList_cnt:
#     try:
#         _wordList_item = input()
#     except:
#         _wordList_item = None
#     _wordList.append(_wordList_item)
#     _wordList_i += 1
#
# res = ladderLength(_beginWord, _endWord, _wordList)
#
# print(str(res) + "\n")


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
# class Solution:
# def shortestDistance(self, grid: List[List[int]]) -> int:
#     m, n = len(grid), len(grid[0])
#     cnt = 0
#     for i in range(m):
#         for j in range(n):
#             if grid[i][j] == 1:
#                 cnt += 1
#
#     def bfs(sr, sc):
#         q = [[sr, sc]]
#         ans = 0
#         step = 0
#         buildings = cnt
#         while q:
#             length = len(q)
#             for _ in range(length):
#                 r, c = q.pop(0)
#                 if grid[r][c] == 1:
#                     buildings -= 1
#                     ans += step
#                     if buildings == 0:
#                         return ans
#                     continue
#
#                 for nr, nc in [[r + 1, c], [r - 1, c], [r, c + 1], [r, c - 1]]:
#                     if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] != 2 and vis[nr][nc] == 0:
#                         vis[nr][nc] = 1
#                         q.append([nr, nc])
#             step += 1
#         return float("inf")
#
#     ret = float("inf")
#     for i in range(m):
#         for j in range(n):
#             if grid[i][j] == 0:
#                 vis = [[0] * n for _ in range(m)]
#                 ret = min(bfs(i, j), ret)
#     return ret if ret != float("inf") else -1
#
#
# # ******************************结束写代码******************************
#
#
# _grid_rows = 0
# _grid_cols = 0
# _grid_rows = int(input())
# _grid_cols = int(input())
#
# _grid = []
# for _grid_i in range(_grid_rows):
#     _grid_temp = map(int, re.split(r'\s+', input().strip()))
#     _grid.append(list(_grid_temp))
#
# res = shortestDistance(_grid)
#
# print(str(res) + "\n")


# !/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re
import queue


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 请完成下面这个函数，实现题目要求的功能
# 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# ******************************开始写代码******************************


def isCousins(root, x, y):
    parent = {}
    q = [root]
    parent[root]=None
    px, py = None, None
    dx, dy = 0, 0
    d = 0
    while q:
        n = len(q)
        for _ in range(n):
            c = q.pop(0)
            if c.val==x:
                px = parent[c]
                dx=d
            if c.val==y:
                py = parent[c]
                dy=d
            if c.left:
                parent[c.left]=c
                q.append(c.left)
            if c.right:
                parent[c.right]=c
                q.append(c.right)
        d+=1
    print(px.val, py.val, dx, dy)
    return px!=py and dx==dy



# ******************************结束写代码******************************

def construct_tree(arr):
    root = TreeNode(int(arr[0]))
    ind = 1
    q = queue.Queue()
    q.put(root)
    while ind < len(arr):
        node = q.get()
        if arr[ind] != 'null':
            node.left = TreeNode(int(arr[ind]))
            q.put(node.left)
        ind += 1
        if ind >= len(arr):
            break
        if arr[ind] != 'null':
            node.right = TreeNode(int(arr[ind]))
            q.put(node.right)
        ind += 1
    return root


def get_input(string):
    if not string:
        return None
    string = string.split(',')
    root = construct_tree(string)
    return root


_root = get_input(input()[1:-1])

_x = int(input())

_y = int(input())

res = isCousins(_root, _x, _y)

print("true" if res else "false")