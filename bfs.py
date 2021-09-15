# A*
class Solution:
    def shortestPathBinaryMatrix(self, grid):
        shortest_path = a_star_graph_search(
            start              = (0, 0),
            goal_function      = get_goal_function(grid),
            successor_function = get_successor_function(grid),
            heuristic          = get_heuristic(grid)
        )
        if shortest_path is None or grid[0][0] == 1:
            return -1
        else:
            return len(shortest_path)
def a_star_graph_search(
            start,
            goal_function,
            successor_function,
            heuristic
	):
    visited = set()
    came_from = dict()
    distance = {start: 0}
    frontier = PriorityQueue()
    frontier.add(start)
    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        if goal_function(node):
            return reconstruct_path(came_from, start, node)
        visited.add(node)
        for successor in successor_function(node):
            frontier.add(
                successor,
                priority = distance[node] + 1 + heuristic(successor)
            )
            if (successor not in distance
                or distance[node] + 1 < distance[successor]):
                distance[successor] = distance[node] + 1
                came_from[successor] = node
    return None

def reconstruct_path(came_from, start, end):
    """
    >>> came_from = {'b': 'a', 'c': 'a', 'd': 'c', 'e': 'd', 'f': 'd'}
    >>> reconstruct_path(came_from, 'a', 'e')
    ['a', 'c', 'd', 'e']
    """
    reverse_path = [end]
    while end != start:
        end = came_from[end]
        reverse_path.append(end)
    return list(reversed(reverse_path))

def get_goal_function(grid):
    """
    >>> f = get_goal_function([[0, 0], [0, 0]])
    >>> f((0, 0))
    False
    >>> f((0, 1))
    False
    >>> f((1, 1))
    True
    """
    M = len(grid)
    N = len(grid[0])
    def is_bottom_right(cell):
        return cell == (M-1, N-1)
    return is_bottom_right

def get_successor_function(grid):
    """
    >>> f = get_successor_function([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> sorted(f((1, 2)))
    [(0, 1), (0, 2), (2, 1), (2, 2)]
    >>> sorted(f((2, 1)))
    [(1, 0), (1, 2), (2, 2)]
    """
    def get_clear_adjacent_cells(cell):
        i, j = cell
        return (
            (i + a, j + b)
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            if a != 0 or b != 0
            if 0 <= i + a < len(grid)
            if 0 <= j + b < len(grid[0])
            if grid[i + a][j + b] == 0
        )
    return get_clear_adjacent_cells

def get_heuristic(grid):
    """
    >>> f = get_heuristic([[0, 0], [0, 0]])
    >>> f((0, 0))
    1
    >>> f((0, 1))
    1
    >>> f((1, 1))
    0
    """
    M, N = len(grid), len(grid[0])
    (a, b) = goal_cell = (M - 1, N - 1)
    def get_clear_path_distance_from_goal(cell):
        (i, j) = cell
        return max(abs(a - i), abs(b - j))
    return get_clear_path_distance_from_goal


from heapq import heappush, heappop


class PriorityQueue:

    def __init__(self, iterable=[]):
        self.heap = []
        for value in iterable:
            heappush(self.heap, (0, value))

    def add(self, value, priority=0):
        heappush(self.heap, (priority, value))

    def pop(self):
        priority, value = heappop(self.heap)
        return value

    def __len__(self):
        return len(self.heap)


# normal BFS
from collections import deque

def breadth_first_search(grid):
    N = len(grid)

    def is_clear(cell):
        return grid[cell[0]][cell[1]] == 0

    def get_neighbours(cell):
        (i, j) = cell
        return (
            (i + a, j + b)
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            if a != 0 or b != 0
            if 0 <= i + a < N
            if 0 <= j + b < N
            if is_clear( (i + a, j + b) )
        )

    start = (0, 0)
    goal = (N - 1, N - 1)

    queue = deque()
    if is_clear(start):
        queue.append(start)
    visited = set()
    path_len = {start: 1}

    while queue:
        cell = queue.popleft()
        if cell in visited:
            continue
        if cell == goal:
            return path_len[cell]
        visited.add(cell)
        for neighbour in get_neighbours(cell):
            if neighbour not in path_len:
                path_len[neighbour] = path_len[cell] + 1
            queue.append(neighbour)

    return -1
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