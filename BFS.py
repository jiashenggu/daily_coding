# 314. Binary Tree Vertical Order Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import defaultdict
class Solution:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []

        columnTable = defaultdict(list)
        min_column = max_column = 0
        queue = deque([(root, 0)])

        while queue:
            node, column = queue.popleft()

            if node is not None:
                columnTable[column].append(node.val)
                min_column = min(min_column, column)
                max_column = max(max_column, column)

                queue.append((node.left, column - 1))
                queue.append((node.right, column + 1))

        return [columnTable[x] for x in range(min_column, max_column + 1)]

# 1654. Minimum Jumps to Reach Home
class Solution:
    def minimumJumps(self, forbidden: List[int], a: int, b: int, x: int) -> int:
        forbidden = set(forbidden)
        # each node in the queue is (position, jumped backward last time, number of steps)
        queue = collections.deque([(0, False, 0)])
        # Notice it is important to keep the False in seen not just the position
        seen = set([0, False])
        upper_bound = 6000

        def valid_position(pos, back):
            return 0 <= pos <= upper_bound and (pos, back) not in seen and pos not in forbidden

        while queue:
            for _ in range(len(queue)):
                pos, back, steps = queue.pop()
                if pos == x:
                    return steps
                if valid_position(pos + a, False):
                    seen.add((pos + a, back))
                    queue.appendleft((pos + a, False, steps + 1))
                if not back and valid_position(pos - b, True):
                    seen.add((pos - b, True))
                    queue.appendleft((pos - b, True, steps + 1))
        return -1
    # 1345. Jump Game IV
class Solution:
    def minJumps(self, arr: List[int]) -> int:

        tmp = []
        i = 0
        while i < len(arr):
            if i == len(arr) - 1 or arr[i] != arr[i + 1]:
                tmp.append(arr[i])
                i += 1
                continue
            a = i
            while i < len(arr) - 1 and arr[i] == arr[i + 1]:
                i += 1
            b = i
            tmp.extend([arr[a], arr[b]])
            i += 1

        arr = tmp
        n = len(arr)
        # print(arr)
        m = defaultdict(list)
        vis = n * [0]
        for i in range(n):
            m[arr[i]].append(i)

        ans = 0
        q = deque([0])
        vis[0] = 1
        while q:
            nq = len(q)
            # print(q)
            for _ in range(nq):
                u = q.popleft()
                if u == n - 1:
                    return ans
                for v in reversed(m[arr[u]]):
                    if u != v and not vis[v]:
                        vis[v] = 1
                        q.append(v)
                m[arr[u]].clear()
                if u - 1 >= 0 and not vis[u - 1]:
                    vis[u - 1] = 1
                    q.append(u - 1)
                if u + 1 < n and not vis[u + 1]:
                    vis[u + 1] = 1
                    q.append(u + 1)
            ans += 1
        return

    # 488. Zuma Game
ptn = re.compile(r'(\w)\1{2,}')
class Solution:
    def findMinStep(self, board: str, hand: str) -> int:
        q = deque([(board, hand)])
        seen = {(board, hand)}
        ans = 0
        while q:
            nq = len(q)
            for _ in range(nq):
                board, hand = q.popleft()
                if not board:
                    return ans
                for i in range(len(board)):
                    for j in range(len(hand)):
                        new_board = board[:i] + hand[j] + board[i:]
                        new_hand = hand[:j] + hand[j + 1:]
                        if (new_board, new_hand) in seen:
                            continue
                        seen.add((new_board, new_hand))
                        cnt = 1
                        while cnt:
                            new_board, cnt = ptn.subn('', new_board)

                        q.append((new_board, new_hand))

            ans += 1
        return -1
# 407. Trapping Rain Water II
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        if len(heightMap) <= 2 or len(heightMap[0]) <= 2:
            return 0

        m, n = len(heightMap), len(heightMap[0])
        visited = [[0 for _ in range(n)] for _ in range(m)]
        pq = []
        for i in range(m):
            for j in range(n):
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                    visited[i][j] = 1
                    heapq.heappush(pq, (heightMap[i][j], i * n + j))

        res = 0
        dirs = [-1, 0, 1, 0, -1]
        while pq:
            height, position = heapq.heappop(pq)
            for k in range(4):
                nx, ny = position // n + dirs[k], position % n + dirs[k + 1]
                if nx >= 0 and nx < m and ny >= 0 and ny < n and visited[nx][ny] == 0:
                    if height > heightMap[nx][ny]:
                        res += height - heightMap[nx][ny]
                    visited[nx][ny] = 1
                    heapq.heappush(pq, (max(height, heightMap[nx][ny]), nx * n + ny))
        return res

# 301. Remove Invalid Parentheses
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        def isValid(x):
            stk = []
            for ch in x:
                if ch not in "()":
                    continue
                if ch == '(':
                    stk.append(ch)
                elif ch == ')':
                    if not stk or stk[-1] != '(':
                        return False
                    else:
                        stk.pop()
            return len(stk) == 0

        ans = []
        q = collections.deque([s])
        vis = set()
        while q:
            nq = len(q)
            for _ in range(nq):
                cur = q.popleft()
                # print(cur)
                if isValid(cur):
                    ans.append(cur)
                else:
                    for i in range(len(cur)):
                        if cur[i] not in "()":
                            continue
                        nex = cur[:i] + cur[i + 1:]
                        if nex in vis:
                            continue
                        vis.add(nex)
                        q.append(nex)
            if len(ans) > 0:
                break
        return ans


# 279. Perfect Squares
class Solution:
    def numSquares(self, n):

        # list of square numbers that are less than `n`
        square_nums = [i * i for i in range(1, int(n ** 0.5) + 1)]

        level = 0
        queue = {n}
        while queue:
            level += 1
            # ! Important: use set() instead of list() to eliminate the redundancy,
            # which would even provide a 5-times speedup, 200ms vs. 1000ms.
            next_queue = set()
            # construct the queue for the next level
            for remainder in queue:
                for square_num in square_nums:
                    if remainder == square_num:
                        return level  # find the node!
                    elif remainder < square_num:
                        break
                    else:
                        next_queue.add(remainder - square_num)
            queue = next_queue
        return level
# 1368. Minimum Cost to Make at Least One Valid Path in a Grid
class Solution:
    def minCost(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        min_cost = collections.defaultdict(lambda: math.inf, {(0, 0): 0})

        q = collections.deque([(0, 0, 0)])

        def neightbourhood(x, y):
            if y + 1 < n:
                yield x, y + 1, int(grid[x][y] != 1)
            if y - 1 >= 0:
                yield x, y - 1, int(grid[x][y] != 2)
            if x + 1 < m:
                yield x + 1, y, int(grid[x][y] != 3)
            if x - 1 >= 0:
                yield x - 1, y, int(grid[x][y] != 4)

        while q:
            i, j, cost = q.popleft()
            if i == m - 1 and j == n - 1:
                return cost
            for ni, nj, step_cost in neightbourhood(i, j):
                ncost = cost + step_cost
                if ncost < min_cost[ni, nj]:
                    min_cost[ni, nj] = ncost
                    if step_cost:
                        q.append((ni, nj, ncost))
                    else:
                        q.appendleft((ni, nj, ncost))
        return 0


# 1293. Shortest Path in a Grid with Obstacles Elimination
# BFS
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        rows, cols = len(grid), len(grid[0])
        target = (rows - 1, cols - 1)

        # if we have sufficient quotas to eliminate the obstacles in the worst case,
        # then the shortest distance is the Manhattan distance
        if k >= rows + cols - 2:
            return rows + cols - 2

        # (row, col, remaining quota to eliminate obstacles)
        state = (0, 0, k)
        # (steps, state)
        queue = deque([(0, state)])
        seen = set(state)

        while queue:
            steps, (row, col, k) = queue.popleft()

            # we reach the target here
            if (row, col) == target:
                return steps

            # explore the four directions in the next step
            for new_row, new_col in [(row, col + 1), (row + 1, col), (row, col - 1), (row - 1, col)]:
                # if (new_row, new_col) is within the grid boundaries
                if (0 <= new_row < rows) and (0 <= new_col < cols):
                    new_eliminations = k - grid[new_row][new_col]
                    new_state = (new_row, new_col, new_eliminations)
                    # add the next move in the queue if it qualifies
                    if new_eliminations >= 0 and new_state not in seen:
                        seen.add(new_state)
                        queue.append((steps + 1, new_state))

        # did not reach the target
        return -1
# A*
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:

        rows, cols = len(grid), len(grid[0])
        target = (rows - 1, cols - 1)

        def manhattan_distance(row, col):
            return target[0] - row + target[1] - col

        # (row, col, remaining_elimination)
        state = (0, 0, k)

        # (estimation, steps, state)
        # h(n) = manhattan distance,  g(n) = 0
        queue = [(manhattan_distance(0, 0), 0, state)]
        seen = set(state)

        while queue:
            estimation, steps, (row, col, remain_eliminations) = heapq.heappop(queue)

            # we can reach the target in the shortest path (manhattan distance),
            #   even if the remaining steps are all obstacles
            remain_min_distance = estimation - steps
            if remain_min_distance <= remain_eliminations:
                return estimation

            # explore the four directions in the next step
            for new_row, new_col in [(row, col + 1), (row + 1, col), (row, col - 1), (row - 1, col)]:
                # if (new_row, new_col) is within the grid boundaries
                if (0 <= new_row < rows) and (0 <= new_col < cols):
                    new_eliminations = remain_eliminations - grid[new_row][new_col]
                    new_state = (new_row, new_col, new_eliminations)

                    # if the next direction is worth exploring
                    if new_eliminations >= 0 and new_state not in seen:
                        seen.add(new_state)
                        new_estimation = manhattan_distance(new_row, new_col) + steps + 1
                        queue.append((new_estimation, steps + 1, new_state))

        # did not reach the target
        return -1
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