# Tiktok
# Input Dictionary:
# 1: ['a', 'b', 'c']
# 2: ['b', 'd']
# 3: ['a', 'e']
# key can be any integer
# values would be a list
# <Integer, List<Character>> map
# Input Query:
# //and, or, not, 3 operators
# "and(not(a), or(b,c))"
# "a" -> [1, 3]
# "not(a)"‍‌‌‌‌‍‌‌‍‍‌‍‍‌‌‌‌‍‌‌ -> [2]
# "or(b,c)" -> [1,2]
# "and(not(a), or(b,c))" -> [2]
# Output:
# [2]
from collections import deque, defaultdict
dictionary = {1: set(['a', 'b', 'c']), 2: set(['b', 'd']), 3: set(['a', 'e'])}
query = "and(not(a), or(b,c))"


class Solution:
    def quer2res(self, dictionary, query):
        q = deque(query.replace(" ", "").replace(",", " ").replace("(", " ").replace(")", " ").split())
        char2val, notchar2val = self.process_dict(dictionary)
        def dfs():
            token = q.popleft()
            print(token)
            if token == "and":
                return dfs() & dfs()
            elif token == "or":
                return dfs() | dfs()
            elif token == "not":
                char = q.popleft()
                return notchar2val[char]
            else:
                return char2val[token]
        return dfs()


def process_dict(self, dictionary):
    char2val, notchar2val = defaultdict(set), defaultdict(set)

    chars = set()
    for key, vals in dictionary.items():
        for val in vals:
            chars.add(val)
    for char in chars:
        for key, vals in dictionary.items():
            if char in vals:
                char2val[char].add(key)
    for char in chars:
        for key, vals in dictionary.items():
            if char not in vals:
                notchar2val[char].add(key)
    return char2val, notchar2val
solution = Solution()
print(solution.quer2res(dictionary, query))

class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        N = len(num)
        ans = []

        def recurse(index, pre, cur, val, s):
            if index == N:
                if val == target and cur == 0:
                    ans.append("".join(s[1:]))
                return
            cur = cur * 10 + int(num[index])
            str_op = str(cur)

            if cur > 0:
                recurse(index + 1, pre, cur, val, s)
            s.append('+')
            s.append(str_op)
            recurse(index + 1, cur, 0, val + cur, s)
            s.pop()
            s.pop()

            if s:
                s.append('-')
                s.append(str_op)
                recurse(index + 1, -cur, 0, val - cur, s)
                s.pop()
                s.pop()

                s.append('*')
                s.append(str_op)
                recurse(index + 1, cur * pre, 0, val - pre + pre * cur, s)
                s.pop()
                s.pop()

        recurse(0, 0, 0, 0, [])
        return ans
# 721. Accounts Merge
class Solution:
    def accountsMerge(self, accounts):
        names = {}
        graph = defaultdict(set)
        for acc in accounts:
            name = acc[0]
            for email in acc[1:]:
                graph[acc[1]].add(email)
                graph[email].add(acc[1])
                names[email] = name

        comps, seen, ans, i = defaultdict(list), set(), [], 0

        def dfs(node, i):
            comps[i].append(node)
            seen.add(node)
            for neib in graph[node]:
                if neib not in seen: dfs(neib, i)

        for email in graph:
            if email not in seen:
                dfs(email, i)
                i += 1

        return [[names[val[0]]] + sorted(val) for _, val in comps.items()]
# 339. Nested List Weight Sum
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
# class NestedInteger:
#    def __init__(self, value=None):
#        """
#        If value is not specified, initializes an empty list.
#        Otherwise initializes a single integer equal to value.
#        """
#
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def add(self, elem):
#        """
#        Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
#        :rtype void
#        """
#
#    def setInteger(self, value):
#        """
#        Set this NestedInteger to hold a single integer equal to value.
#        :rtype void
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class Solution:
    def __init__(self):
        self.ans = 0

    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        def dfs(nestedList):
            tmp = 0
            for nest in nestedList:
                if nest.isInteger():
                    tmp += nest.getInteger()
                else:
                    tmp += dfs(nest.getList())
            self.ans += tmp
            return tmp

        dfs(nestedList)
        return self.ans


# 93. Restore IP Addresses
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        SEG_COUNT = 4
        ans = list()
        segments = [0] * SEG_COUNT

        def dfs(segId: int, segStart: int):
            # 如果找到了 4 段 IP 地址并且遍历完了字符串，那么就是一种答案
            if segId == SEG_COUNT:
                if segStart == len(s):
                    ipAddr = ".".join(str(seg) for seg in segments)
                    ans.append(ipAddr)
                return

            # 如果还没有找到 4 段 IP 地址就已经遍历完了字符串，那么提前回溯
            if segStart == len(s):
                return

            # 由于不能有前导零，如果当前数字为 0，那么这一段 IP 地址只能为 0
            if s[segStart] == "0":
                segments[segId] = 0
                dfs(segId + 1, segStart + 1)

            # 一般情况，枚举每一种可能性并递归
            addr = 0
            for segEnd in range(segStart, len(s)):
                addr = addr * 10 + (ord(s[segEnd]) - ord("0"))
                if 0 < addr <= 0xFF:
                    segments[segId] = addr
                    dfs(segId + 1, segEnd + 1)
                else:
                    break

        dfs(0, 0)
        return ans


# 785. Is Graph Bipartite?
# O(N+E), O(N)
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        colors = n * [-1]
        self.flag = True

        def dfs(u, color):
            for v in graph[u]:
                if colors[v] == -1:
                    colors[v] = color ^ 1
                    dfs(v, color ^ 1)
                elif colors[v] ^ 1 != color:
                    print(colors)
                    self.flag = False
                    return

        for i in range(n):
            if colors[i] == -1:
                dfs(i, 0)
        return self.flag


# # #!/bin/python
# # # -*- coding: utf8 -*-
# # import sys
# # import os
# # import re
# #
# #
# # # 请完成下面这个函数，实现题目要求的功能
# # # 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# # # ******************************开始写代码******************************
# #
# #
# # def numIslands(grids):
# #     m, n = len(grids), len(grids[0])
# #
# #     def dfs(r, c):
# #         ans = 1
# #         grids[r][c] = 0
# #         for nr, nc in [[r - 1, c], [r + 1, c], [r, c + 1], [r, c - 1]]:
# #             if 0 <= nr < m and 0 <= nc < n and grids[nr][nc] == 1:
# #                 ans += dfs(nr, nc)
# #         # grids[r][c] = 1
# #         return ans
# #
# #     vis = set()
# #     cnt = 0
# #     for i in range(m):
# #         for j in range(n):
# #             if grids[i][j]==1:
# #                 v = dfs(i, j)
# #                 if v not in vis:
# #                     vis.add(v)
# #                     cnt+=1
# #     return cnt
# #
# #
# # # ******************************结束写代码******************************
# #
# #
# # _grids_rows = 0
# # _grids_cols = 0
# # _grids_rows = int(input())
# # _grids_cols = int(input())
# #
# # _grids = []
# # for _grids_i in range(_grids_rows):
# #     _grids_temp = list(map(int, re.split(r'\s+', input().strip())))
# #     _grids.append(_grids_temp)
# # # print(_grids)
# # res = numIslands(_grids)
# #
# # print(str(res) + "\n")
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
# # def findAllConcatenated(words):
# #     res = []
# #     vis = set()
# #     words.sort(key=len)
# #     minlen = max(1,len(words[0]))
# #     def check(word):
# #         if word in vis:
# #             return True
# #         for i in range(minlen, len(word)-minlen+1):
# #             if word[:i] in vis and check(word[i:]):
# #                 return True
# #         return False
# #
# #
# #     for word in words:
# #         if check(word):
# #             res.append(word)
# #         vis.add(word)
# #     return res
#
# def findAllConcatenated(words):
#     trie = {}
#     for word in words:
#         if not word:
#             continue
#         cur = trie
#         for ch in word:
#             cur = cur.setdefault(ch, {})
#         cur['$'] = '$'
#     res = []
#     # words.sort()
#     def dfs(word, idx, cnt, cur):
#         if idx == len(word):
#             if cnt>=1 and '$' in cur:
#                 return True
#             return False
#         if '$' in cur:
#             if dfs(word, idx, cnt+1, trie):
#                 return True
#         if word[idx] not in cur:
#             return False
#         if dfs(word, idx+1, cnt, cur[word[idx]]):
#             return True
#         return False
#     for word in words:
#         if dfs(word, 0, 0, trie):
#             res.append(word)
#     return res
#
#
#
# # ******************************结束写代码******************************
#
#
# _words_cnt = 0
# _words_cnt = int(input())
# _words_i = 0
# _words = []
# while _words_i < _words_cnt:
#     try:
#         _words_item = input()
#     except:
#         _words_item = None
#     _words.append(_words_item)
#     _words_i += 1
#
# res = findAllConcatenated(_words)
# res.sort()
#
# for res_cur in res:
#     print(str(res_cur))

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


def findLeaves(root):

    parent = {}
    res=[]
    while root:
        tmp=[]
        q = [root]
        while q:
            c = q.pop(0)
            if c.left:
                parent[c.left]=c
                q.append(c.left)
            if c.right:
                parent[c.right]=c
                q.append(c.right)
            if c.left==None and c.right==None:
                tmp.append(c.val)
                if c==root:
                    root=None
                    break
                fa = parent[c]
                if fa.left==c:
                    fa.left = None
                if fa.right==c:
                    fa.right = None
                c = None
        res.append(tmp)
    return res




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


root = get_input(input()[1:-1])

res = findLeaves(root)

print("[{}]".format(",".join(["[" + ",".join([str(item) for item in sorted(line)]) + "]" for line in res])))