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