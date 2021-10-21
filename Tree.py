# 437. Path Sum III
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        prefix = collections.defaultdict(int)
        prefix[0] = 1

        def dfs(root, curr):
            if not root:
                return 0

            ret = 0
            curr += root.val
            ret += prefix[curr - targetSum]
            prefix[curr] += 1
            ret += dfs(root.left, curr)
            ret += dfs(root.right, curr)
            prefix[curr] -= 1

            return ret

        return dfs(root, 0)

# 1008. Construct Binary Search Tree from Preorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        def helper(lower=float("-inf"), upper=float("inf")):
            nonlocal idx
            if idx == n:
                return None
            val = preorder[idx]
            if val < lower or val > upper:
                return None
            idx += 1
            root = TreeNode(val)
            root.left = helper(lower, val)
            root.right = helper(val, upper)
            return root

        idx = 0
        n = len(preorder)
        return helper()
# 428. Serialize and Deserialize N-ary Tree
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""


class Codec:
    def serialize(self, root: 'Node') -> str:
        """Encodes a tree to a single string.

        :type root: Node
        :rtype: str
        """
        if not root:
            return '#'
        data = ""
        data += str(root.val) + '-' + str(len(root.children))
        for child in root.children:
            data += '-' + self.serialize(child)
        return data

    def deserialize(self, data: str) -> 'Node':
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: Node
        """
        if data == '#':
            return None
        data = data.split('-')
        return self.dfs(data)

    def dfs(self, data):
        root = Node(int(data.pop(0)))
        root.children = []
        n = int(data.pop(0))
        for i in range(n):
            child = self.dfs(data)
            root.children.append(child)
        return root


#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re


class TreeNode:
    def __init__(self, data):
        self.val = data
        self.left = None
        self.right = None


def str2tree(s):
    if s=="":
        return None
    position = s.find('(')
    value = 0
    if position != -1:
        value = int(s[:position])
    else:
        value = int(s)

    cur = TreeNode(value)
    if position == -1:
        return cur
    count = 0
    start = position
    for i in range(start, len(s)):
        if s[i] == '(':
            count += 1
        elif s[i] == ')':
            count -= 1
        if count == 0 and start == position:
            cur.left = str2tree(s[start + 1:i])
            start = i + 1
        elif count == 0:
            cur.right = str2tree(s[start + 1: i])
    return cur


# 请完成下面这个函数，实现题目要求的功能
# 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# ******************************开始写代码******************************
# Remember to see the definition of the tree node above

def findMax(root):
    if root == None:
        return -sys.maxsize
    if root.left==None and None==root.right:
        return root.val
    ml = findMax(root.left)
    mr = findMax(root.right)

    return max(ml, mr, root.val)


# ******************************结束写代码******************************
s = input()

print(s)
try:
    _treenode = str2tree(s)
except:
    _treenode = None

res = findMax(_treenode)

print(str(res) + "\n")

#
# # !/bin/python
# # -*- coding: utf8 -*-
# import sys
# import os
# import re
#
# class TreeNode:
#     def __init__(self, val):
#         self.val = val
#         self.left = None
#         self.right = None
#
#
# def deserialize(tree):
#     tree = tree.strip('[]')
#     if len(tree) == 0:
#         return None
#     nodes = [None if item == 'null' else TreeNode(int(item))
#              for item in tree.split(',')]
#     kids = list(reversed(nodes))
#     root = kids.pop()
#     for node in nodes:
#         if node:
#             if kids: node.left = kids.pop()
#             if kids: node.right = kids.pop()
#     return root
#
#
# # 请完成下面这个函数，实现题目要求的功能
# # 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# # ******************************开始写代码******************************
#
#
# def  rangeSumBST(root, L, R):
#     if root==None:
#         return 0
#
#     l = rangeSumBST(root.left, L, R) if root.val>=L else 0
#     r = rangeSumBST(root.right, L, R) if root.val<=R else 0
#     v = root.val if L<=root.val<=R else 0
#     return l + r + v
#
#
# # ******************************结束写代码******************************
#
#
# try:
#     _tree = input()
# except:
#     _root = None
#
# _L = int(input())
#
# _R = int(input())
#
# _root = deserialize(_tree)
# res = rangeSumBST(_root, _L, _R)
#
# print(str(res) + "\n")

#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

#请完成下面这个函数，实现题目要求的功能
#当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
#******************************开始写代码******************************

def delete(node):
    if node==None:
        return
    if node.left==None and node.right==None:
        node = None
    elif node.left==None:
        node = node.right
    elif node.right==None:
        node = node.left
    else:
        node.val = findAndRemove(node)
    return node

def findAndRemove(node):
    if node.left.right==None:
        res = node.left.val
        node.left = node.left.left
        return res
    nodeL = node.left
    if nodeL.right.right!=None:
        nodeL = nodeL.right
    res = nodeL.right.val
    nodeL.right = nodeL.right.left
    return res

def trimBST(root, L, R):

    while root and (root.val<L or root.val>R):
        root = delete(root)
    if root==None:
        return root
    root.left = trimBST(root.left, L, R)
    root.right = trimBST(root.right, L, R)
    return root

#******************************结束写代码******************************


def deserialize(tree):
    tree = tree.strip('[]')
    if len(tree) == 0:
        return None
    nodes = [None if item == 'null' else TreeNode(int(item))
             for item in tree.split(',')]
    kids = list(reversed(nodes))
    root = kids.pop()
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return root

def serialize(root):
    if root is None:
        return '[]'
    nodes = [root]
    idx = 0
    current_null = 0
    while idx < len(nodes):
        if nodes[idx]:
            if nodes[idx].left:
                nodes.extend([None] * current_null)
                current_null = 0
                nodes.append(nodes[idx].left)
            else:
                current_null += 1

            if nodes[idx].right:
                nodes.extend([None] * current_null)
                current_null = 0
                nodes.append(nodes[idx].right)
            else:
                current_null += 1
        idx += 1
    res = [str(node.val) if node else 'null' for node in nodes]
    return '[{}]'.format(','.join(res))

try:
    _tree = input()
except:
    _tree = '[]'

_L = int(input())

_R = int(input())

_root = deserialize(_tree)
res = trimBST(_root, _L, _R)

print(serialize(res) + "\n")