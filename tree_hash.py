428. Serialize and Deserialize N-ary Tree
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