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

def  trimBST(root, L, R):

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