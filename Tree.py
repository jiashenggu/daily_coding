# 510. Inorder Successor in BST II
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""


class Solution:
    def inorderSuccessor(self, node: 'Node') -> 'Optional[Node]':

        if not node.right:
            while node.parent and node.parent.right == node:
                node = node.parent
            return node.parent
        node = node.right
        while node.left:
            node = node.left
        return node


# 987. Vertical Order Traversal of a Binary Tree
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        q = deque([(root, 0, 0)])
        nodes = []
        while q:
            cur, row, col = q.popleft()
            nodes.append((col, row, cur.val))
            if cur.left:
                q.append((cur.left, row+1, col-1))
            if cur.right:
                q.append((cur.right, row+1, col+1))
        nodes.sort()
        ans = defaultdict(list)
        for col, row, val in nodes:
            ans[col].append(val)
        return list(ans.values())

# 440. K-th Smallest in Lexicographical Order
class Solution:
    def count(self, c, n):
        c1 = c
        c2 = c+1
        steps = 0
        while c1<=n:
            steps+=min(c2, n+1)-c1
            c1*=10
            c2*=10
        return steps
    def findKthNumber(self, n: int, k: int) -> int:
        k-=1
        cur = 1
        while k>0:
            cnt = self.count(cur, n)
            if cnt<=k:
                k-=cnt
                cur+=1
            else:
                k-=1
                cur*=10
        return cur
# 297. Serialize and Deserialize Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return "None,"
        s = ""
        s += str(root.val) + ','
        s += self.serialize(root.left)
        s += self.serialize(root.right)
        print(s)
        return s

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        data_list = data.split(',')
        # print(data_list)
        return self.dfs(data_list)

    def dfs(self, l):
        if not l:
            return None
        val = l.pop(0)
        if val == 'None':
            return None
        root = TreeNode(val)
        root.left = self.dfs(l)
        root.right = self.dfs(l)
        return root


# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

# 450. Delete Node in a BST
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def successor(self, root):
        root = root.right
        while root.left:
            root = root.left
        return root.val

    def predecessor(self, root):
        root = root.left
        while root.right:
            root = root.right
        return root.val

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        else:
            if not root.left and not root.right:
                return None
            elif root.left:
                root.val = self.predecessor(root)
                root.left = self.deleteNode(root.left, root.val)
            elif root.right:
                root.val = self.successor(root)
                root.right = self.deleteNode(root.right, root.val)

        return root


# 230. Kth Smallest Element in a BST
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        s = []
        while True:
            while root:
                s.append(root)
                root = root.left
            root = s.pop()
            k-=1
            if not k:
                return root.val
            root = root.right

# cache dfs
class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        edges = defaultdict(list)
        for fr, to in relations:
            edges[fr - 1].append(to - 1)

        @lru_cache(None)
        def dfs(u):
            cur = 0
            for v in edges[u]:
                cur = max(cur, dfs(v))
            return cur + time[u]

        ans = 0
        for i in range(n):
            ans = max(ans, dfs(i))
        return ans


# topology sort
class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        graph = defaultdict(list)
        inDegree = [0] * n
        for prv, nxt in relations:
            prv, nxt = prv - 1, nxt - 1  # convert into zero-based index
            graph[prv].append(nxt)
            inDegree[nxt] += 1

        q = deque([])
        dist = [0] * n
        for u in range(n):
            if inDegree[u] == 0:
                q.append(u)
                dist[u] = time[u]

        while q:
            u = q.popleft()
            for v in graph[u]:
                dist[v] = max(dist[u] + time[v], dist[v])  # Update `dist[v]` using the maximum dist of the predecessor nodes
                inDegree[v] -= 1
                if inDegree[v] == 0:
                    q.append(v)
        return max(dist)
# 2049. Count Nodes With the Highest Score
class Solution:
    def countHighestScoreNodes(self, parents: List[int]) -> int:
        self.best = -1
        self.ans = 0
        edges = defaultdict(list)
        for i, parent in enumerate(parents):
            edges[parent].append(i)
        n = len(parents)
        sz = n * [1]

        def dfs(x):
            cur = 1
            for i in edges[x]:
                dfs(i)
                sz[x] += sz[i]
                cur *= sz[i]
            if x:
                cur *= n - sz[x]
            if cur > self.best:
                self.best = cur
                self.ans = 1
            elif cur == self.best:
                self.ans += 1

        dfs(0)
        return self.ans
# 222. Count Complete Tree Nodes
class Solution:
    def compute_depth(self, node: TreeNode) -> int:
        """
        Return tree depth in O(d) time.
        """
        d = 0
        while node.left:
            node = node.left
            d += 1
        return d

    def exists(self, idx: int, d: int, node: TreeNode) -> bool:
        """
        Last level nodes are enumerated from 0 to 2**d - 1 (left -> right).
        Return True if last level node idx exists.
        Binary search with O(d) complexity.
        """
        left, right = 0, 2 ** d - 1
        for _ in range(d):
            pivot = left + (right - left) // 2
            if idx <= pivot:
                node = node.left
                right = pivot
            else:
                node = node.right
                left = pivot + 1
        return node is not None

    def countNodes(self, root: TreeNode) -> int:
        # if the tree is empty
        if not root:
            return 0

        d = self.compute_depth(root)
        # if the tree contains 1 node
        if d == 0:
            return 1

        # Last level nodes are enumerated from 0 to 2**d - 1 (left -> right).
        # Perform binary search to check how many nodes exist.
        left, right = 1, 2 ** d - 1
        while left <= right:
            pivot = left + (right - left) // 2
            if self.exists(pivot, d, root):
                left = pivot + 1
            else:
                right = pivot - 1

        # The tree contains 2**d - 1 nodes on the first (d - 1) levels
        # and left nodes on the last level.
        return (2 ** d - 1) + left
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