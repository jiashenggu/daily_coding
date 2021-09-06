# # 133 cloneGraph
# """
# # Definition for a Node.
# class Node:
#     def __init__(self, val = 0, neighbors = None):
#         self.val = val
#         self.neighbors = neighbors if neighbors is not None else []
# """
#
#
# class Solution:
#     def __init__(self) -> None:
#         # use cache to store the cloned node
#         self.cacheNode = {}
#
#     def cloneGraph(self, node: 'Node') -> 'Node':
#         if node == None:
#             return None
#         # initialize the cloned node and clone the graph recursively
#         # if node not in self.cacheNode:
#         #     newnode = Node(node.val)
#         #     self.cacheNode[node] = newnode
#         #     for neighbor in node.neighbors:
#         #         newnode.neighbors.append(self.cloneGraph(neighbor))
#         head = Node(node.val)
#         self.cacheNode[node] = head
#         q = [node]
#         while q:
#             curnode = q.pop(0)
#             for neighbor in curnode.neighbors:
#                 if neighbor not in self.cacheNode:
#                     newneighbor = Node(neighbor.val)
#                     q.append(neighbor)
#                     self.cacheNode[neighbor] = newneighbor
#                 self.cacheNode[curnode].neighbors.append(self.cacheNode[neighbor])
#
#         return self.cacheNode[node]
# # 269. Alien Dictionary
# class Solution:
#     def alienOrder(self, words: List[str]) -> str:
#         m = {}
#         n = len(words)
#         q = collections.deque()
#         for word in words:
#             for ch in word:
#                 m[ch] = []
#         for i in range(n - 1):
#             minLen = min(len(words[i]), len(words[i + 1]))
#             j = 0
#             while j < minLen:
#                 if words[i][j] in m[words[i + 1][j]]:
#                     break
#                 if words[i][j] != words[i + 1][j]:
#                     m[words[i + 1][j]].append(words[i][j])
#                     break
#                 j += 1
#             if j == minLen and len(words[i]) > len(words[i + 1]):
#                 return ""
#
#         res = []
#         for k, v in m.items():
#             if v == []:
#                 q.append(k)
#                 res.append(k)
#
#         while q:
#             cur = q.popleft()
#             for k, v in m.items():
#                 if cur in v:
#                     m[k].remove(cur)
#                     if len(m[k]) == 0:
#                         q.append(k)
#                         res.append(k)
#         res = "".join(res)
#         if len(res) != len(m):
#             return ""
#         return res
#
#
# # 310. Minimum Height Trees
#     def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
#         if n==1:
#             return [0]
#         graph = collections.defaultdict(list)
#         degree = n*[0]
#         depth = n*[0]
#         for u, v in edges:
#             graph[u].append(v)
#             graph[v].append(u)
#             degree[u]+=1
#             degree[v]+=1
#         q = collections.deque([i for i in range(len(degree)) if degree[i]==1])
#         d = 0
#         while q:
#             d+=1
#             numQ = len(q)
#             for _ in range(numQ):
#                 cur = q.popleft()
#                 for nex in graph[cur]:
#                     # graph[cur].remove(nex)
#                     graph[nex].remove(cur)
#                     if len(graph[nex])==0:
#                         break
#                     if len(graph[nex])==1:
#                         q.append(nex)
#                     depth[nex]=d
#
#         return [i for i in range(len(depth)) if depth[i]==d-1]

#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re

class Node:
    def __init__(self, val):
        self.val = int(val)
        self.next = None
        self.random = None

#请完成下面这个函数，实现题目要求的功能
#当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
#******************************开始写代码******************************

cacheNode = {}
def  copyRandomList(head):
    if head == None:
        return head
    if head not in cacheNode:
        newNode = Node(head.val)
        cacheNode[head] = newNode
        newNode.next = copyRandomList(head.next)
        newNode.random = copyRandomList(head.random)
    return cacheNode[head]


#******************************结束写代码******************************

def construct(arr):
    idx = []
    nodes = []
    for match in re.finditer('\\[[^\\[\\]]*\\]', arr):
        node = match.group(0)[1:-1].split(',')
        nodes.append(Node(node[0]))
        idx.append(-1 if node[1].strip() == 'null' else int(node[1]))
    nodes.append(None)
    idx.append(-1)
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
        if idx[i] != -1:
            nodes[i].random = nodes[idx[i]]
    return nodes[0]

def check(origin, res):
    while origin is not None:
        if res is None or id(origin) == id(res) or origin.val != res.val or (origin.random is not None and id(origin.random) == id(res.random)):
            return False
        origin = origin.next
        res = res.next
    return True

try:
    _head = input()
except:
    _head = None

head = construct(_head)
res = copyRandomList(head)

print("1" if check(head, res) else "0" + "\n")
