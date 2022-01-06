# 210. Course Schedule II
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        ind = numCourses*[0]
        graph = defaultdict(list)
        for v, u in prerequisites:
            graph[u].append(v)
            ind[v]+=1
        q = deque([i for i in range(numCourses) if ind[i]==0])
        ans = list(q)
        while q:
            # print(ind, q)
            u = q.popleft()
            for v in graph[u]:
                ind[v]-=1
                if ind[v]==0:
                    ans.append(v)
                    q.append(v)
        return ans if len(ans)==numCourses else []
# 1976. Number of Ways to Arrive at Destination
class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        dist = [[float("inf")] * n for _ in range(n)]

        for i in range(n):
            dist[i][i] = 0
        for u, v, t in roads:
            dist[u][v] = t
            dist[v][u] = t

        seen = set()
        for _ in range(n):
            u = None
            for i in range(n):
                if i not in seen and (not u or (dist[0][i] < dist[0][u])):
                    u = i
            seen.add(u)
            for i in range(n):
                dist[0][i] = min(dist[0][i], dist[0][u] + dist[u][i])

        g = defaultdict(list)
        for u, v, t in roads:
            if dist[0][u] - dist[0][v] == t:
                g[v].append(u)
            if dist[0][v] - dist[0][u] == t:
                g[u].append(v)

        @lru_cache(None)
        def dfs(u):
            if u == n - 1:
                return 1
            ans = 0
            for v in g[u]:
                ans += (dfs(v)) % mod
            return ans % mod

        ans = dfs(0)
        dfs.cache_clear()
        return ans


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
