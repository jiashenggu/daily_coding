# 133 cloneGraph
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""


class Solution:
    def __init__(self) -> None:
        # use cache to store the cloned node
        self.cacheNode = {}

    def cloneGraph(self, node: 'Node') -> 'Node':
        if node == None:
            return None
        # initialize the cloned node and clone the graph recursively
        # if node not in self.cacheNode:
        #     newnode = Node(node.val)
        #     self.cacheNode[node] = newnode
        #     for neighbor in node.neighbors:
        #         newnode.neighbors.append(self.cloneGraph(neighbor))
        head = Node(node.val)
        self.cacheNode[node] = head
        q = [node]
        while q:
            curnode = q.pop(0)
            for neighbor in curnode.neighbors:
                if neighbor not in self.cacheNode:
                    newneighbor = Node(neighbor.val)
                    q.append(neighbor)
                    self.cacheNode[neighbor] = newneighbor
                self.cacheNode[curnode].neighbors.append(self.cacheNode[neighbor])

        return self.cacheNode[node]
# 269. Alien Dictionary
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        m = {}
        n = len(words)
        q = collections.deque()
        for word in words:
            for ch in word:
                m[ch] = []
        for i in range(n - 1):
            minLen = min(len(words[i]), len(words[i + 1]))
            j = 0
            while j < minLen:
                if words[i][j] in m[words[i + 1][j]]:
                    break
                if words[i][j] != words[i + 1][j]:
                    m[words[i + 1][j]].append(words[i][j])
                    break
                j += 1
            if j == minLen and len(words[i]) > len(words[i + 1]):
                return ""

        res = []
        for k, v in m.items():
            if v == []:
                q.append(k)
                res.append(k)

        while q:
            cur = q.popleft()
            for k, v in m.items():
                if cur in v:
                    m[k].remove(cur)
                    if len(m[k]) == 0:
                        q.append(k)
                        res.append(k)
        res = "".join(res)
        if len(res) != len(m):
            return ""
        return res


# 310. Minimum Height Trees
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n==1:
            return [0]
        graph = collections.defaultdict(list)
        degree = n*[0]
        depth = n*[0]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
            degree[u]+=1
            degree[v]+=1
        q = collections.deque([i for i in range(len(degree)) if degree[i]==1])
        d = 0
        while q:
            d+=1
            numQ = len(q)
            for _ in range(numQ):
                cur = q.popleft()
                for nex in graph[cur]:
                    # graph[cur].remove(nex)
                    graph[nex].remove(cur)
                    if len(graph[nex])==0:
                        break
                    if len(graph[nex])==1:
                        q.append(nex)
                    depth[nex]=d

        return [i for i in range(len(depth)) if depth[i]==d-1]