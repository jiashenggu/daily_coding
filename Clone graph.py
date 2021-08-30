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