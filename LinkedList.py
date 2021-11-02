# 430. Flatten a Multilevel Doubly Linked List
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""
class Solution(object):

    def flatten(self, head):
        if not head:
            return head

        # pseudo head to ensure the `prev` pointer is never none
        pseudoHead = Node(None, None, head, None)
        self.flatten_dfs(pseudoHead, head)

        # detach the pseudo head from the real head
        pseudoHead.next.prev = None
        return pseudoHead.next


    def flatten_dfs(self, prev, curr):
        """ return the tail of the flatten list """
        if not curr:
            return prev

        curr.prev = prev
        prev.next = curr

        # the curr.next would be tempered in the recursive function
        tempNext = curr.next
        tail = self.flatten_dfs(curr, curr.child)
        curr.child = None
        return self.flatten_dfs(tail, tempNext)

# My Solution
"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""


class Solution:
    def flatten(self, head: 'Node') -> 'Node':
        if not head:
            return head

        def flat(head):
            node = head
            while node:
                while node.next:
                    if node.child:
                        start, end = flat(node.child)
                        tmp = node.next
                        node.next = start
                        start.prev = node
                        tmp.prev = end
                        end.next = tmp
                        node.child = None
                        node = end
                    node = node.next
                if not node.next and node.child:
                    start, end = flat(node.child)
                    node.next = start
                    start.prev = node
                    node.child = None
                    return head, end
                return head, node

        head, node = flat(head)

        return head