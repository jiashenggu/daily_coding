# 138. Copy List with Random Pointer
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""


class Solution:
    def __init__(self):
        self.vis = {}

    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return head
        if head in self.vis:
            return self.vis[head]
        # print(head.val)
        ret = Node(head.val)
        self.vis[head] = ret
        ret.next = self.copyRandomList(head.next)
        ret.random = self.copyRandomList(head.random)

        return ret


# 23. Merge k Sorted Lists
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def merge2(self, l1, l2):
        head = node = ListNode(-1)
        while l1 and l2:
            if l1.val<l2.val:
                node.next = l1
                l1 = l1.next
            else:
                node.next = l2
                l2 = l2.next
            node = node.next
        if l1:
            node.next = l1
        if l2:
            node.next = l2
        return head.next
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        interval = 1
        n = len(lists)
        if not n:
            return None
        while interval<n:
            for i in range(0, n - interval, 2*interval):
                lists[i] = self.merge2(lists[i], lists[i+interval])
            interval*=2
        return lists[0]

# 25. Reverse Nodes in k-Group
# O(n),O(1)
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def reverse(begin, k):
            pre = None
            node = begin
            while node and k:
                k -= 1
                tmp = node.next
                node.next = pre
                pre = node
                node = tmp
            return pre

        node = head
        ktail = None

        new_head = None
        while node:
            node = head
            cnt = 0
            while cnt < k and node:
                cnt += 1
                node = node.next
            if cnt == k:
                begin1 = reverse(head, k)
                if ktail:
                    ktail.next = begin1
                else:
                    new_head = begin1
                ktail = head
                head = node
        if ktail:
            ktail.next = head
        return new_head if new_head else head

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
