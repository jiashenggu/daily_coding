# 295. Find Median from Data Stream
class MedianFinder:
    def __init__(self):
        self.l = []
        self.r = []

    def addNum(self, num: int) -> None:
        l = self.l
        r = self.r
        if not l or num<-l[0]:
            heapq.heappush(l, -num)
            while len(l)>len(r)+1:
                heapq.heappush(r, -heapq.heappop(l))
        else:
            heapq.heappush(r, num)
            while len(l)<len(r):
                heapq.heappush(l, -heapq.heappop(r))
    def findMedian(self) -> float:
        l = self.l
        r = self.r
        if len(l)>len(r):
            return -l[0]
        return (-l[0]+r[0])/2



# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
# 786. K-th Smallest Prime Fraction
class Frac:
    def __init__(self, idx: int, idy: int, x: int, y: int) -> None:
        self.idx = idx
        self.idy = idy
        self.x = x
        self.y = y

    def __lt__(self, other: "Frac") -> bool:
        return self.x * other.y < self.y * other.x


class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        n = len(arr)
        q = [Frac(0, i, arr[0], arr[i]) for i in range(1, n)]
        heapq.heapify(q)

        for _ in range(k - 1):
            frac = heapq.heappop(q)
            i, j = frac.idx, frac.idy
            if i + 1 < j:
                heapq.heappush(q, Frac(i + 1, j, arr[i + 1], arr[j]))

        return [q[0].x, q[0].y]


# 973. K Closest Points to Origin
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        h = []
        for x, y in points:
            d = x ** 2 + y ** 2
            if len(h) == k:
                if d < -h[0][0]:
                    heapq.heappop(h)
                    heapq.heappush(h, (-d, [x, y]))
            else:
                heapq.heappush(h, (-d, [x, y]))
        ans = [p[1] for p in h]
        return ans


# 253. Meeting Rooms II
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        h = []
        intervals.sort(key = lambda x: x[0])
        for i, (s, e) in enumerate(intervals):
            if h and h[0]<=s:
                heapq.heappop(h)
            heapq.heappush(h, e)
        return len(h)



# #!/bin/python
# # -*- coding: utf8 -*-
# import sys
# import os
# import re
# import collections
# #请完成下面这个函数，实现题目要求的功能
# #******************************开始写代码******************************
#
#
# def  isMaxHeap(nums):
#     q = collections.deque([0])
#     n = len(nums)
#     while q:
#         cur  = q.popleft()
#         if cur*2+1<n:
#             q.append(cur*2+1)
#             if nums[cur]<nums[cur*2+1]:
#                 return False
#         if cur*2+2<n:
#             q.append(cur*2+2)
#             if nums[cur]<nums[cur*2+2]:
#                 return False
#     return True
#
#
# #******************************结束写代码******************************
#
# arr = input()
# nums = [int(item.group(0)) for item in re.finditer(r'-?\d+', arr)]
#
# res = isMaxHeap(nums)
#
# print('true' if res else 'false')


23. Merge k Sorted Lists
binary search  and  merge
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: List[ListNode], list2: List[ListNode]) -> ListNode:
        if list1 == None:
            return list2
        if list2 == None:
            return list1

        dummy = ListNode(-1)
        cur = dummy
        while list1 and list2:
            if list1.val<list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        cur.next = list1 or list2
        return dummy.next
        # if list1.val < list2.val:
        #     list1.next = self.mergeTwoLists(list1.next, list2)
        #     return list1
        # else:
        #     list2.next = self.mergeTwoLists(list1, list2.next)
        #     return list2
    def mergeLists(self, lists: List[ListNode], l: int, r:int) -> ListNode:
        if l == r:
            return lists[l]
        mid = l + (r-l)//2
        list1 = self.mergeLists(lists, l, mid)
        list2 = self.mergeLists(lists, mid+1, r)
        return self.mergeTwoLists(list1, list2)
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        n = len(lists)
        if n==0:
            return None
        return self.mergeLists(lists, 0, n-1)


heap


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        h = []
        n = len(lists)
        for i, l in enumerate(lists):
            if l:
                heapq.heappush(h, (l.val, i))

        dummy = ListNode(-1)
        cur = dummy
        while h:
            p = heapq.heappop(h)
            cur.next = lists[p[1]]
            idx = p[0]
            cur = cur.next
            lists[p[1]] = cur.next
            if cur.next:
                heapq.heappush(h, (cur.next.val, p[1]))
        return dummy.next


