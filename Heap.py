# 378. Kth Smallest Element in a Sorted Matrix
import heapq
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:

        # The size of the matrix
        N = len(matrix)

        # Preparing our min-heap
        minHeap = []
        for r in range(min(k, N)):
            # We add triplets of information for each cell
            minHeap.append((matrix[r][0], r, 0))

        # Heapify our list
        heapq.heapify(minHeap)

        # Until we find k elements
        while k:

            # Extract-Min
            element, r, c = heapq.heappop(minHeap)

            # If we have any new elements in the current row, add them
            if c < N - 1:
                heapq.heappush(minHeap, (matrix[r][c + 1], r, c + 1))

            # Decrement k
            k -= 1

        return element
    # 1696. Jump Game VI
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        n = len(nums)
        score = nums[0]
        dp = deque()
        dp.append((0, score))
        for i in range(1, n):
            while dp and dp[0][0] < i - k:
                dp.popleft()
            score = dp[0][1] + nums[i]
            while dp and dp[-1][1] <= score:
                dp.pop()
            dp.append((i, score))
        return score


# 1354. Construct Target Array With Multiple Sums
def isPossible(self, target: List[int]) -> bool:
    # Handle the n = 1 case.
    if len(target) == 1:
        return target == [1]

    total_sum = sum(target)

    target = [-num for num in target]
    heapq.heapify(target)
    while -target[0] > 1:
        largest = -target[0]
        rest = total_sum - largest

        # This will only occur if n = 2.
        if rest == 1:
            return True

        x = largest % rest

        # If x is now 0 (invalid) or didn't
        # change, then we know this is impossible.
        if x == 0 or x == largest:
            return False
        heapq.heapreplace(target, -x)
        total_sum = total_sum - largest + x

    return True
# 630. Course Schedule III
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses.sort(key=lambda x: x[1])
        time, count = 0, 0
        n = len(courses)
        q = []
        for c in courses:
            if time + c[0] <= c[1]:
                heapq.heappush(q, -c[0])
                time += c[0]
            elif len(q) > 0 and -q[0] > c[0]:
                time += c[0] - (-heapq.heappop(q))
                heapq.heappush(q, -c[0])
        return len(q)


# 1229. Meeting Scheduler
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        slots = list(filter(lambda x: x[1]-x[0]>=duration, slots1+slots2))
        heapq.heapify(slots)
        while len(slots)>1:
            start1, end1 = heapq.heappop(slots)
            start2, end2 = slots[0]
            if end1>=start2+duration:
                return [start2, start2 + duration]
        return []
# 1642. Furthest Building You Can Reach
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        ladder_allocations = [] # We'll use heapq to treat this as a min-heap.
        for i in range(len(heights) - 1):
            climb = heights[i + 1] - heights[i]
            # If this is actually a "jump down", skip it.
            if climb <= 0:
                continue
            # Otherwise, allocate a ladder for this climb.
            heapq.heappush(ladder_allocations, climb)
            # If we haven't gone over the number of ladders, nothing else to do.
            if len(ladder_allocations) <= ladders:
                continue
            # Otherwise, we will need to take a climb out of ladder_allocations
            bricks -= heapq.heappop(ladder_allocations)
            # If this caused bricks to go negative, we can't get to i + 1
            if bricks < 0:
                return i
        # If we got to here, this means we had enough to cover every climb.
        return len(heights) - 1
# 253. Meeting Rooms II
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[0])
        freeroom = []
        for interval in intervals:
            s, e = interval
            if freeroom and freeroom[0] <= s:
                heapq.heappop(freeroom)
            heapq.heappush(freeroom, e)
        return len(freeroom)


# 352. Data Stream as Disjoint Intervals
class SummaryRanges(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.intervals = []
        self.seen = set()

    def addNum(self, val):
        """
        :type val: int
        :rtype: None
        """
        if val not in self.seen:
            self.seen.add(val)
            heapq.heappush(self.intervals, [val, val])

    def getIntervals(self):
        """
        :rtype: List[List[int]]
        """

        tmp = []

        while self.intervals:
            cur = heapq.heappop(self.intervals)
            if tmp and cur[0] <= tmp[-1][1] + 1:
                tmp[-1][1] = max(tmp[-1][1], cur[1])
            else:
                tmp.append(cur)

        self.intervals = tmp
        return self.intervals
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


