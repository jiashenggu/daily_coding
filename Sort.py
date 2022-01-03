# 973. K Closest Points to Origin
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        return self.quick_select(points, k)

    def quick_select(self, points, k):
        left, right = 0, len(points) - 1
        pivot_index = len(points)
        while pivot_index != k:
            pivot_index = self.partition(points, left, right)
            if pivot_index < k:
                left = pivot_index
            else:
                right = pivot_index - 1
        return points[:k]

    def partition(self, points, left, right):
        pivot = self.choose_pivot(points, left, right)
        pivot_dist = self.squared_distance(pivot)
        # while left<right:
        #     if self.squared_distance(points[left]) < pivot_dist:
        #         left+=1
        #     else:
        #         points[left], points[right] = points[right], points[left]
        #         right-=1
        # if self.squared_distance(points[left])<pivot_dist:
        #     left+=1
        # return left
        i = left
        for j in range(left, right + 1):
            if self.squared_distance(points[j]) < pivot_dist:
                points[i], points[j] = points[j], points[i]
                i += 1
        return i

    def choose_pivot(self, points: List[List[int]], left: int, right: int) -> List[int]:
        """Choose a pivot element of the list"""
        # return points[left + (right - left) // 2]
        return points[random.randint(left, right)]

    def squared_distance(self, point):
        return point[0] ** 2 + point[1] ** 2


# 148. Sort List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        mid = self.getMid(head)
        left = self.sortList(head)
        right = self.sortList(mid)
        return self.merge(left, right)

    def merge(self, l1, l2):
        dummyHead = ListNode()
        node = dummyHead
        while l1 and l2:
            if l1.val < l2.val:
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

        return dummyHead.next

    def getMid(self, head):
        midPrev = None
        while head and head.next:
            midPrev = head if not midPrev else midPrev.next
            head = head.next.next
        mid = midPrev.next
        midPrev.next = None
        return mid

#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re
import queue
import heapq
import collections
#请完成下面这个函数，实现题目要求的功能
#******************************开始写代码******************************

def  sortFrequency(nums):
    cnt = collections.defaultdict(int)
    for num in nums:
        cnt[num]+=1
    numsc = nums.copy()
    nums.sort(key = lambda x: (-cnt[x], numsc.index(x)))
    return nums


#******************************结束写代码******************************

arr = input()
nums = [int(item.group(0)) for item in re.finditer(r'-?\d+', arr)]

res = sortFrequency(nums)

print('[{}]'.format(','.join(str(num) for num in res)))

# # 502. IPO
# class Solution:
#     def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
#         if min(capital)>w:
#             return 0
#         if w>=max(capital):
#             return sum(heapq.nlargest(k, profits))+w
#         projects = []
#         n = len(profits)
#         for i in range(n):
#             projects.append([profits[i], capital[i]])
#         projects.sort(key = lambda x: (x[1]))
#
#         pq = []
#         idx = 0
#         while k:
#             while idx<n and projects[idx][1]<=w:
#                 heapq.heappush(pq, -projects[idx][0])
#                 idx+=1
#             if pq:
#                 w-=pq[0]
#                 heapq.heappop(pq)
#             else:
#                 break
#             k-=1
#         return w
#
#
#
# # 179. Largest Number
# class Solution(object):
#     def largestNumber(self, nums):
#         nums_str = map(str, nums)
#         compare = lambda x, y: 1 if x + y < y + x else -1
#         nums_str.sort(cmp=compare)
#         res = "".join(nums_str)
#         if res[0] == "0":
#             res = "0"
#         return res
#
# # 296. Best Meeting Point
# class Solution:
#     def minTotalDistance(self, grid: List[List[int]]) -> int:
#         posX = []
#         posY = []
#         m, n = len(grid), len(grid[0])
#
#         for i in range(m):
#             for j in range(n):
#                 if grid[i][j] == 1:
#                     posX.append(i)
#                     posY.append(j)
#
#         posX.sort()
#         posY.sort()
#
#         l, r = 0, len(posX) - 1
#         ans = 0
#         while l < r:
#             ans += -posX[l] + posX[r]
#             ans += -posY[l] + posY[r]
#             l += 1
#             r -= 1
#         return ans
#
# # 56. Merge Intervals
# class Solution:
#     def merge(self, intervals: List[List[int]]) -> List[List[int]]:
#         intervals.sort()
#
#         merged = []
#         for interval in intervals:
#             # 如果列表为空，或者当前区间与上一区间不重合，直接添加
#             if not merged or merged[-1][1] < interval[0]:
#                 merged.append(interval)
#             else:
#                 # 否则的话，我们就可以与上一区间进行合并
#                 merged[-1][1] = max(merged[-1][1], interval[1])
#
#         return merged
#
# # 164. Maximum Gap
# # radix sort
# class Solution:
#     def maximumGap(self, nums: List[int]) -> int:
#         n = len(nums)
#         if n < 2:
#             return 0
#
#         exp = 1
#         buf = n * [0]
#         maxVal = max(nums)
#
#         while maxVal >= exp:
#             cnt = 10 * [0]
#             for i in range(n):
#                 digit = nums[i] // exp % 10
#                 cnt[digit] += 1
#             for i in range(1, 10):
#                 cnt[i] += cnt[i - 1]
#
#             for i in range(n - 1, -1, -1):
#                 digit = nums[i] // exp % 10
#                 buf[cnt[digit] - 1] = nums[i]
#                 cnt[digit] -= 1
#             nums = buf.copy()
#             exp *= 10
#         ans = 0
#         for i in range(1, n):
#             ans = max(ans, abs(nums[i] - nums[i - 1]))
#         return ans
#
# # bucket sort
# class Solution:
#     def maximumGap(self, nums: List[int]) -> int:
#         ma = max(nums)
#         mi = min(nums)
#         n = len(nums)
#         if n<2 or ma==mi:
#             return 0
#         # use the average of all numbers as bucket
#         d = ceil((ma-mi)/(n-1))
#         da = n*[0]
#         di = n*[sys.maxsize]
#         ans=0
#         # t decide which bucket to save the number
#         # maintain the max and min in the bucket
#         for i,num in enumerate(nums):
#             t = (num-mi)//d
#             da[t]=max(da[t],num)
#             di[t]=min(di[t],num)
#         tmp=0
#         for i in range(1,n):
#             if di[i]==sys.maxsize:
#                 continue
#             ans=max(di[i]-da[tmp],ans)
#             tmp=i
#         return ans
#
# # 面试题 17.14. Smallest K LCCI
# # quick sort idea partition
# class Solution:
#     def partition(self, nums, l, r):
#         pivot = nums[r]
#         i = l
#         for j in range(l, r):
#             if nums[j] <= pivot:
#                 nums[i], nums[j] = nums[j], nums[i]
#                 i += 1
#         nums[i], nums[r] = nums[r], nums[i]
#         return i
#
#     def randomized_partition(self, nums, l, r):
#         i = random.randint(l, r)
#         nums[r], nums[i] = nums[i], nums[r]
#         return self.partition(nums, l, r)
#
#     def randomized_selected(self, arr, l, r, k):
#         pos = self.randomized_partition(arr, l, r)
#         num = pos - l + 1
#         if k < num:
#             self.randomized_selected(arr, l, pos - 1, k)
#         elif k > num:
#             self.randomized_selected(arr, pos + 1, r, k - num)
#
#     def smallestK(self, arr: List[int], k: int) -> List[int]:
#         if k == 0:
#             return list()
#         self.randomized_selected(arr, 0, len(arr) - 1, k)
#         return arr[:k]