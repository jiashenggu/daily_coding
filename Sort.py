# 315. Count of Smaller Numbers After Self
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        # implement segment tree
        def update(index, value, tree, size):
            index += size  # shift the index to the leaf
            # update from leaf to root
            tree[index] += value
            while index > 1:
                index //= 2
                tree[index] = tree[index * 2] + tree[index * 2 + 1]

        def query(left, right, tree, size):
            # return sum of [left, right)
            result = 0
            left += size  # shift the index to the leaf
            right += size
            while left < right:
                # if left is a right node
                # bring the value and move to parent's right node
                if left % 2 == 1:
                    result += tree[left]
                    left += 1
                # else directly move to parent
                left //= 2
                # if right is a right node
                # bring the value of the left node and move to parent
                if right % 2 == 1:
                    right -= 1
                    result += tree[right]
                # else directly move to parent
                right //= 2
            return result

        offset = 10**4  # offset negative to non-negative
        size = 2 * 10**4 + 1  # total possible values in nums
        tree = [0] * (2 * size)
        result = []
        for num in reversed(nums):
            smaller_count = query(0, num + offset, tree, size)
            result.append(smaller_count)
            update(num + offset, 1, tree, size)
        return reversed(result)
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        # implement Binary Index Tree
        def update(index, value, tree, size):
            index += 1  # index in BIT is 1 more than the original index
            while index < size:
                tree[index] += value
                index += index & -index

        def query(index, tree):
            # return sum of [0, index)
            result = 0
            while index >= 1:
                result += tree[index]
                index -= index & -index
            return result

        offset = 10**4  # offset negative to non-negative
        size = 2 * 10**4 + 2  # total possible values in nums plus one dummy
        tree = [0] * size
        result = []
        for num in reversed(nums):
            smaller_count = query(num + offset, tree)
            result.append(smaller_count)
            update(num + offset, 1, tree, size)
        return reversed(result)
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        n = len(nums)
        arr = [[v, i] for i, v in enumerate(nums)]  # record value and index
        result = [0] * n

        def merge_sort(arr, left, right):
            # merge sort [left, right) from small to large, in place
            if right - left <= 1:
                return
            mid = (left + right) // 2
            merge_sort(arr, left, mid)
            merge_sort(arr, mid, right)
            merge(arr, left, right, mid)

        def merge(arr, left, right, mid):
            # merge [left, mid) and [mid, right)
            i = left  # current index for the left array
            j = mid  # current index for the right array
            # use temp to temporarily store sorted array
            temp = []
            while i < mid and j < right:
                if arr[i][0] <= arr[j][0]:
                    # j - mid numbers jump to the left side of arr[i]
                    result[arr[i][1]] += j - mid
                    temp.append(arr[i])
                    i += 1
                else:
                    temp.append(arr[j])
                    j += 1
            # when one of the subarrays is empty
            while i < mid:
                # j - mid numbers jump to the left side of arr[i]
                result[arr[i][1]] += j - mid
                temp.append(arr[i])
                i += 1
            while j < right:
                temp.append(arr[j])
                j += 1
            # restore from temp
            for i in range(left, right):
                arr[i] = temp[i - left]

        merge_sort(arr, 0, n)

        return result
# 1465. Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts
class Solution:
    def maxArea(self, h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]) -> int:
        # Start by sorting the inputs
        horizontalCuts.sort()
        verticalCuts.sort()

        # Consider the edges first
        max_height = max(horizontalCuts[0], h - horizontalCuts[-1])
        for i in range(1, len(horizontalCuts)):
            # horizontalCuts[i] - horizontalCuts[i - 1] represents the distance between
            # two adjacent edges, and thus a possible height
            max_height = max(max_height, horizontalCuts[i] - horizontalCuts[i - 1])

        # Consider the edges first
        max_width = max(verticalCuts[0], w - verticalCuts[-1])
        for i in range(1, len(verticalCuts)):
            # verticalCuts[i] - verticalCuts[i - 1] represents the distance between
            # two adjacent edges, and thus a possible width
            max_width = max(max_width, verticalCuts[i] - verticalCuts[i - 1])

        # Python doesn't need to worry about overflow - don't forget the modulo though!
        return (max_height * max_width) % (10 ** 9 + 7)
# 1710. Maximum Units on a Truck
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        boxTypes.sort(key = lambda x: -x[1])
        ans = 0
        i = 0
        while i<len(boxTypes) and truckSize>0:
            ans += min(truckSize, boxTypes[i][0])*boxTypes[i][1]
            truckSize -= boxTypes[i][0]
            i+=1
        return ans
# 462. Minimum Moves to Equal Array Elements II
class Solution:
    def minMoves2(self, nums: List[int]) -> int:

        n = len(nums)
        if n == 1:
            return 0
        nums.sort()

        def check(mid):
            ret = 0
            for num in nums:
                ret += abs(num - mid)
            return ret

        return check(nums[n // 2])


class Solution:
    def partition(self, nums, l, r):
        pivot = nums[r]
        i = l
        for j in range(l, r + 1):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i

    def select(self, nums, l, r, k):
        if l == r:
            return nums[l]
        pivot_index = self.partition(nums, l, r)
        if k == pivot_index:
            return nums[k]
        elif k < pivot_index:
            return self.select(nums, l, pivot_index - 1, k)
        else:
            return self.select(nums, pivot_index + 1, r, k)

    def minMoves2(self, nums: List[int]) -> int:
        s = 0
        median = self.select(nums, 0, len(nums) - 1, len(nums) // 2)

        for num in nums:
            s += abs(median - num)
        return s


class Solution:
    def partition(self, nums, l, r, val):
        idx = l
        while idx < r:
            if nums[idx] == val:
                break
            idx += 1
        nums[idx], nums[r] = nums[r], nums[idx]
        pivot = nums[r]
        i = l
        for j in range(l, r + 1):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i

    def findMedian(self, nums, l, n):
        nums[l:l + n] = sorted(nums[l: l + n])
        return nums[l + n // 2]

    def kthSmallest(self, nums, l, r, k):
        if k > 0 and k <= r - l + 1:
            n = r - l + 1
            median = (n + 4) // 5 * [0]
            i = 0
            while i < n // 5:
                median[i] = self.findMedian(nums, l + i * 5, 5)
                i += 1
            if i * 5 < n:
                median[i] = self.findMedian(nums, l + i * 5, n % 5)
                i += 1
            medOfMed = median[i - 1] if i == 1 else self.kthSmallest(median, 0, i - 1, i // 2)

            pos = self.partition(nums, l, r, medOfMed)
            if pos - l == k - 1:
                return nums[pos]
            if pos - l > k - 1:
                return self.kthSmallest(nums, l, pos - 1, k)
            return self.kthSmallest(nums, pos + 1, r, k - pos + l - 1)
        return float("inf")

    def minMoves2(self, nums: List[int]) -> int:
        s = 0
        median = self.kthSmallest(nums, 0, len(nums) - 1, len(nums) // 2 + 1)

        for num in nums:
            s += abs(median - num)
        return s


# 215. Kth Largest Element in an Array
class Solution:
    def partition(self, nums, l, r):
        pivot = random.randint(l, r)
        nums[pivot], nums[r] = nums[r], nums[pivot]
        i = l
        for j in range(l, r):
            if nums[j] >= nums[r]:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i

    def quick(self, nums, l, r, k):
        pos = self.partition(nums, l, r)
        rank = pos - l + 1
        if rank < k:
            self.quick(nums, pos + 1, r, k - rank)
        elif rank > k:
            self.quick(nums, l, pos - 1, k)

    def findKthLargest(self, nums: List[int], k: int) -> int:
        self.quick(nums, 0, len(nums) - 1, k)
        return nums[k - 1]


# 953. Verifying an Alien Dictionary
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        orderMap = {}
        for i, c in enumerate(order):
            orderMap[c] = i

        for i in range(len(words) - 1):
            for j in range(len(words[i])):
                if j >= len(words[i + 1]):
                    return False
                if words[i][j] != words[i + 1][j]:
                    if orderMap[words[i][j]] > orderMap[words[i + 1][j]]:
                        return False
                    break
        return True


# 347. Top K Frequent Elements
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        if k == len(nums):
            return nums
        c = Counter(nums)
        unique = list(c.keys())
        n = len(unique)

        def partition(l, r):
            pivot = random.randint(l, r)
            pivotF = c[unique[pivot]]
            unique[r], unique[pivot] = unique[pivot], unique[r]

            i = l
            for j in range(l, r):
                if c[unique[j]] < pivotF:
                    unique[i], unique[j] = unique[j], unique[i]
                    i += 1
            unique[r], unique[i] = unique[i], unique[r]
            return i

        def quick(l, r, k):
            if l == r:
                return
            pos = partition(l, r)
            if k == pos:
                return
            elif k > pos:
                quick(pos + 1, r, k)
            else:
                quick(l, pos - 1, k)

        quick(0, n - 1, n - k)
        return unique[n - k:]


# 973. K Closest Points to Origin
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        return self.quick_select(points, k)

    def quick_select(self, points: List[List[int]], k: int) -> List[List[int]]:
        """Perform the QuickSelect algorithm on the list"""
        left, right = 0, len(points) - 1
        pivot_index = len(points)
        while pivot_index != k:
            # Repeatedly partition the list
            # while narrowing in on the kth element
            pivot_index = self.partition(points, left, right)
            if pivot_index < k:
                left = pivot_index
            else:
                right = pivot_index - 1

        # Return the first k elements of the partially sorted list
        return points[:k]

    def partition(self, points: List[List[int]], left: int, right: int) -> int:
        """Partition the list around the pivot value"""
        pivot = self.choose_pivot(points, left, right)
        pivot_dist = self.squared_distance(pivot)
        while left < right:
            # Iterate through the range and swap elements to make sure
            # that all points closer than the pivot are to the left
            if self.squared_distance(points[left]) >= pivot_dist:
                points[left], points[right] = points[right], points[left]
                right -= 1
            else:
                left += 1

        # Ensure the left pointer is just past the end of
        # the left range then return it as the new pivotIndex
        if self.squared_distance(points[left]) < pivot_dist:
            left += 1
        return left

    def choose_pivot(self, points: List[List[int]], left: int, right: int) -> List[int]:
        """Choose a pivot element of the list"""
        return points[left + (right - left) // 2]

    def squared_distance(self, point: List[int]) -> int:
        """Calculate and return the squared Euclidean distance."""
        return point[0] ** 2 + point[1] ** 2

# different quick_select,  use while pivot_index!=k
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
class Solution:
    def partition(self, nums, l, r):
        pivot = random.randint(l, r)
        nums[r], nums[pivot] = nums[pivot], nums[r]

        i = l
        for j in range(l, r):
            if nums[j] <= nums[r]:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i

    def randomized_selected(self, arr, l, r, k):
        pos = self.partition(arr, l, r)
        if k < pos:
            self.randomized_selected(arr, l, pos - 1, k)
        elif k > pos:
            self.randomized_selected(arr, pos + 1, r, k)

    def smallestK(self, arr: List[int], k: int) -> List[int]:
        if k == 0:
            return list()
        self.randomized_selected(arr, 0, len(arr) - 1, k)
        return arr[:k]