# 1539. Kth Missing Positive Number
class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        l, r = 0, len(arr) - 1
        while l <= r:
            mid = (l + r) >> 1
            if arr[mid] - mid - 1 < k:
                l = mid + 1
            else:
                r = mid - 1
        return r + 1 + k


# 658. Find K Closest Elements
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        # Initialize binary search bounds
        left = 0
        right = len(arr) - k

        # Binary search against the criteria described
        while left < right:
            mid = (left + right) // 2
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            else:
                right = mid

        return arr[left:left + k]
# 1428. Leftmost Column with at Least a One
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
# class BinaryMatrix(object):
#    def get(self, row: int, col: int) -> int:
#    def dimensions(self) -> list[]:

class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        rows, cols = binaryMatrix.dimensions()

        cur_row = 0
        cur_col = cols - 1
        while cur_row < rows and cur_col >= 0:
            if binaryMatrix.get(cur_row, cur_col) == 1:
                cur_col -= 1
            else:
                cur_row += 1
        return cur_col + 1 if cur_col != cols - 1 else -1
class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        rows, cols = binaryMatrix.dimensions()
        smallest_index = cols
        for row in range(rows):
            # Binary Search for the first 1 in the row.
            lo = 0
            hi = cols - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if binaryMatrix.get(row, mid) == 0:
                    lo = mid + 1
                else:
                    hi = mid
            # If the last element in the search space is a 1, then this row
            # contained a 1.
            if binaryMatrix.get(row, lo) == 1:
                smallest_index = min(smallest_index, lo)
        # If smallest_index is still set to cols, then there were no 1's in
        # the grid.
        return -1 if smallest_index == cols else smallest_index

# 4. Median of Two Sorted Arrays
class Solution:
    def findMedianSortedArrays(self, A: List[int], B: List[int]) -> float:
        m, n = len(A), len(B)
        if m > n:
            A, B, m, n = B, A, n, m
        if n == 0:
            raise ValueError

        imin, imax, half_len = 0, m, (m + n + 1) // 2
        while imin <= imax:
            i = (imin + imax) // 2
            j = half_len - i
            if i < m and B[j-1] > A[i]:
                # i is too small, must increase it
                imin = i + 1
            elif i > 0 and A[i-1] > B[j]:
                # i is too big, must decrease it
                imax = i - 1
            else:
                # i is perfect

                if i == 0: max_of_left = B[j-1]
                elif j == 0: max_of_left = A[i-1]
                else: max_of_left = max(A[i-1], B[j-1])

                if (m + n) % 2 == 1:
                    return max_of_left

                if i == m: min_of_right = B[j]
                elif j == n: min_of_right = A[i]
                else: min_of_right = min(A[i], B[j])

                return (max_of_left + min_of_right) / 2.0

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1 = len(nums1)
        n2 = len(nums2)
        i1 = 0
        i2 = 0
        mid = (n1 + n2) // 2 + 1
        left = right = 0
        while i1 < n1 and i2 < n2 and i1 + i2 < mid:
            if nums1[i1] < nums2[i2]:
                left = right
                right = nums1[i1]
                i1 += 1
            else:
                left = right
                right = nums2[i2]
                i2 += 1
        while i1 < n1 and i1 + i2 < mid:
            left = right
            right = nums1[i1]
            i1 += 1
        while i2 < n2 and i1 + i2 < mid:
            left = right
            right = nums2[i2]
            i2 += 1
        if (n1 + n2) % 2 == 0:
            return (left + right) / 2
        return right


# 162. Find Peak Element
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        l, r = 0, len(nums)-1
        while l<r:
            mid = (l+r)>>1
            if nums[mid]>nums[mid+1]:
                r = mid
            else:
                l = mid+1
        return l
# 2111. Minimum Operations to Make the Array K-Increasing
class Solution:
    def longestNonDecreasingSubsequence(self, arr):
        sub = []
        for i, x in enumerate(arr):
            if len(sub) == 0 or sub[-1] <= x:  # Append to LIS if new element is >= last element in LIS
                sub.append(x)
            else:
                idx = bisect_right(sub, x)  # Find the index of the smallest number > x
                sub[idx] = x  # Replace that number with x
        return len(sub)

    def kIncreasing(self, arr: List[int], k: int) -> int:
        n = len(arr)
        ans = 0
        for i in range(k):
            newArr = []
            for j in range(i, n, k):
                newArr.append(arr[j])
            ans += len(newArr) - self.longestNonDecreasingSubsequence(newArr)
        return ans

# 33. Search in Rotated Sorted Array
# O(log(n)), O(1)
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        while l<=r:
            mid = (l+r)>>1
            if target==nums[mid]:
                return mid
            elif nums[mid]>=nums[0]:
                if target>=nums[0] and target<nums[mid]:
                    r = mid-1
                else:
                    l = mid+1
            else:
                if target<=nums[-1] and target>nums[mid]:
                    l = mid+1
                else:
                    r = mid-1
        return -1

# 786. K-th Smallest Prime Fraction
class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        n = len(arr)
        l, r = 0.0, 1.0

        while True:
            mid = (l + r) / 2
            i = -1
            count = 0
            x, y = 0, 1
            for j in range(n):
                while arr[i + 1] / arr[j] < mid:
                    i += 1
                    if arr[i] * y > arr[j] * x:
                        x, y = arr[i], arr[j]
                count += i + 1
            if count == k:
                return [x, y]
            elif count > k:
                r = mid
            else:
                l = mid


# 1231. Divide Chocolate
class Solution:
    def maximizeSweetness(self, sweetness: List[int], k: int) -> int:
        # Initialize the left and right boundaries.
        # left = 1 and right = (total sweetness) / (number of people).
        number_of_people = k + 1
        left = min(sweetness)
        right = sum(sweetness) // number_of_people

        while left < right:
            # Get the middle index between left and right boundary indexes.
            # cur_sweetness stands for the total sweetness for the current person.
            # people_with_chocolate stands for the number of people that have
            # a piece of chocolate of sweetness greater than or equal to mid.
            mid = (left + right + 1) // 2
            cur_sweetness = 0
            people_with_chocolate = 0

            # Start assigning chunks to the current person.
            for s in sweetness:
                cur_sweetness += s

                # If the total sweetness is no less than mid, this means we can break off
                # the current piece and move on to assigning chunks to the next person.
                if cur_sweetness >= mid:
                    people_with_chocolate += 1
                    cur_sweetness = 0

            if people_with_chocolate >= k + 1:
                left = mid
            else:
                right = mid - 1

        return right


# 1044. Longest Duplicate Substring
class Solution:
    def search(self, L: int, a: int, MOD: int, n: int, nums: List[int]) -> str:
        """
        Rabin-Karp with polynomial rolling hash.
        Search a substring of given length
        that occurs at least 2 times.
        @return start position if the substring exits and -1 otherwise.
        """
        # Compute the hash of the substring S[:L].
        h = 0
        for i in range(L):
            h = (h * a + nums[i]) % MOD

        # Store the already seen hash values for substrings of length L.
        seen = collections.defaultdict(list)
        seen[h].append(0)

        # Const value to be used often : a**L % MOD
        aL = pow(a, L, MOD)
        for start in range(1, n - L + 1):
            # Compute the rolling hash in O(1) time
            h = (h * a - nums[start - 1] * aL + nums[start + L - 1]) % MOD
            if h in seen:
                # Check if the current substring matches any of the previous substrings with hash h.
                current_substring = nums[start: start + L]
                if any(current_substring == nums[index: index + L] for index in seen[h]):
                    return start
            seen[h].append(start)
        return -1

    def longestDupSubstring(self, S: str) -> str:
        # Modulus value for the rolling hash function to avoid overflow.
        MOD = 10 ** 9 + 7

        # Select a base value for the rolling hash function.
        a = 26
        n = len(S)

        # Convert string to array of integers to implement constant time slice.
        nums = [ord(S[i]) - ord('a') for i in range(n)]

        # Use binary search to find the longest duplicate substring.
        start = -1
        left, right = 1, n - 1
        while left <= right:
            # Guess the length of the longest substring.
            L = left + (right - left) // 2
            start_of_duplicate = self.search(L, a, MOD, n, nums)

            # If a duplicate substring of length L exists, increase left and store the
            # starting index of the duplicate substring.  Otherwise decrease right.
            if start_of_duplicate != -1:
                left = L + 1
                start = start_of_duplicate
            else:
                right = L - 1

        # The longest substring (if any) begins at index start and ends at start + left.
        return S[start: start + left - 1]
# My solution(not optimal):
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        n = len(s)
        l, r = 1, n-1
        def check(mid):
            cnt = defaultdict(int)
            for i in range(n-mid+1):
                # print(s[i:i+mid])
                cnt[s[i:i+mid]]+=1
                if cnt[s[i:i+mid]]>1:
                    return True
            return False
        while l<r:
            mid = (l+r+1)>>1
            # print(l,r,mid)
            if check(mid):
                l = mid
            else:
                r = mid-1
        cnt = defaultdict(int)
        for i in range(n-l+1):
            cnt[s[i:i+l]]+=1
            if cnt[s[i:i+l]]>1:
                return s[i:i+l]
        return ""
# 154. Find Minimum in Rotated Sorted Array II
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) >> 1
            if nums[mid] > nums[r]:
                l = mid + 1
            elif nums[mid] < nums[r]:
                r = mid
            else:
                r -= 1
        return nums[l]


# 875. Koko Eating Bananas
class Solution(object):
    def minEatingSpeed(self, piles, H):
        # Can Koko eat all bananas in H hours with eating speed K?
        def possible(K):
            return sum((p-1) // K + 1 for p in piles) <= H

        lo, hi = 1, max(piles)
        while lo < hi:
            mi = (lo + hi) // 2
            if not possible(mi):
                lo = mi + 1
            else:
                hi = mi
        return lo
# Maximum of minimum difference of all pairs from subsequences of given size
# Python3 program to implement
# the above approach

# Function to check a subsequence can
# be formed with min difference mid
def can_place(A, n, B, mid):
    count = 1
    last_position = A[0]

    # If a subsequence of size B
    # with min diff = mid is possible
    # return true else false
    for i in range(1, n):
        if (A[i] - last_position >= mid):
            last_position = A[i]
            count = count + 1

            if (count == B):
                return bool(True)

    return bool(False)


# Function to find the maximum of
# all minimum difference of pairs
# possible among the subsequence
def find_min_difference(A, n, B):
    # Sort the Array
    A.sort()

    # Stores the boundaries
    # of the search space
    s = 0
    e = A[n - 1] - A[0]

    # Store the answer
    ans = 0

    # Binary Search
    while (s <= e):
        mid = (int)((s + e) / 2)

        # If subsequence can be formed
        # with min diff mid and size B
        if (can_place(A, n, B, mid)):
            ans = mid

            # Right half
            s = mid + 1

        else:

            # Left half
            e = mid - 1

    return ans


# Driver code
A = [1, 2, 3, 5]
n = len(A)
B = 3

min_difference = find_min_difference(A, n, B)

print(min_difference)

# This code is contributed by divyeshrabadiya07

# 704. Binary Search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        l, r, = 0, n-1
        while l<=r:
            mid = (l+r)>>1
            if nums[mid]==target:
                return mid
            elif nums[mid]<target:
                l=mid+1
            else:
                r=mid-1
        return -1


# Problem:
# This problem is a variant of closest pair sum. You'll be given two arrays
# arr1 = { {1, 2000}, {2, 3000}, {3, 4000} }
# arr2 = { { 1, 5000 }, {2, 3000} }
# the first element of every pair represents id and the second value represents the value.
# and a target x = 5000
# Find the pairs from both the arrays whose value add upto a sum which is less than given target and should be close to the target.
#
# Output for the above example:
# { {1, 2} } // Note that the output should be in id's

def find(A, B, target):
    ans = [0, 0, 0]
    A = sorted([x, i] for i, x in A)
    for idy, y in B:
        l = 0
        r = len(A)
        z = target - y
        while l != r:
            m = (l + r) // 2
            if A[m][0] <= z:
                l = m + 1
            else:
                r = m
        if l != 0 and y + A[l - 1][0] > ans[0]:
            ans = [y + A[l - 1][0], A[l - 1][1], idy]
    return ans[1:]


print(find([(1, 3000), (2, 5000), (3, 4000), (4, 10000)],
     [(1, 2000), (2, 3000), (3, 4000)], 11000))
# Sort one of the arrays of length N.
# Iterate the other array of length M and do a binary search in the first array updating the global maximum. O(N*log(N) + M*log(N))

# codesignal

import bisect


def boundedSquareSum(a, b, lower, upper):
    aa = [x ** 2 for x in a]
    bb = [x ** 2 for x in b]
    aa.sort()

    n1, n2 = len(aa), len(bb)
    ans = 0
    print(aa)
    for i in range(n2):
        if bb[i] > upper:
            continue

        r = bisect.bisect_right(aa, upper - bb[i])
        l = bisect.bisect_left(aa, lower - bb[i])
        print(l, r)

        ans += (r - l)
    return ans

