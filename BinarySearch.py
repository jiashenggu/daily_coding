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

