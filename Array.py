# 4. Median of Two Sorted Arrays
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1 = len(nums1)
        n2 = len(nums2)
        idx_median = (n1 + n2) // 2
        i1, i2 = 0, 0
        left, right = 0, 0
        while i1 < n1 and i2 < n2 and i1 + i2 < idx_median + 1:
            if nums1[i1] < nums2[i2]:
                left = right
                right = nums1[i1]
                i1 += 1
            else:
                left = right
                right = nums2[i2]
                i2 += 1
        # print(left, right)
        while i1 < n1 and i1 + i2 < idx_median + 1:
            left = right
            right = nums1[i1]
            i1 += 1
        while i2 < n2 and i1 + i2 < idx_median + 1:
            left = right
            right = nums2[i2]
            i2 += 1
        if (n1 + n2) % 2 == 0:
            return (left + right) / 2
        else:
            return right


# 442. Find All Duplicates in an Array
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        ans = []
        for num in nums:
            nums[abs(num) - 1] *= -1
        for num in nums:
            if nums[abs(num) - 1] > 0:
                ans.append(abs(num))
                nums[abs(num) - 1] *= -1
        return ans


# Python3 implementation of the approach
def getZarr(s, n, Z):
    L, R, k = 0, 0, 0

    # [L, R] make a window which matches
    # with prefix of s
    for i in range(n):
        # if i>R nothing matches so we will
        # calculate Z[i] using naive way.

        if i > R:
            L, R = i, i

            '''
            R-L = 0 in starting, so it will start
            checking from 0'th index. For example,
            for "ababab" and i = 1, the value of R
            remains 0 and Z[i] becomes 0. For string
            "aaaaaa" and i = 1, Z[i] and R become 5
            '''
            while R < n and s[R - L] == s[R]:
                R += 1
            print('L', L, 'R', R)
            Z[i] = R - L
            R -= 1
        else:

            # k = i-L so k corresponds to number
            # which matches in [L, R] interval.
            k = i - L
            print("i", i, 'L', L, 'R', R)
            print(k, Z[k])

            # if Z[k] is less than remaining interval
            # then Z[i] will be equal to Z[k].
            # For example, str = "ababab", i = 3, R = 5
            # and L = 2
            if Z[k] < R - i + 1:
                Z[i] = Z[k]
            else:
                L = i
                while R < n and s[R - L] == s[R]:
                    R += 1
                Z[i] = R - L
                R -= 1
        print("Z", Z)


def sumSimilarities(s, n):
    Z = [0 for i in range(n)]

    # Compute the Z-array for the
    # given string
    getZarr(s, n, Z)

    total = n

    # summation of the Z-values
    for i in range(n):
        total += Z[i]
    return total


# Driver Code
s = "ababa"

n = len(s)

print(sumSimilarities(s, n))

# This code is contributed
# by Mohit kumar 29
