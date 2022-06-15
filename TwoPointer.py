# 11. Container With Most Water
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        maxarea = 0

        while l < r:
            width = r - l
            maxarea = max(maxarea, min(height[l], height[r]) * width)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return maxarea


# 1658. Minimum Operations to Reduce X to Zero
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        total = sum(nums)
        n = len(nums)
        maxi = -1
        left = 0
        current = 0
        for right in range(n):
            current += nums[right]

            while current > total - x and left <= right:
                current -= nums[left]
                left += 1
            if current == total - x:
                maxi = max(maxi, right - left + 1)
        return n - maxi if maxi != -1 else -1

class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        current = sum(nums)
        n = len(nums)
        mini = inf
        left = 0

        for right in range(n):
            # sum([0,..,left) + (right,...,n-1]) = x
            current -= nums[right]
            # if smaller, move `left` to left
            while current < x and left <= right:
                current += nums[left]
                left += 1
            # check if equal
            if current == x:
                mini = min(mini, (n-1-right)+left)

        return mini if mini != inf else -1

# 88. Merge Sorted Array
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p = m + n
        m -= 1
        n -= 1
        while p > 0:
            p -= 1
            if m < 0:
                nums1[p] = nums2[n]
                n -= 1
            elif n < 0:
                nums1[p] = nums1[m]
                m -= 1
            elif nums1[m] < nums2[n]:
                nums1[p] = nums2[n]
                n -= 1
            else:
                nums1[p] = nums1[m]
                m -= 1

            # print(nums1, m, n)


# 283. Move Zeroes
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        pre = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[pre] = nums[pre], nums[i]
                pre += 1


# 15. 3Sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ans = []
        n = len(nums)
        nums.sort()

        def twoSum(i):
            l, r = i + 1, n - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s == 0:
                    ans.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                elif s > 0:
                    r -= 1
                else:
                    l += 1

        for i in range(n):
            if nums[i] > 0:
                break
            if i == 0 or nums[i - 1] != nums[i]:
                twoSum(i)
        return ans


# 1868. Product of Two Run-Length Encoded Arrays
class Solution:
    def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
        n1 = len(encoded1)
        n2 = len(encoded2)
        i1 = 0
        i2 = 0
        ans = []
        while i1 < n1 and i2 < n2:
            num1 = encoded1[i1][0]
            num2 = encoded2[i2][0]
            v = num1 * num2
            cnt1 = encoded1[i1][1]
            cnt2 = encoded2[i2][1]
            f = 0
            if cnt1 < cnt2:
                f = cnt1
                encoded2[i2][1] -= cnt1
                i1 += 1
            elif cnt1 > cnt2:
                f = cnt2
                encoded1[i1][1] -= cnt2
                i2 += 1
            else:
                f = cnt1
                encoded1[i1][1] -= cnt2
                encoded2[i2][1] -= cnt1
                i1 += 1
                i2 += 1
            if ans and ans[-1][0] == v:
                ans[-1][1] += f
            else:
                ans.append([v, f])
        return ans


# 31. Next Permutation
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i = n-2;
        while i>=0 and nums[i]>=nums[i+1]:
            i-=1
        if i==-1:
            nums.reverse()
            return
        j = n-1
        while nums[j]<=nums[i]:
            j-=1
        nums[i], nums[j] = nums[j], nums[i]
        l, r = i+1, n-1
        while l<r:
            nums[l], nums[r] = nums[r], nums[l]
            l+=1
            r-=1

# 443. String Compression
class Solution:
    def compress(self, chars: List[str]) -> int:
        n = len(chars)
        if n == 1:
            return n
        cnt = 1
        ans = 0
        start = 0
        for i in range(1, n):
            if chars[i - 1] == chars[i]:
                cnt += 1
            else:
                if cnt == 1:
                    ans += 1
                    chars[start] = chars[i - 1]
                    start += 1
                else:
                    ccnt = 0
                    chars[start] = chars[i - 1]
                    q = []
                    while cnt:
                        q.append(str(cnt % 10))
                        cnt //= 10
                        ccnt += 1
                    while q:
                        # print(q)
                        start += 1
                        chars[start] = q.pop(-1)
                    start += 1
                    ans += 1 + ccnt
                cnt = 1

        if cnt == 1:
            ans += 1
            chars[start] = chars[i]
            start += 1
        else:
            ccnt = 0
            chars[start] = chars[i]
            q = []
            while cnt:
                q.append(str(cnt % 10))
                cnt //= 10
                ccnt += 1
            while q:
                # print(q)
                start += 1
                chars[start] = q.pop(-1)
            start += 1
            ans += 1 + ccnt

        return ans


# 1868. Product of Two Run-Length Encoded Arrays
class Solution:
    def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
        # two pointers
        p1, p2 = 0, 0
        ans = []
        while p1 < len(encoded1) and p2 < len(encoded2):
            l1 = encoded1[p1][1]
            l2 = encoded2[p2][1]
            current_v = encoded1[p1][0] * encoded2[p2][0]

            if l1 < l2:
                current_f = l1
                p1 += 1
                encoded2[p2][1] = l2 - l1  # update frequency for uncounted encoded2
            elif l2 < l1:
                current_f = l2
                p2 += 1
                encoded1[p1][1] = l1 - l2  # update frequency for uncounted encoded1
            else:  # l1 == l2
                current_f = l2
                p1 += 1
                p2 += 1

            # update ans
            if ans and ans[-1][0] == current_v:  # check if current_v == previous value in ans
                ans[-1][1] += current_f
            else:
                ans.append([current_v, current_f])
        return ans