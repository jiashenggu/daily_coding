# 229. Majority Element II
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        vote1, vote2 = 0, 0
        element1, element2 = 0, 0
        for num in nums:
            if vote1>0 and num==element1:
                vote1+=1
            elif vote2>0 and num==element2:
                vote2+=1
            elif vote1==0:
                element1 = num
                vote1+=1
            elif vote2==0:
                element2 = num
                vote2+=1

            else:
                vote1-=1
                vote2-=1
        cnt1, cnt2 = 0, 0
        for num in nums:
            if vote1>0 and num==element1:
                cnt1+=1
            if vote2>0 and  num==element2:
                cnt2+=1
        ans = []
        n = len(nums)
        # print(element1, element2)
        if vote1>0 and cnt1>n//3:
            ans.append(element1)
        if vote2>0 and cnt2>n//3:
            ans.append(element2)
        return ans
# 剑指 Offer 42. 连续子数组的最大和
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0]*n
        dp[0]=nums[0]
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i-1]+nums[i])
        return max(dp)

class Status:
    def __init__(self, iSum,lSum, rSum, mSum):
        self.iSum = iSum
        self.lSum = lSum
        self.rSum = rSum
        self.mSum = mSum
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        def pushUp(l, r):
            iSum = l.iSum + r.iSum
            lSum = max(l.lSum, l.iSum+r.lSum)
            rSum = max(r.rSum, r.iSum+l.rSum)
            mSum = max(l.mSum, r.mSum, l.rSum+r.lSum)
            return Status(iSum, lSum, rSum, mSum)
        def get(l, r):
            if l==r:
                return Status(nums[l], nums[l], nums[l], nums[l])
            mid = (l+r)>>1
            lSub = get(l, mid)
            rSub = get(mid+1, r)
            return pushUp(lSub, rSub)
        return get(0, len(nums)-1).mSum
# 509. Fibonacci Number
import numpy as np
class Solution:
    def fib(self, n: int) -> int:
        MOD = 10 ** 9 + 7
        if n < 2:
            return n

        def multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
            lk = len(a[0])
            # c = [[0, 0], [0, 0]]
            c = [ [0]*lk for _ in range(lk) ]
            for k in range(2):
                for i in range(2):
                    for j in range(2):
                        c[i][j] = (c[i][j] + (a[i][k] * b[k][j]) % MOD)%MOD
            return c

        def matrix_pow(a: List[List[int]], n: int) -> List[List[int]]:
            ret = [[1, 0], [0, 1]]
            while n > 0:
                if n & 1:
                    ret = multiply(ret, a)
                n >>= 1
                a = multiply(a, a)
            return ret

        res = matrix_pow([[1, 1], [1, 0]], n - 1)
        return res[0][0]

# 5865. First Day Where You Have Been in All the Rooms

class Solution:
    def firstDayBeenInAllRooms(self, nextVisit: List[int]) -> int:
        n = len(nextVisit)
        ans = 0
        pre = [0] * n
        m = 10 ** 9 + 7
        for i, vis in enumerate(nextVisit):
            if i + 1 >= n:
                break
            tmp = (m + 2 + pre[i] - (0 if vis == 0 else pre[vis])) % m
            ans = (ans + tmp) % m
            pre[i + 1] = ans
        return ans % m


