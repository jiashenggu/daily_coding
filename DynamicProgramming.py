# 600. Non-negative Integers without Consecutive Ones
class Solution:
    def findIntegers(self, n: int) -> int:
        dp = 31 * [0]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, 31):
            dp[i] = dp[i - 1] + dp[i - 2]

        pre = 0
        res = 0
        for i in range(29, -1, -1):
            val = 1 << i
            if val & n:
                n -= val
                res += dp[i + 1]
                if pre == 1:
                    break
                pre = 1
            else:
                pre = 0
            if i == 0:
                res += 1
        return res

# 1326. Minimum Number of Taps to Open to Water a Garden
class Solution:
    def minTaps(self, n: int, ranges: List[int]) -> int:
        pre = [ x for x in range(n+1)]
        for i in range(n+1):
            l = max(i-ranges[i], 0)
            r = min(i+ranges[i], n)
            pre[r] = min(pre[r], l)

        dp = [0] + [float("inf")]*n
        for i in range(n+1):
            for j in range(pre[i], i):
                # if dp[j]!=float("inf"):
                dp[i] = min(dp[j]+1, dp[i])

        return dp[n] if dp[n]!=float("inf") else -1