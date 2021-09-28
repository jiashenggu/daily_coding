# 639. Decode Ways II
class Solution:
    def numDecodings(self, s: str) -> int:
        mod = 10**9+7
        n = len(s)
        s = list(s)
        dp = [0]*(n+1)
        dp[0]=1
        for i in range(1, 1+n):
            a = 0
            if s[i-1]=='*':
                a = 9*dp[i-1]
            elif s[i-1]=='0':
                a = 0
            elif s[i-1].isdigit():
                a = dp[i-1]
            a%=mod
            b = 0
            if i>1:
                if s[i-1]=='*' and s[i-2]=='*':
                    b = 15*dp[i-2]
                elif s[i-2]=='*':
                    if 0<=int(s[i-1])<=6:
                        b = 2*dp[i-2]
                    else:
                        b = dp[i-2]
                elif s[i-1]=='*':
                    if int(s[i-2])==0:
                        b = 0
                    elif int(s[i-2])==1:
                        b = 9*dp[i-2]
                    elif int(s[i-2])==2:
                        b = 6*dp[i-2]
                elif s[i-2]!='0' and (int(s[i-2])*10+int(s[i-1]))<=26:
                    b = dp[i-2]
            b%=mod
            # print(a, b)
            dp[i] = (a+b)%mod

        return dp[n]

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

# 673. Number of Longest Increasing Subsequence
class Solution(object):
    def findNumberOfLIS(self, nums):
        n = len(nums)
        cnt = [1]*n
        length = [1]*n

        for i in range(n):
            for j in range(i):
                if nums[i]>nums[j]:
                    if length[i]<=length[j]:
                        length[i] = length[j]+1
                        cnt[i]=cnt[j]
                    elif length[j]+1==length[i]:
                        cnt[i]+=cnt[j]
        target = max(length)
        # print(target)
        # print(length)
        # print(cnt)
        return sum([c for i, c in enumerate(cnt) if length[i]==target])
# 72. Edit Distance
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)

        if n * m == 0:
            return n + m

        d = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = d[i][j - 1] + 1
                down = d[i - 1][j] + 1
                left_down = d[i - 1][j - 1]
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                d[i][j] = min(left, down, left_down)
        return d[n][m]

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)

        if n*m == 0:
            return n+m

        d = [ [0]*(m+1) for _ in range(n+1)]

        for i in range(n+1):
            d[i][0] = i
        for j in range(m+1):
            d[0][j] = j

        for i in range(1, n+1):
            for j in range(1, m+1):
                left = d[i][j-1]+1
                down = d[i-1][j]+1
                left_down = d[i-1][j-1]
                if word1[i-1]!=word2[j-1]:
                    left_down+=1
                d[i][j] = min(left, down, left_down)
        return d[n][m]

# 650. 2 Keys Keyboard
class Solution:
    def minSteps(self, n: int) -> int:
        f = [float("inf")]*(1+n)
        f[0] = 0
        f[1] = 0
        for i in range(2, 1+n):
            j = 1
            while j*j<=i:
                if i%j==0:
                    f[i] = min(f[i], f[j]+i//j, f[i//j]+j)
                j+=1
        print(f)

        return f[n]
# prime factorization
class Solution:
    def minSteps(self, n: int) -> int:
        ans = 0
        d = 2
        while n>1:
            while n%d==0:
                ans+=d
                n//=d
            d+=1
        return ans
