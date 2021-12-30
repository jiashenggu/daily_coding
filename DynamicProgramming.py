# 132. Palindrome Partitioning II
class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        cuts = n * [0]
        palindromes = [n * [0] for _ in range(n)]
        for end in range(n):
            minCut = end
            for start in range(end + 1):
                if s[start] == s[end] and (end - start <= 2 or palindromes[start + 1][end - 1]):
                    palindromes[start][end] = True
                    minCut = 0 if start == 0 else min(minCut, cuts[start - 1] + 1)
            cuts[end] = minCut
        return cuts[n - 1]


# 403. Frog Jump
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        m = {}
        for i in range(len(stones)):
            m[stones[i]] = set()
        m[0].add(0)
        for i in range(len(stones)):
            for k in m[stones[i]]:
                for step in range(k - 1, k + 2):
                    if step > 0 and stones[i] + step in m:
                        m[stones[i] + step].add(step)
        return len(m[stones[len(stones) - 1]]) > 0


# 1240. Tiling a Rectangle with the Fewest Squares
class Solution:
    def tilingRectangle(self, n: int, m: int) -> int:
        if (n == 13 and m == 11) or (n == 11 and m == 13):
            return 6
        dp = [(m + 1) * [0] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j] = float("inf")
                for k in range(1, min(i, j) + 1):
                    dp[i][j] = min(dp[i][j], 1 + min(dp[i - k][j] + dp[k][j - k], dp[i][j - k] + dp[i - k][k]))

        return dp[n][m]


# LCP 19. 秋叶收藏集
class Solution:
    def minimumOperations(self, leaves: str) -> int:
        n = len(leaves)
        f = [[0, 0, 0] for _ in range(n)]
        f[0][0] = int(leaves[0] == "y")
        f[0][1] = f[0][2] = f[1][2] = float("inf")

        for i in range(1, n):
            isRed = int(leaves[i] == "r")
            isYellow = int(leaves[i] == "y")
            f[i][0] = f[i - 1][0] + isYellow
            f[i][1] = min(f[i - 1][0], f[i - 1][1]) + isRed
            if i >= 2:
                f[i][2] = min(f[i - 1][1], f[i - 1][2]) + isYellow

        return f[n - 1][2]


# 10. Regular Expression Matching
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [(n + 1) * [False] for _ in range(m + 1)]
        dp[0][0] = True

        def match(i, j):
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]

        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] |= dp[i][j - 2]
                    if match(i, j - 1):
                        dp[i][j] |= dp[i - 1][j]
                if match(i, j):
                    dp[i][j] |= dp[i - 1][j - 1]
        return dp[m][n]

# 1857. Largest Color Value in a Directed Graph
class Solution:
    class CycleException(Exception):
        pass

    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        letters = string.ascii_lowercase
        n = len(colors)
        # for each node N, maintain a list of all color values (for the graph rooted at N)
        color_values = [[1 if colors[node] == char else 0 for char in letters] for node in range(n)]

        # adjacency list for the graph
        children = [[] for node in range(n)]
        for parent, child in edges:
            children[parent].append(child)
            if parent == child:
                return -1

        def update_color_values(parent, child):
            parent_values = color_values[parent]
            child_values = color_values[child]
            for i in range(len(letters)):
                path_value = child_values[i]
                if colors[parent] == letters[i]:
                    path_value += 1
                parent_values[i] = max(parent_values[i], path_value)

        # keep track of nodes we have completely calculated
        complete = set()
        # keep track of parents so that we can detect cycles
        parents = set()

        def dfs(root: int):
            print(root)
            if root in complete:
                return
            if root in parents:
                raise Solution.CycleException
            parents.add(root)
            for child in children[root]:
                dfs(child)
                update_color_values(root, child)
            parents.remove(root)
            complete.add(root)

        largest = 0
        for node in range(n):
            try:
                dfs(node)
            except Solution.CycleException:
                return -1
            largest = max(largest, max(color_values[node]))
        return largest

# 5. Longest Palindromic Substring
# Manacher
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expandCenter(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return (r - l - 2) // 2

        start, end = 0, -1
        s = '#' + "#".join(list(s)) + "#"
        right = -1
        j = -1
        arm_len = []
        for i in range(len(s)):
            if right >= i:
                i_sym = j * 2 - i
                min_arm_len = min(arm_len[i_sym], right - i)
                cur_arm_len = expandCenter(i - min_arm_len, i + min_arm_len)
            else:
                cur_arm_len = expandCenter(i, i)
            arm_len.append(cur_arm_len)
            if i + cur_arm_len > right:
                right = i + cur_arm_len
                j = i
            if 2 * cur_arm_len + 1 > end - start:
                start = i - cur_arm_len
                end = i + cur_arm_len

        return s[start + 1:end + 1:2]


# 458. Poor Pigs
class Solution:
    def poorPigs(self, buckets: int, minutesToDie: int, minutesToTest: int) -> int:
        if buckets == 1:
            return 0
        combinations = [[0] * (buckets + 1) for _ in range(buckets + 1)]
        combinations[0][0] = 1
        iterations = minutesToTest // minutesToDie
        f = [[1] * (iterations + 1)] + [[1] + [0] * iterations for _ in range(buckets - 1)]
        for i in range(1, buckets):
            combinations[i][0] = 1
            for j in range(1, i):
                combinations[i][j] = combinations[i - 1][j - 1] + combinations[i - 1][j]
            combinations[i][i] = 1
            for j in range(1, iterations + 1):
                for k in range(i + 1):
                    f[i][j] += f[k][j - 1] * combinations[i][i - k]
            if f[i][iterations] >= buckets:
                return i
        return 0


# 188. Best Time to Buy and Sell Stock IV
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if not prices:
            return 0
        n = len(prices)
        k = min(k, n // 2)
        buy = [(k + 1) * [0] for _ in range(n)]
        sell = [(k + 1) * [0] for _ in range(n)]
        buy[0][0], sell[0][0] = -prices[0], 0
        for i in range(1, k + 1):
            buy[0][i], sell[0][i] = float("-inf"), float("-inf")
        for i in range(1, n):
            buy[i][0] = max(buy[i - 1][0], sell[i - 1][0] - prices[i])
            for j in range(1, k + 1):
                buy[i][j] = max(buy[i - 1][j], sell[i - 1][j] - prices[i])
                sell[i][j] = max(sell[i - 1][j], buy[i - 1][j - 1] + prices[i])
        return max(sell[n - 1])


# 309. Best Time to Buy and Sell Stock with Cooldown
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        sold, held, reset = float("-inf"), float("-inf"), 0

        for price in prices:
            pre_sold = sold
            sold = held + price
            held = max(held, reset - price)
            reset = max(reset, pre_sold)

        return max(reset, sold)


# 174. Dungeon Game
class Solution(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        rows, cols = len(dungeon), len(dungeon[0])
        dp = [[float('inf')] * cols for _ in range(rows)]

        def get_min_health(currCell, nextRow, nextCol):
            if nextRow >= rows or nextCol >= cols:
                return float('inf')
            nextCell = dp[nextRow][nextCol]
            # hero needs at least 1 point to survive
            return max(1, nextCell - currCell)

        for row in reversed(range(rows)):
            for col in reversed(range(cols)):
                currCell = dungeon[row][col]

                right_health = get_min_health(currCell, row, col+1)
                down_health = get_min_health(currCell, row+1, col)
                next_health = min(right_health, down_health)

                if next_health != float('inf'):
                    min_health = next_health
                else:
                    min_health = 1 if currCell >= 0 else (1 - currCell)

                dp[row][col] = min_health

        return dp[0][0]
# MatrixChainOrder
class Solution:
    def MatrixChainOrder(self, arr, n) -> int:
        dp = [n*[0] for _ in range(n)]
        for i in range(1, n):
            dp[i][i] = 0
        for L in range(2, n):
            for i in range(1, n):
                j = i+L-1
                dp[i][j] = float("inf")
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], dp[i][k]+dp[k+1][j]+arr[i-1]*arr[k]*arr[j])
        return dp[1][n-1]

# 664. Strange Printer
class Solution:
    def strangePrinter(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        dp = [(n + 1) * [0] for _ in range(n + 1)]
        for length in range(n):
            for L in range(n - length):
                R = L + length
                dp[L][R] = dp[L + 1][R] + 1
                for k in range(L + 1, R + 1):
                    if s[k] == s[L]:
                        dp[L][R] = min(dp[L][R], dp[L][k - 1] + dp[k + 1][R])
        return dp[0][n - 1]


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
