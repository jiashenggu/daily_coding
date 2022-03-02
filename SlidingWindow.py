# 76. Minimum Window Substring
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        m = defaultdict(int)
        for c in t:
            m[c] += 1
        required = list(m.keys()).copy()
        i, j = 0, 0
        n = len(s)
        start = float("-inf")
        end = float("inf")

        def check(m):
            for k in m:
                if m[k] > 0:
                    return False
            return True

        while j < n:
            # print(m)
            # print(i, j, s[i: j+1])
            if s[j] in required:
                m[s[j]] -= 1
            while check(m) and i <= j:
                ch = s[i]
                if j - i + 1 < end - start + 1:
                    start = i
                    end = j
                if ch in required:
                    m[ch] += 1
                i += 1
            j += 1
        return s[start:end + 1] if start != float("-inf") else ""



# 480. Sliding Window Median
from sortedcontainers import SortedList
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        medians, window = [], []

        for i in range(len(nums)):
            if i >= k:
                window.remove(nums[i - k])
                # window.pop(bisect.bisect_left(window, nums[i-k]))
            bisect.insort(window, nums[i])
            if i >= k - 1:
                if k & 1:
                    medians.append(window[k // 2])
                else:
                    medians.append((window[k // 2 - 1] + window[k // 2]) / 2)
        return medians


# 239. Sliding Window Maximum
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = []
        ans = []
        if len(nums) <= k:
            return [max(nums)]
        for i in range(k - 1):
            while q and q[-1] < nums[i]:
                q.pop()
            q.append(nums[i])
        for i, num in enumerate(nums):
            if i < k - 1:
                continue
            while q and q[-1] < num:
                q.pop()
            q.append(num)
            if nums[i - k] == q[0]:
                q.pop(0)
            ans.append(q[0])
        return ans
