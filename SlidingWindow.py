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
