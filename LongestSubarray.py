# 1438. 绝对差不超过限制的最长连续子数组
class Solution:
    def longestSubarray(self, nums, limit):
        max_deque = collections.deque()
        min_deque = collections.deque()
        left = 0
        max_length = 0

        for right in range(len(nums)):
            # 维护最大值双端队列（递减）
            while max_deque and nums[right] > max_deque[-1]:
                max_deque.pop()
            max_deque.append(nums[right])

            # 维护最小值双端队列（递增）
            while min_deque and nums[right] < min_deque[-1]:
                min_deque.pop()
            min_deque.append(nums[right])

            # 如果当前窗口的最大值 - 最小值 > limit，移动左指针
            while max_deque[0] - min_deque[0] > limit:
                if nums[left] == max_deque[0]:
                    max_deque.popleft()
                if nums[left] == min_deque[0]:
                    min_deque.popleft()
                left += 1

            # 更新最大长度
            max_length = max(max_length, right - left + 1)

        return max_length
