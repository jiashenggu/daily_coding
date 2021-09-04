# 84. Largest Rectangle in Histogram
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stk = []
        ans = 0
        n = len(heights)
        for i, h in enumerate(heights):
            while stk and heights[stk[-1]]>=h:
                start = stk.pop()
                width = i if not stk else i-stk[-1]-1
                ans = max(ans, heights[start]*width)
            stk.append(i)
        while stk:
            start = stk.pop()
            width = n if not stk else n-stk[-1]-1
            ans = max(ans, heights[start]*width)
        return ans