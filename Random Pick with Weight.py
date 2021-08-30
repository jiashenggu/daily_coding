class Solution:

    def __init__(self, w: List[int]):
        n = len(w)
        s = 0
        self.prefix = []
        for i in range(n):
            s+=w[i]
            self.prefix.append(s)

    def pickIndex(self) -> int:
        r = random.randint(1, self.prefix[-1])
        idx = bisect.bisect_left(self.prefix, r)
        return idx


# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()