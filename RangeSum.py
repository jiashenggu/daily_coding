# 493. Reverse Pairs
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        def merge(l, r):
            if l == r:
                return 0
            mid = (l + r) >> 1
            lp = merge(l, mid)
            rp = merge(mid + 1, r)
            ret = lp + rp

            i = l
            j = mid + 1

            while i <= mid:
                while j <= r and nums[i] > nums[j] * 2:
                    j += 1
                ret += (j - mid - 1)
                i += 1

            i = l
            j = mid + 1
            m = [0] * (r - l + 1)
            for k in range(r - l + 1):
                if i > mid:
                    m[k] = nums[j]
                    j += 1
                elif j > r:
                    m[k] = nums[i]
                    i += 1
                else:
                    if nums[i] < nums[j]:
                        m[k] = nums[i]
                        i += 1
                    else:
                        m[k] = nums[j]
                        j += 1
            nums[l:r + 1] = m

            return ret

        n = len(nums)
        if n == 0:
            return 0
        return merge(0, n - 1)


import sortedcontainers


class BIT:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (1 + n)

    def lowbit(self, x):
        return x & (-x)

    def update(self, x, d):
        while x <= self.n:
            self.tree[x] += d
            x += self.lowbit(x)

    def query(self, x):
        ans = 0
        while x:
            ans += self.tree[x]
            x -= self.lowbit(x)
        return ans


class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        n = len(nums)
        numSet = sortedcontainers.SortedSet()
        for num in nums:
            numSet.add(num)
            numSet.add(num * 2)

        m = {}
        idx = 0
        for num in numSet:
            idx += 1
            m[num] = idx

        bit = BIT(len(m))
        ret = 0
        for num in nums:
            l = m[num * 2]
            r = len(m)
            ret += bit.query(r) - bit.query(l)
            bit.update(m[num], 1)
        return ret


