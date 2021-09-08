# 327. Count of Range Sum
class Solution(object):
    def countRangeSum(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: int
        """
        from sortedcontainers import SortedList

        sl = SortedList([0])
        res = 0
        count = 0
        for num in nums:
            count += num
            res += sl.bisect_right(count-lower) - sl.bisect_left(count-upper)
            sl.add(count)
        return res
class SegTree:
    def __init__(self, n):
        self.n = n
        self.tree = (4 * self.n) * [0]

    def update(self, ID, diff):
        self._update(0, 0, self.n - 1, ID, diff)

    def query(self, ql, qr):
        return self._query(0, 0, self.n - 1, ql, qr)

    def _update(self, root, l, r, ID, diff):
        if l == r == ID:
            self.tree[root] += diff
            return
        left = root * 2 + 1
        right = root * 2 + 2
        mid = (l + r) >> 1
        if ID <= mid:
            self._update(left, l, mid, ID, diff)
        else:
            self._update(right, mid + 1, r, ID, diff)
        self.tree[root] = self.tree[left] + self.tree[right]

    def _query(self, root, l, r, ql, qr):
        if l == ql and r == qr:
            return self.tree[root]
        left = root * 2 + 1
        right = root * 2 + 2
        mid = l + r >> 1
        if qr <= mid:
            return self._query(left, l, mid, ql, qr)
        elif mid + 1 <= ql:
            return self._query(right, mid + 1, r, ql, qr)
        else:
            return self._query(left, l, mid, ql, mid) + self._query(right, mid + 1, r, mid + 1, qr)


class BIT:
    def __init__(self, n):
        self.n = n
        self.tree = (n + 1) * [0]

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
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        accu = [0] + list(accumulate(nums))

        n = len(nums)
        numSet = set()
        for num in accu:
            numSet.add(num)
            numSet.add(num - lower)
            numSet.add(num - upper)
        m = len(numSet)
        numSet = list(numSet)
        numSet.sort()
        bit = BIT(m)
        val = {}
        idx = 0
        for num in numSet:
            val[num] = idx
            idx += 1
        ans = 0
        for num in accu:
            r = val[num - lower]
            l = val[num - upper]
            ans += bit.query(r + 1) - bit.query(l)
            bit.update(val[num] + 1, 1)
        return ans

class Solution(object):
    def countRangeSum(self, nums, lower, upper):
        n = len(nums)
        presum = (n + 1) * [0]
        for i in range(n):
            presum[i + 1] = presum[i] + nums[i]
        all_num = []
        for p in presum:
            all_num += [p, p - lower, p - upper]
        all_num = list(set(all_num))
        all_num.sort()
        n2 = len(all_num)
        val_id = {}
        for id, val in enumerate(all_num):
            val_id[val] = id
        res = 0

        st = SegTree(n2)
        for p in presum:
            l = val_id[p - upper]
            r = val_id[p - lower]
            res += st.query(l, r)
            st.update(val_id[p], 1)
        return res

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


