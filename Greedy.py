# 376. Wiggle Subsequence
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n
        prevdiff = nums[1] - nums[0]
        cnt = 2 if prevdiff else 1
        for i in range(2, n):
            diff = nums[i] - nums[i - 1]
            if (diff > 0 and prevdiff <= 0) or (diff < 0 and prevdiff >= 0):
                cnt += 1
                prevdiff = diff
        return cnt


# 1564. Put Boxes Into the Warehouse I
class Solution:
    def maxBoxesInWarehouse(self, boxes: List[int], warehouse: List[int]) -> int:
        n = len(warehouse)
        m = len(boxes)
        left = n * [0]
        left[0] = warehouse[0]
        for i in range(1, n):
            left[i] = min(warehouse[i], left[i - 1])

        boxes.sort()
        ans = 0
        for i in reversed(range(n)):
            if not boxes:
                break
            if boxes[0] <= left[i]:
                ans += 1
                boxes.pop(0)
        return ans


# 406. Queue Reconstruction by Height
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key = lambda x: (-x[0], x[1]))
        output = []
        for p in people:
            output.insert(p[1],p)
        return output

# 1647. Minimum Deletions to Make Character Frequencies Unique
class Solution:
    def minDeletions(self, s: str) -> int:
        c = collections.Counter(s)
        d = collections.defaultdict(list)
        for k, v in c.items():
            d[v].append(k)
        ans = 0
        for k in reversed(sorted(d.keys())):
            v = d[k]
            if k == 0 or len(v) == 1:
                continue
            while len(v) > 1:
                vv = v.pop()
                cnt = 1
                while k - cnt > 0 and len(d[k - cnt]) > 0:
                    cnt += 1
                d[k - cnt].append(vv)
                ans += cnt
        return ans


# 665. Non-decreasing Array
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        n = len(nums)
        flag = False
        for i in range(1, n):
            if nums[i - 1] > nums[i]:
                if flag:
                    return False
                if i == 1:
                    nums[i - 1] = nums[i]
                elif nums[i - 2] > nums[i]:
                    nums[i] = nums[i - 1]
                else:
                    nums[i - 1] = nums[i]
                flag = True
        return True


# 135. Candy
class Solution:
    def candy(self, ratings: List[int]) -> int:

        n = len(ratings)
        candy = [1] * n
        ans = n
        for i in range(n - 1):
            if ratings[i] < ratings[i + 1]:
                if candy[i] >= candy[i + 1]:
                    tmp = candy[i] - candy[i + 1] + 1
                    candy[i + 1] += tmp
                    ans += tmp
        for i in reversed(range(n - 1)):
            if ratings[i] > ratings[i + 1]:
                if candy[i] <= candy[i + 1]:
                    tmp = candy[i + 1] - candy[i] + 1
                    candy[i] += tmp
                    ans += tmp

        # print(candy)
        return ans


class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        l2r = n * [1]
        r2l = n * [1]

        for i in range(1, n):
            if ratings[i - 1] < ratings[i]:
                l2r[i] = l2r[i - 1] + 1
        for i in reversed(range(1, n)):
            if ratings[i - 1] > ratings[i]:
                r2l[i - 1] = r2l[i] + 1
        ans = 0
        for i in range(n):
            ans += max(l2r[i], r2l[i])
        return ans


# 134. Gas Station
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        diff = [gas[i]-cost[i] for i in range(n)]
        total = 0
        mi = sys.maxsize
        idx = 0
        for i, d in enumerate(diff):
            total += d
            if total<mi:
                mi = total
                idx = i
        if total<0:
            return -1
        return (idx+1)%n