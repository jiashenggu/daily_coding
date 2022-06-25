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