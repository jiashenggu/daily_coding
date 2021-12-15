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