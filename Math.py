# 780. Reaching Points
class Solution(object):
    def reachingPoints(self, sx, sy, tx, ty):
        while tx >= sx and ty >= sy:
            if tx == ty:
                break
            elif tx > ty:
                if ty > sy:
                    tx %= ty
                elif ty==sy:
                    return (tx - sx) % ty == 0
            else:
                if tx > sx:
                    ty %= tx
                elif tx==sx:
                    return (ty - sy) % tx == 0

        return tx == sx and ty == sy
# 1492. The kth Factor of n
class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        l, r = [], []
        for i in range(1, int(n**0.5)+1):
            ans, re = divmod(n, i)
            if re==0:
                if ans==i:
                    l.append(i)
                else:
                    l = l+[i]
                    r = [ans] + r
                if len(l)>=k:
                    return l[k-1]
        factors = l+r
        if len(factors)>=k:
            return factors[k-1]
        return -1