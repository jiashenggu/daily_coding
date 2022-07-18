# linear sieve
n = int(input())
isnp = (n+1)*[0]
primes = []
for i in range(2, n+1):
    if not isnp[i]:
        primes.append(i)
    for p in primes:
        if p*i>n:
            break
        isnp[i*p] = 1
        if i%p == 0:
            break
# 50. Pow(x, n)
class Solution:
    def myPow(self, x: float, n: int) -> float:
        ans = 1
        if n < 0:
            n = abs(n)
            x = 1 / x
        while n:
            if n & 1:
                ans *= x
            x = x * x
            n = n >> 1
        return ans


# 469. Convex Polygon
class Solution:
    def isConvex(self, points: List[List[int]]) -> bool:
        n = len(points)
        pre = 0
        for i in range(n):
            x1 = points[(i+1)%n][0] - points[i][0]
            y1 = points[(i+1)%n][1] - points[i][1]

            x2 = points[(i+2)%n][0] - points[(i+1)%n][0]
            y2 = points[(i+2)%n][1] - points[(i+1)%n][1]

            tmp = x1*y2 - x2*y1
            if tmp:
                if tmp*pre<0:
                    return False
                pre = tmp
        return True

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