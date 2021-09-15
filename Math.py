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