# 371. Sum of Two Integers
MASK1 = 4294967296  # 2^32
MASK2 = 2147483648  # 2^31
MASK3 = 2147483647  # 2^31-1


class Solution:
    def getSum(self, a: int, b: int) -> int:
        ab1 = a^b
        ab2 = (a&b)<<1
        while ab2:
            tmp = ab1
            ab1 = (ab1 ^ (ab2))%MASK1
            ab2 = ((tmp&(ab2))<<1)%MASK1
        if ab1 & MASK2:
            return ~((ab1^MASK2)^MASK3)
        return ab1