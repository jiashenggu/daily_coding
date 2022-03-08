# 484. Find Permutation
class Solution:
    def findPermutation(self, s: str) -> List[int]:
        stk = []
        ans = []
        num = 1
        for c in s:
            if c == 'I':
                stk.append(num)
                while stk:
                    ans.append(stk.pop())
            elif c == 'D':
                stk.append(num)
            num += 1
        stk.append(num)
        while stk:
            ans.append(stk.pop())

        return ans