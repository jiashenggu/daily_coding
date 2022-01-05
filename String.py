# 227. Basic Calculator II
class Solution:
    def calculate(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        stk = []
        cur = 0
        sign = '+'
        for i in range(n):
            c = s[i]
            if c.isdigit():
                cur = cur * 10 + int(c)
            if c in "+-*/" or i == n - 1:
                if sign == '-':
                    stk.append(-cur)
                elif sign == '+':
                    stk.append(cur)
                elif sign == "*":
                    stk.append(stk.pop() * cur)
                elif sign == '/':
                    if stk[-1] > 0:
                        stk.append(stk.pop() // cur)
                    else:
                        stk.append(-(-stk.pop() // cur))
                sign = c
                cur = 0
        res = 0
        # print(stk)
        for num in stk:
            res += num
        return res


# 1153. String Transforms Into Another String
class Solution:
    def canConvert(self, str1: str, str2: str) -> bool:
        if str1==str2:
            return True
        uni = set()
        conversion = dict()
        for l1, l2 in zip(str1, str2):
            if l1 not in conversion:
                conversion[l1] = l2
                uni.add(l2)
            elif conversion[l1]!=l2:
                return False
        if len(uni)<26:
            return True
        return False
# 408. Valid Word Abbreviation
class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        i, j = 0, 0
        n = len(word)
        m = len(abbr)
        while i<n and j<m:
            if word[i]==abbr[j]:
                i+=1
                j+=1
            elif abbr[j]=='0':
                return False
            elif abbr[j].isdigit():
                num = 0
                while j<m and abbr[j].isdigit():
                    num = num*10+int(abbr[j])
                    j+=1
                i+=num
            else:
                return False
        return i==n and j==m