# 139. Word Break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        @lru_cache
        def wordBreakMemo(s: str, word_dict: FrozenSet[str], start: int):
            if start == len(s):
                return True
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_dict and wordBreakMemo(s, word_dict, end):
                    return True
            return False

        return wordBreakMemo(s, frozenset(wordDict), 0)
# 722. Remove Comments
class Solution:
    def removeComments(self, source: List[str]) -> List[str]:
        s = ""
        block = False
        ans = []
        for line in source:
            i = 0
            n = len(line)
            while i < n:
                if line[i] == '/' and i + 1 < n and line[i + 1] == '/' and not block:
                    i = len(line)
                elif line[i] == '/' and i + 1 < n and line[i + 1] == '*' and not block:
                    block = True
                    i += 1
                elif line[i] == '*' and i + 1 < n and line[i + 1] == '/' and block:
                    block = False
                    i += 1
                elif not block:
                    s += line[i]
                i += 1
            if s and not block:
                ans.append(s)
                s = ""
        return ans


class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expandCenter(l, r):
            while l>=0 and r<len(s) and s[l]==s[r]:
                l-=1
                r+=1
            return (r-l)-1
        if not s:
            return ""
        start, end = 0, 0
        for i in range(len(s)):
            l1 = expandCenter(i, i)
            l2 = expandCenter(i, i+1)
            l = max(l1, l2)
            if l>end-start:
                # deal with odd and even simultaneously
                start = i - (l-1)//2
                end = i + l//2
        return s[start:end+1]
# manacher
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expandCenter(l, r):
            while l>=0 and r<len(s) and s[l]==s[r]:
                l-=1
                r+=1
            return (r-l)//2 - 1
        s = "*" + "*".join(list(s)) + "*"
        start, end = 0, -1
        arr_len = []
        right = -1
        j = -1
        for i in range(len(s)):
            if right>=i:
                i_sym = 2*j - i
                pre_len = min(arr_len[i_sym], right-i)
                cur_len = expandCenter(i-pre_len, i+pre_len)
            else:
                cur_len = expandCenter(i, i)
            arr_len.append(cur_len)
            if i+cur_len>right:
                j = i
                right = i+cur_len
            if 2*cur_len+1>end-start:
                end = i+cur_len
                start = i-cur_len

        return s[start+1: end+1: 2]
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