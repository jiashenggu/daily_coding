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