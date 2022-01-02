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