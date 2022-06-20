# 820. Short Encoding of Words
Trie = lambda: collections.defaultdict(Trie)


class Solution:
    def minimumLengthEncoding(self, words: List[str]) -> int:
        trie = Trie()
        for word in words:
            t = trie
            for ch in reversed(word):
                t['@'] = False
                t = t[ch]
            t['@'] = True

        def dfs(t, index):
            tmp = 0
            for ch in t:
                if ch == '@':
                    continue
                ret = dfs(t[ch], index + 1)
                tmp += ret
            if tmp != 0:
                return tmp
            if t['@']:
                return index + 1
            return 0

        return dfs(trie, 0)


# 1268. Search Suggestions System
Trie = lambda: collections.defaultdict(Trie)
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        trie = Trie()
        for word in products:
            t = trie
            for ch in word:
                t = t[ch]
                if '@' not in t:
                    t['@'] = []
                heapq.heappush(t['@'], word)
        t = trie
        ans = []
        for ch in searchWord:
            t = t[ch]
            tmp = []
            k = 3
            while k>0 and len(t['@']) != 0:
                k-=1
                tmp.append(heapq.heappop(t['@']))
            ans.append(tmp)
        return ans
# 745. Prefix and Suffix Search
class WordFilter:

    def __init__(self, words: List[str]):
        vis = set()
        self.trie1 = {}
        n = len(words)
        for index, word in enumerate(reversed(words)):
            i = n - 1 - index
            if word in vis:
                continue
            vis.add(word)
            t1 = self.trie1
            for ch in word:
                if ch not in t1:
                    t1[ch] = {}
                t1 = t1[ch]
                if '@' not in t1:
                    t1['@'] = set()
                t1['@'].add(i)

        self.trie2 = {}

        vis = set()
        for index, word in enumerate(reversed(words)):
            i = n - 1 - index
            if word in vis:
                continue
            vis.add(word)
            t2 = self.trie2
            for ch in reversed(word):
                if ch not in t2:
                    t2[ch] = {}
                t2 = t2[ch]
                if '@' not in t2:
                    t2['@'] = set()
                t2['@'].add(i)

    def f(self, prefix: str, suffix: str) -> int:

        t1 = self.trie1
        flag = True
        for ch in prefix:
            if ch not in t1:
                flag = False
                break
            t1 = t1[ch]
        if flag:
            s1 = t1['@']
        else:
            return -1
        t2 = self.trie2
        flag = True
        for ch in reversed(suffix):
            if ch not in t2:
                flag = False
                break
            t2 = t2[ch]
        if flag:
            s2 = t2['@']
        else:
            return -1
        # print(s1, s2)
        s = s1 & s2
        # print(len(s))
        if len(s) == 0:
            return -1
        return sorted(s1 & s2)[-1]


# Your WordFilter object will be instantiated and called as such:
# obj = WordFilter(words)
# param_1 = obj.f(prefix,suffix)

from itertools import zip_longest
Trie = lambda: collections.defaultdict(Trie)
WEIGHT = False

class WordFilter(object):
    def __init__(self, words):
        self.trie = Trie()

        for weight, word in enumerate(words):
            cur = self.trie
            cur[WEIGHT] = weight
            for i, x in enumerate(word):
                #Put all prefixes and suffixes
                tmp = cur
                for letter in word[i:]:
                    tmp = tmp[letter, None]
                    tmp[WEIGHT] = weight

                tmp = cur
                for letter in word[:-i][::-1]:
                    tmp = tmp[None, letter]
                    tmp[WEIGHT] = weight

                #Advance letters
                cur = cur[x, word[~i]]
                cur[WEIGHT] = weight

    def f(self, prefix, suffix):
        cur = self.trie
        for a, b in zip_longest(prefix, suffix[::-1]):
            if (a, b) not in cur: return -1
            cur = cur[a, b]
        return cur[WEIGHT]


Trie = lambda: collections.defaultdict(Trie)
from itertools import zip_longest


class WordFilter:
    def __init__(self, words: List[str]):
        self.trie = Trie()

        for weight, word in enumerate(words):
            word += '#'
            for i in range(len(word)):
                cur = self.trie
                cur['@'] = weight
                for j in range(i, 2 * len(word) - 1):
                    cur = cur[word[j % len(word)]]
                    cur['@'] = weight

    def f(self, prefix: str, suffix: str) -> int:
        cur = self.trie
        for letter in suffix + '#' + prefix:
            if letter not in cur:
                return -1
            cur = cur[letter]
        return cur['@']


# Your WordFilter object will be instantiated and called as such:
# obj = WordFilter(words)
# param_1 = obj.f(prefix,suffix)
# 527. Word Abbreviation

class Solution:
    def wordsAbbreviation(self, words: List[str]) -> List[str]:
        groups = collections.defaultdict(list)
        for index, word in enumerate(words):
            groups[len(word), word[0], word[-1]].append((word, index))
        ans = [None] * len(words)
        Trie = lambda: collections.defaultdict(Trie)
        COUNT = "$"
        for group in groups.values():
            trie = Trie()
            for word, _ in group:
                cur = trie
                for letter in word[1:]:
                    cur[COUNT] = cur.get(COUNT, 0) + 1
                    cur = cur[letter]

            for word, index in group:
                cur = trie
                for i, letter in enumerate(word[1:], 1):
                    if cur[COUNT] == 1:
                        break
                    cur = cur[letter]
                if len(word) - i - 1 > 1:
                    ans[index] = word[:i] + str(len(word) - i - 1) + word[-1]
                else:
                    ans[index] = word
        return ans


# 677. Map Sum Pairs
class MapSum:

    def __init__(self):
        self.trie = {}

    def insert(self, key: str, val: int) -> None:
        trie = self.trie
        for ch in key:
            if ch not in trie:
                trie[ch] = {}
            trie = trie[ch]
        trie["#"] = val

    def suffixSum(self, node):
        ans = 0
        for ch in node:
            if ch=="#":
                ans+=node[ch]
            else:
                ans += self.suffixSum(node[ch])
        return ans
    def sum(self, prefix: str) -> int:
        trie = self.trie
        for ch in prefix:
            if ch not in trie:
                return 0
            else:
                trie = trie[ch]
        return self.suffixSum(trie)



# Your MapSum object will be instantiated and called as such:
# obj = MapSum()
# obj.insert(key,val)
# param_2 = obj.sum(prefix)

class TrieNode:
    def __init__(self):
        self.val = 0
        self.next = [None for _ in range(26)]

class MapSum:
    def __init__(self):
        self.root = TrieNode()
        self.map = {}

    def insert(self, key: str, val: int) -> None:
        delta = val
        if key in self.map:
            delta -= self.map[key]
        self.map[key] = val
        node = self.root
        for c in key:
            if node.next[ord(c) - ord('a')] is None:
                node.next[ord(c) - ord('a')] = TrieNode()
            node = node.next[ord(c) - ord('a')]
            node.val += delta

    def sum(self, prefix: str) -> int:
        node = self.root
        for c in prefix:
            if node.next[ord(c) - ord('a')] is None:
                return 0
            node = node.next[ord(c) - ord('a')]
        return node.val
# 1178. Number of Valid Words for Each Puzzle
# list trie
class Solution:
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        SIZE = 26  # 26 letters in the alphabet
        trie = [[0] * SIZE]  # we use list to mimic the trie tree
        count = [0]  # the number of words ending at node i
        for word in words:
            word = sorted(set(word))
            if len(word) <= 7:  # longer words are never valid
                # insert into trie
                node = 0
                for letter in word:
                    i = ord(letter) - ord('a')
                    if trie[node][i] == 0:  # push empty node
                        trie.append([0] * SIZE)
                        count.append(0)
                        trie[node][i] = len(trie) - 1
                    node = trie[node][i]
                count[node] += 1

        # search for valid words
        def dfs(node, has_first):
            total = count[node] if has_first else 0
            for letter in puzzle:  # catch puzzle from outside environment
                i = ord(letter) - ord('a')
                if trie[node][i]:
                    total += dfs(trie[node][i], has_first or letter == puzzle[0])
            return total

        result = []
        for puzzle in puzzles:
            result.append(dfs(0, False))
        return result
# dict trie
class Solution:
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        ori_trie = {}
        count = defaultdict(int)
        for word in words:
            trie = ori_trie
            word = sorted(set(word))
            if len(word)<=7:
                for letter in word:
                    if letter not in trie:
                        trie[letter] = {}
                    trie = trie[letter]
                if "#" in trie:
                    trie["#"]+=1
                else:
                    trie["#"] = 1
        # print(ori_trie)
        def dfs(trie, has_first):
            if has_first and "#" in trie:
                total = trie["#"]
            else:
                total = 0
            for letter in puzzle:
                if letter in trie:
                    total += dfs(trie[letter], has_first or letter==puzzle[0])
            return total
        res = []
        for puzzle in puzzles:
            res.append(dfs(ori_trie, False))
        return res
# 212. Word Search II
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        ans = []
        trie = {}
        m, n = len(board), len(board[0])

        for word in words:
            node = trie
            for c in word:
                if c not in node:
                    node[c] = {}
                node = node[c]
            node['$'] = word

        def backtrack(r, c, trie):

            ch = board[r][c]
            board[r][c] = '#'
            # if ch not in trie:
            #     return

            wordMatch = trie[ch].pop('$', False)
            if wordMatch:
                ans.append(wordMatch)

            for nr, nc in [[r + 1, c], [r - 1, c], [r, c + 1], [r, c - 1]]:
                if 0 <= nr < m and 0 <= nc < n and board[nr][nc] in trie[ch]:
                    backtrack(nr, nc, trie[ch])
            board[r][c] = ch
            if not trie[ch]:
                trie.pop(ch)

        for i in range(m):
            for j in range(n):
                if board[i][j] in trie:
                    backtrack(i, j, trie)
        return ans
