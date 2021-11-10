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
