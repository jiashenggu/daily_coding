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
