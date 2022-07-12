# 473. Matchsticks to Square
class Solution:
    def makesquare(self, matchsticks: List[int]) -> bool:
        s = sum(matchsticks)
        a, b, c, d = 0, 0, 0, 0
        n = len(matchsticks)
        if s // 4 * 4 != s:
            return False
        matchsticks.sort(reverse=True)

        @cache
        def dfs(a, b, c, d, idx):
            if idx == n:
                if a == b == c == d == s // 4:
                    return True
                return False
            if a > s // 4 or b > s // 4 or c > s // 4 or d > s // 4:
                return False

            if dfs(a + matchsticks[idx], b, c, d, idx + 1):
                return True
            if dfs(a, b + matchsticks[idx], c, d, idx + 1):
                return True
            if dfs(a, b, c + matchsticks[idx], d, idx + 1):
                return True
            if dfs(a, b, c, d + matchsticks[idx], idx + 1):
                return True

        return dfs(a, b, c, d, 0)
# 44. Wildcard Matching
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        s_len, p_len = len(s), len(p)
        s_idx = p_idx = 0
        star_idx = s_tmp_idx = -1

        while s_idx < s_len:
            # If the pattern caracter = string character
            # or pattern character = '?'
            if p_idx < p_len and p[p_idx] in ['?', s[s_idx]]:
                s_idx += 1
                p_idx += 1

            # If pattern character = '*'
            elif p_idx < p_len and p[p_idx] == '*':
                # Check the situation
                # when '*' matches no characters
                star_idx = p_idx
                s_tmp_idx = s_idx
                p_idx += 1

            # If pattern character != string character
            # or pattern is used up
            # and there was no '*' character in pattern
            elif star_idx == -1:
                return False

            # If pattern character != string character
            # or pattern is used up
            # and there was '*' character in pattern before
            else:
                # Backtrack: check the situation
                # when '*' matches one more character
                p_idx = star_idx + 1
                s_idx = s_tmp_idx + 1
                s_tmp_idx = s_idx

        # The remaining characters in the pattern should all be '*' characters
        return all(p[i] == '*' for i in range(p_idx, p_len))

# 52. N-Queens II
class Solution:
    def totalNQueens(self, n: int) -> int:

        def backtrack(row, diags, anti_diags, cols):
            if row == n:
                return 1
            ans = 0
            for col in range(n):
                curr_diag = row - col
                curr_anti_diag = row + col
                if col in cols or curr_diag in diags or curr_anti_diag in anti_diags:
                    continue
                diags.add(curr_diag)
                anti_diags.add(curr_anti_diag)
                cols.add(col)

                ans += backtrack(row + 1, diags, anti_diags, cols)

                diags.remove(curr_diag)
                anti_diags.remove(curr_anti_diag)
                cols.remove(col)
            return ans

        return backtrack(0, set(), set(), set())


# 51. N-Queens
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def create_board(state):
            board = []
            for row in state:
                board.append("".join(row))
            return board

        def backtrack(row, diagonals, anti_diagonals, cols, state):
            if row == n:
                ans.append(create_board(state))
                return

            for col in range(n):
                curr_diagonal = row - col
                curr_anti_diagonal = row + col
                if (col in cols
                        or curr_diagonal in diagonals
                        or curr_anti_diagonal in anti_diagonals):
                    continue

                cols.add(col)
                diagonals.add(curr_diagonal)
                anti_diagonals.add(curr_anti_diagonal)
                state[row][col] = 'Q'

                backtrack(row + 1, diagonals, anti_diagonals, cols, state)

                cols.remove(col)
                diagonals.remove(curr_diagonal)
                anti_diagonals.remove(curr_anti_diagonal)
                state[row][col] = '.'

        ans = []
        empty_board = [['.'] * n for _ in range(n)]
        backtrack(0, set(), set(), set(), empty_board)
        return ans
# 131. Palindrome Partitioning
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        self.ans = []
        n = len(s)

        def check(s):
            if not s:
                return False
            l, r = 0, len(s) - 1
            while l < r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        def backtrack(idx, string, tmp):
            # print(idx, string, tmp)
            if n == idx:
                if check(string):
                    tmp.append(string)
                    self.ans.append(tmp[:])
                    tmp.pop()
                return
            backtrack(idx + 1, string + s[idx], tmp)
            if check(string):
                tmp.append(string)
                backtrack(idx + 1, s[idx], tmp)
                tmp.pop()

        backtrack(0, "", [])
        return self.ans


class Solution:
    def partition(self, s: str) -> List[List[str]]:
        self.ans = []
        n = len(s)
        dp = [n * [False] for _ in range(n)]

        def backtrack(idx, tmp):
            # print(idx, string, tmp)
            if n == idx:
                self.ans.append(tmp[:])
                return
            for end in range(idx, n):
                if s[idx] == s[end] and (end - idx <= 2 or dp[idx + 1][end - 1]):
                    dp[idx][end] = True
                    tmp.append(s[idx: end + 1])
                    backtrack(end + 1, tmp)
                    tmp.pop()

        backtrack(0, [])
        return self.ans


# 37. Sudoku Solver
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        self.solved = False

        def box_index(row, col):
            return row // 3 * 3 + col // 3

        def could_place(d, row, col):
            return not (d in rows[row] or d in cols[col] or d in boxes[box_index(row, col)])

        def place_number(d, row, col):
            rows[row][d] += 1
            cols[col][d] += 1
            boxes[box_index(row, col)][d] += 1
            board[row][col] = str(d)

        def remove_number(d, row, col):
            del rows[row][d]
            del cols[col][d]
            del boxes[box_index(row, col)][d]
            board[row][col] = '.'

        def backtrack(row, col):
            if board[row][col] == '.':
                for d in range(1, 10):
                    if could_place(d, row, col):
                        place_number(d, row, col)
                        if row == 8 and col == 8:
                            self.solved = True
                        else:
                            if col == 8:
                                backtrack(row + 1, 0)
                            else:
                                backtrack(row, col + 1)
                        if not self.solved:
                            remove_number(d, row, col)
            else:
                if row == 8 and col == 8:
                    self.solved = True
                else:
                    if col == 8:
                        backtrack(row + 1, 0)
                    else:
                        backtrack(row, col + 1)

        rows = [collections.defaultdict(int) for _ in range(9)]
        cols = [collections.defaultdict(int) for _ in range(9)]
        boxes = [collections.defaultdict(int) for _ in range(9)]
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    d = int(board[i][j])
                    place_number(d, i, j)
        backtrack(0, 0)
        return self.solved


# 140. Word Break II
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        self.trie = {}
        for word in wordDict:
            node = self.trie
            for ch in word:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            node["#"] = word
        n = len(s)
        self.ans = []

        def dfs(index, trie, words):
            if index == n:
                length = sum([len(word) for word in words])
                tmp = " ".join(words)
                if length == n:
                    self.ans.append(tmp)
                return
            if s[index] in trie:
                node = trie[s[index]]
            else:
                return
            if "#" in node:
                dfs(index + 1, self.trie, words + [node["#"]])
            dfs(index + 1, node, words)

        dfs(0, self.trie, [])
        return self.ans


# 282. Expression Add Operators
class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        N = len(num)
        ans = []

        def recurse(index, pre, cur, val, s):
            if index == N:
                if val == target and cur == 0:
                    ans.append("".join(s[1:]))
                return
            cur = cur * 10 + int(num[index])
            str_op = str(cur)

            if cur > 0:
                recurse(index + 1, pre, cur, val, s)
            s.append('+')
            s.append(str_op)
            recurse(index + 1, cur, 0, val + cur, s)
            s.pop()
            s.pop()

            if s:
                s.append('-')
                s.append(str_op)
                recurse(index + 1, -cur, 0, val - cur, s)
                s.pop()
                s.pop()

                s.append('*')
                s.append(str_op)
                recurse(index + 1, cur * pre, 0, val - pre + pre * cur, s)
                s.pop()
                s.pop()

        recurse(0, 0, 0, 0, [])
        return ans

# 1286. Iterator for Combination
class CombinationIterator:
    def __init__(self, characters: str, combinationLength: int):
        self.n = n = len(characters)
        self.k = k = combinationLength
        self.chars = characters

        # generate first bitmask 1(k)0(n - k)
        self.b = (1 << n) - (1 << n - k)

    def next(self) -> str:
        # convert bitmasks into combinations
        # 111 --> "abc", 000 --> ""
        # 110 --> "ab", 101 --> "ac", 011 --> "bc"
        curr = [self.chars[j] for j in range(self.n) if self.b & (1 << self.n - j - 1)]

        # generate next bitmask
        self.b -= 1
        while self.b > 0 and bin(self.b).count('1') != self.k:
            self.b -= 1

        return ''.join(curr)

    def hasNext(self) -> bool:
        return self.b > 0
    # 2048. Next Greater Numerically Balanced Number
class Solution:
    def nextBeautifulNumber(self, n: int) -> int:
        def backtrack(idx, numLen, curNum, counter):
            if idx == numLen:
                isB = True
                for k, v in counter.items():
                    if v != 0 and k != v:
                        isB = False
                if isB:
                    yield curNum

            for d in range(1, 10):
                if counter[d] >= d:
                    continue
                if counter[d] + (numLen - idx) < d:
                    continue
                counter[d] += 1
                yield from backtrack(idx + 1, numLen, curNum * 10 + d, counter)
                counter[d] -= 1

        nn = len(str(n))
        for numLen in range(nn, nn + 2):
            nums = backtrack(0, numLen, 0, Counter())
            for num in nums:
                if num > n:
                    return num


# 638. Shopping Offers
from functools import lru_cache

class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        n = len(price)

        # 过滤不需要计算的大礼包，只保留需要计算的大礼包
        filter_special = []
        for sp in special:
            if sum(sp[i] for i in range(n)) > 0 and sum(sp[i] * price[i] for i in range(n)) > sp[-1]:
                filter_special.append(sp)

        # 记忆化搜索计算满足购物清单所需花费的最低价格
        @lru_cache(None)
        def dfs(cur_needs):
            # 不购买任何大礼包，原价购买购物清单中的所有物品
            min_price = sum(need * price[i] for i, need in enumerate(cur_needs))
            for cur_special in filter_special:
                special_price = cur_special[-1]
                nxt_needs = []
                for i in range(n):
                    if cur_special[i] > cur_needs[i]:  # 不能购买超出购物清单指定数量的物品
                        break
                    nxt_needs.append(cur_needs[i] - cur_special[i])
                if len(nxt_needs) == n:  # 大礼包可以购买
                    min_price = min(min_price, dfs(tuple(nxt_needs)) + special_price)
            return min_price

        return dfs(tuple(needs))

# 282. Expression Add Operators
class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        n = len(num)
        ans = []
        def backtrack(expr, i, res, mul):
            if i==n:
                if res==target:
                    ans.append("".join(expr))
                return
            signIndex = len(expr)
            if i>0:
                expr.append("")
            val = 0
            for j in range(i, n):
                if j>i and num[i]=='0':
                    break
                val = val*10 + int(num[j])
                expr.append(num[j])
                if i==0:
                    backtrack(expr, j+1, val, val)
                else:
                    expr[signIndex] = '+'
                    backtrack(expr, j+1, res+val, val)
                    expr[signIndex] = '-'
                    backtrack(expr, j+1, res-val, -val)
                    expr[signIndex] = '*'
                    backtrack(expr, j+1, res - mul + mul*val, mul*val)
            del expr[signIndex:]
        backtrack([], 0, 0, 0)
        return ans

class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        n = len(num)
        ans = []

        def recursion(idx, pre, cur, value, string):
            if idx == n:
                if value == target and cur == 0:
                    ans.append("".join(string[1:]))
                return

            cur = cur * 10 + int(num[idx])
            str_op = str(cur)
            if cur > 0:
                recursion(idx + 1, pre, cur, value, string)

            string.append('+')
            string.append(str_op)
            recursion(idx + 1, cur, 0, value + cur, string)
            string.pop()
            string.pop()

            if string:
                string.append('-')
                string.append(str_op)
                recursion(idx + 1, -cur, 0, value - cur, string)
                string.pop()
                string.pop()

                string.append('*')
                string.append(str_op)
                recursion(idx + 1, pre * cur, 0, value - pre + pre * cur, string)
                string.pop()
                string.pop()

        recursion(0, 0, 0, 0, [])
        return ans






