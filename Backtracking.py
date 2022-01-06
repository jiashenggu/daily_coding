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






