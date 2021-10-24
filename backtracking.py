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






