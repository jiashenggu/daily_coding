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






