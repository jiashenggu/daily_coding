# 282. Expression Add Operators
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






