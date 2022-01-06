# 65. Valid Number
class Solution:
    def isNumber(self, s: str) -> bool:
        dfa = [
            {"digit": 1, "sign": 2, "dot": 3},
            {"digit": 1, "dot": 4, "exponent": 5},
            {"digit": 1, "dot": 3},
            {"digit": 4},
            {"digit": 4, "exponent": 5},
            {"sign": 6, "digit": 7},
            {"digit": 7},
            {"digit": 7}
        ]
        cur = 0
        for c in s:
            if c.isdigit():
                group = "digit"
            elif c in "+-":
                group = "sign"
            elif c in "eE":
                group = "exponent"
            elif c == '.':
                group = "dot"
            else:
                return False

            if group not in dfa[cur]:
                return False
            cur = dfa[cur][group]
        return cur in [1, 4, 7]

class Solution:
    def isNumber(self, s: str) -> bool:
        seen_digit = seen_exponent = seen_dot = 0
        for i,c in enumerate(s):
            if c.isdigit():
                seen_digit = True
            elif c in "+-":
                if i>0 and s[i-1] not in 'eE':
                    return False
            elif c in ['e', 'E']:
                if seen_exponent or not seen_digit:
                    return False
                seen_exponent = True
                seen_digit = False
            elif c=='.':
                if seen_dot or seen_exponent:
                    return False
                seen_dot = True
            else:
                return False
        return seen_digit


# 8. String to Integer (atoi)
INT_MAX = 2 ** 31 - 1
INT_MIN = -2 ** 31


class Automaton:
    def __init__(self):
        self.state = 'start'
        self.sign = 1
        self.ans = 0
        self.table = {
            'start': ['start', 'signed', 'in_number', 'end'],
            'signed': ['end', 'end', 'in_number', 'end'],
            'in_number': ['end', 'end', 'in_number', 'end'],
            'end': ['end', 'end', 'end', 'end'],
        }

    def get_col(self, c):
        if c.isspace():
            return 0
        if c == '+' or c == '-':
            return 1
        if c.isdigit():
            return 2
        return 3

    def get(self, c):
        self.state = self.table[self.state][self.get_col(c)]
        if self.state == 'in_number':
            self.ans = self.ans * 10 + int(c)
            self.ans = min(self.ans, INT_MAX) if self.sign == 1 else min(self.ans, -INT_MIN)
        elif self.state == 'signed':
            self.sign = 1 if c == '+' else -1


class Solution:
    def myAtoi(self, str: str) -> int:
        automaton = Automaton()
        for c in str:
            automaton.get(c)
        return automaton.sign * automaton.ans


