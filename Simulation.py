# 592. Fraction Addition and Subtraction
class Solution:
    def fractionAddition(self, expression: str) -> str:

        sign = 1
        flag = True

        pre_numerator = float("inf")
        pre_denominator = float("inf")
        if expression[0].isdigit():
            expression = '+' + expression
        n = len(expression)
        i = 0
        numerator = 0
        denominator = 0
        while i <= n:
            if i < n:
                c = expression[i]
            else:
                c = '#'
            if c.isdigit():
                if flag:
                    numerator = numerator * 10 + int(c)
                else:
                    denominator = denominator * 10 + int(c)
            elif i == n or c in "+-":
                flag = True
                if c == '+':
                    sign = 1
                elif c == '-':
                    sign = -1
                if pre_numerator != float("inf") and pre_denominator != float("inf"):
                    g = gcd(pre_denominator, denominator)
                    pre_numerator = pre_numerator * denominator // g + pre_denominator * numerator // g
                    pre_denominator = pre_denominator * denominator // g
                    g = gcd(pre_numerator, pre_denominator)
                    pre_numerator //= g
                    pre_denominator //= g
                elif numerator or denominator:
                    pre_numerator = numerator
                    pre_denominator = denominator
                numerator = 0
                denominator = 0
            elif c == '/':
                flag = False
                numerator *= sign
            i += 1
        return str(pre_numerator) + '/' + str(pre_denominator)


# 43. Multiply Strings
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"

        # Reverse both numbers.
        first_number = num1[::-1]
        second_number = num2[::-1]

        # For each digit in second_number, multipy the digit by first_number and then
        # store the multiplication result (reversed) in the results array.
        results = []
        for index, digit in enumerate(second_number):
            results.append(self.multiply_one_digit(digit, index, first_number))

        # Add all of the results together to get our final answer (in reverse order)
        answer = self.sum_results(results)

        # Reverse answer and join the digits to get the final answer.
        return ''.join(str(digit) for digit in reversed(answer))

    def multiply_one_digit(self, digit2: str, num_zeros: int, first_number: List[str]) -> List[int]:
        """Multiplies first_number by a digit from second_number (digit2)."""
        # Insert zeros at the beginning of the current result based on the current digit's place.
        current_result = [0] * num_zeros
        carry = 0

        # Multiply each digit in first_number with the current digit of the second_number.
        for digit1 in first_number:
            multiplication = int(digit1) * int(digit2) + carry
            # Set carry equal to the tens place digit of multiplication.
            carry = multiplication // 10
            # Append last digit to the current result.
            current_result.append(multiplication % 10)

        if carry != 0:
            current_result.append(carry)
        return current_result

    def sum_results(self, results: List[List[int]]) -> List[int]:
        # Initialize answer as a number from results.
        answer = results.pop()

        # Add each result to answer one at a time.
        for result in results:
            new_answer = []
            carry = 0

            # Sum each digit from answer and result. Note: zip_longest is the
            # same as zip, except that it pads the shorter list with fillvalue.
            for digit1, digit2 in zip_longest(result, answer, fillvalue=0):
                # Add current digit from both numbers.
                curr_sum = digit1 + digit2 + carry
                # Set carry equal to the tens place digit of curr_sum.
                carry = curr_sum // 10
                # Append the ones place digit of curr_sum to the new answer.
                new_answer.append(curr_sum % 10)

            if carry != 0:
                new_answer.append(carry)

            # Update answer to new_answer which equals answer + result
            answer = new_answer

        return answer
# 54. Spiral Matrix
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        vis = [n * [0] for i in range(m)]
        direction = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        r, c = 0, 0
        ans = []
        d = 0
        for i in range(m * n):

            vis[r][c] = 1
            ans.append(matrix[r][c])
            if i == m * n - 1:
                break
            nr = r + direction[d][0]
            nc = c + direction[d][1]
            while not (0 <= nr < m and 0 <= nc < n and vis[nr][nc] == 0):
                d = (d + 1) % 4
                nr = r + direction[d][0]
                nc = c + direction[d][1]

            r = nr
            c = nc
        return ans


# 722. Remove Comments
class Solution:
    def removeComments(self, source: List[str]) -> List[str]:
        in_block = False
        ans = []
        for line in source:
            i = 0
            n = len(line)
            if not in_block:
                newline = []
            while i<n:
                if line[i:i+2] == "/*" and not in_block:
                    in_block = True
                    i+=1
                elif  line[i:i+2] == "*/" and in_block:
                    in_block = False
                    i+=1
                elif not in_block and line[i:i+2]=="//":
                    break
                elif not in_block:
                    newline.append(line[i])
                i+=1
            if newline and not in_block:
                finalline = "".join(newline)
                ans.append(finalline)
        return ans