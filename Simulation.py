
# 353. Design Snake Game
class SnakeGame:

    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = food

        self.snake = [(0, 0)]

    def move(self, direction: str) -> int:
        nr, nc = self.snake[-1]
        if direction == 'L':
            nc -= 1
        elif direction == 'R':
            nc += 1
        elif direction == 'U':
            nr -= 1
        elif direction == 'D':
            nr += 1
        if not (0<=nr<self.height and 0<=nc<self.width):
            return -1

        if not self.food or [nr, nc] != self.food[0]:
            self.snake.pop(0)
        else:
            self.food.pop(0)
        if (nr, nc) in self.snake:
            return -1

        self.snake.append((nr, nc))

        return len(self.snake)-1


# Your SnakeGame object will be instantiated and called as such:
# obj = SnakeGame(width, height, food)
# param_1 = obj.move(direction)
# 498. Diagonal Traverse
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        flag = True
        q = deque([(0, 0)])
        m, n = len(mat), len(mat[0])
        vis = [n * [0] for _ in range(m)]
        vis[0][0] = 1
        ans = []
        while q:
            nq = len(q)
            if flag:
                q.reverse()
                for i, j in q:
                    ans.append(mat[i][j])
                q.reverse()
            else:
                for i, j in q:
                    ans.append(mat[i][j])
            for _ in range(nq):
                i, j = q.popleft()
                for ni, nj in [(0, 1), (1, 0)]:
                    ni += i
                    nj += j
                    if 0 <= ni < m and 0 <= nj < n and vis[ni][nj] == 0:
                        vis[ni][nj] = 1
                        q.append((ni, nj))
            flag = not flag
        return ans


class Solution:

    def findDiagonalOrder(self, matrix: List[List[int]]) -> List[int]:

        # Check for an empty matrix
        if not matrix or not matrix[0]:
            return []

        # The dimensions of the matrix
        N, M = len(matrix), len(matrix[0])

        # Incides that will help us progress through
        # the matrix, one element at a time.
        row, column = 0, 0

        # As explained in the article, this is the variable
        # that helps us keep track of what direction we are
        # processing the current diaonal
        direction = 1

        # Final result array that will contain all the elements
        # of the matrix
        result = []

        # The uber while loop which will help us iterate over all
        # the elements in the array.
        while row < N and column < M:

            # First and foremost, add the current element to
            # the result matrix.
            result.append(matrix[row][column])

            # Move along in the current diagonal depending upon
            # the current direction.[i, j] -> [i - 1, j + 1] if
            # going up and [i, j] -> [i + 1][j - 1] if going down.
            new_row = row + (-1 if direction == 1 else 1)
            new_column = column + (1 if direction == 1 else -1)

            # Checking if the next element in the diagonal is within the
            # bounds of the matrix or not. If it's not within the bounds,
            # we have to find the next head.
            if new_row < 0 or new_row == N or new_column < 0 or new_column == M:

                # If the current diagonal was going in the upwards
                # direction.
                if direction:

                    # For an upwards going diagonal having [i, j] as its tail
                    # If [i, j + 1] is within bounds, then it becomes
                    # the next head. Otherwise, the element directly below
                    # i.e. the element [i + 1, j] becomes the next head
                    row += (column == M - 1)
                    column += (column < M - 1)
                else:

                    # For a downwards going diagonal having [i, j] as its tail
                    # if [i + 1, j] is within bounds, then it becomes
                    # the next head. Otherwise, the element directly below
                    # i.e. the element [i, j + 1] becomes the next head
                    column += (row == N - 1)
                    row += (row < N - 1)

                # Flip the direction
                direction = 1 - direction
            else:
                row = new_row
                column = new_column

        return result

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