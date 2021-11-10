# 1178. Number of Valid Words for Each Puzzle
class Solution:
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:

        def bitmask(word: str) -> int:
            mask = 0
            for letter in word:
                mask |= 1 << (ord(letter) - ord('a'))
            return mask

        # Create a bitmask for each word.
        word_count = Counter(bitmask(word) for word in words)

        result = []
        for puzzle in puzzles:
            first = 1 << (ord(puzzle[0]) - ord('a'))
            count = word_count[first]

            # Make bitmask but ignore the first character since it must always
            # be there.
            mask = bitmask(puzzle[1:])

            # Iterate over every possible subset of characters.
            submask = mask
            while submask:
                # Increment the count by the number of words that match the
                # current submask.
                count += word_count[submask | first]  # add first character
                submask = (submask - 1) & mask
            result.append(count)
        return result
# 782. Transform to Chessboard
class Solution(object):
    def movesToChessboard(self, board):
        N = len(board)
        ans = 0
        # For each count of lines from {rows, columns}...
        for count in (collections.Counter(map(tuple, board)),
                      collections.Counter(zip(*board))):

            # If there are more than 2 kinds of lines,
            # or if the number of kinds is not appropriate ...
            if len(count) != 2 or sorted(count.values()) != [N/2, (N+1)/2]:
                return -1

            # If the lines are not opposite each other, impossible
            line1, line2 = count
            if not all(x ^ y for x, y in zip(line1, line2)):
                return -1

            # starts = what could be the starting value of line1
            # If N is odd, then we have to start with the more
            # frequent element
            starts = [+(line1.count(1) * 2 > N)] if N%2 else [0, 1]

            # To transform line1 into the ideal line [i%2 for i ...],
            # we take the number of differences and divide by two
            ans += min(sum((i-x) % 2 for i, x in enumerate(line1, start))
                       for start in starts) / 2

        return ans
# 371. Sum of Two Integers
MASK1 = 4294967296  # 2^32
MASK2 = 2147483648  # 2^31
MASK3 = 2147483647  # 2^31-1


class Solution:
    def getSum(self, a: int, b: int) -> int:
        ab1 = a^b
        ab2 = (a&b)<<1
        while ab2:
            tmp = ab1
            ab1 = (ab1 ^ (ab2))%MASK1
            ab2 = ((tmp&(ab2))<<1)%MASK1
        if ab1 & MASK2:
            return ~((ab1^MASK2)^MASK3)
        return ab1