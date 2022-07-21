# 92. Reverse Linked List II
class Solution:
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """

        if not head:
            return None

        left, right = head, head
        stop = False
        def recurseAndReverse(right, m, n):
            nonlocal left, stop

            # base case. Don't proceed any further
            if n == 1:
                return

            # Keep moving the right pointer one step forward until (n == 1)
            right = right.next

            # Keep moving left pointer to the right until we reach the proper node
            # from where the reversal is to start.
            if m > 1:
                left = left.next

            # Recurse with m and n reduced.
            recurseAndReverse(right, m - 1, n - 1)

            # In case both the pointers cross each other or become equal, we
            # stop i.e. don't swap data any further. We are done reversing at this
            # point.
            if left == right or right.next == left:
                stop = True

            # Until the boolean stop is false, swap data between the two pointers
            if not stop:
                left.val, right.val = right.val, left.val

                # Move left one step to the right.
                # The right pointer moves one step back via backtracking.
                left = left.next

        recurseAndReverse(right, m, n)
        return head
# 44. Wildcard Matching
class Solution:
    @lru_cache(None)
    def isMatch(self, s: str, p: str) -> bool:
        def remove_duplicate_stars(p: str) -> str:
            new_string = []
            for char in p:
                if not new_string or char != '*':
                    new_string.append(char)
                elif new_string[-1] != '*':
                    new_string.append(char)
            return ''.join(new_string)

        p = remove_duplicate_stars(p)
        m, n = len(s), len(p)
        i = m - 1
        j = n - 1
        while i >= 0 and j >= 0:
            if p[j] == '?' or s[i] == p[j]:
                i -= 1
                j -= 1
            elif p[j] == '*':
                break
            else:
                return False
        i = 0
        j = 0
        if m == 0:
            for pp in p:
                if pp != '*':
                    return False
            return True

        while i < m and j < n:
            if p[j] == '?' or s[i] == p[j]:
                i += 1
                j += 1
            elif p[j] == '*':
                for k in range(i, m):
                    if j == n - 1 or self.isMatch(s[k:], p[j + 1:]):
                        return True
                return False
            elif s[i] != p[j]:
                return False
        while j < n and p[j] == '*':
            j += 1
        return i == m and j == n


def makesquare(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """

    # If there are no matchsticks, then we can't form any square
    if not nums:
        return False

    # Number of matchsticks we have
    L = len(nums)

    # Perimeter of our square (if one can be formed)
    perimeter = sum(nums)

    # Possible side of our square.
    possible_side =  perimeter // 4

    # If the perimeter can be equally split into 4 parts (and hence 4 sides, then we move on).
    if possible_side * 4 != perimeter:
        return False

    # Reverse sort the matchsticks because we want to consider the biggest one first.
    nums.sort(reverse=True)

    # This array represents the 4 sides and their current lengths
    sums = [0 for _ in range(4)]

    # Our recursive dfs function.
    def dfs(index):

        # If we reach the end of matchsticks array, we check if the square was formed or not
        if index == L:
            # If 3 equal sides were formed, 4th will be the same as these three and answer should be True in that case.
            return sums[0] == sums[1] == sums[2] == possible_side

        # The current matchstick can belong to any of the 4 sides (provided their remaining lenghts are >= the size of the current matchstick)
        for i in range(4):
            # If this matchstick can fir in the space left for the current side
            if sums[i] + nums[index] <= possible_side:
                # Recurse
                sums[i] += nums[index]
                if dfs(index + 1):
                    return True
                # Revert the effects of recursion because we no longer need them for other recursions.
                sums[i] -= nums[index]
        return False
    return dfs(0)


def  luckyNumbers(num, target):
    res = []
    n = len(num)
    s = ""
    def dfs(pos, s, cur, tmp):
        if pos==n:
            if cur==target:
                res.append(s)
            return
        for i in range(pos, n):
            if num[pos]=='0' and pos!=i:
                break
            string_num = num[pos:i+1]
            number = int(string_num)
            if pos==0:
                dfs(i+1, s+string_num, number, number)
            else:
                dfs(i+1, s+"+"+string_num, cur + number, number)
                # if "-", tmp should be -number
                dfs(i+1, s+"-"+string_num, cur - number, -number)
                # cur-tmp+tmp*number
                dfs(i+1, s+"*"+string_num, cur-tmp+tmp*number, tmp*number)
                # result should be integer, divided by 0 is not allowed
                if number!=0 and tmp%number==0:
                    dfs(i+1, s+"/"+string_num, cur-tmp+tmp//number, tmp//number)
    dfs(0, s, 0, 0)
    return res