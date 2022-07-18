# 1074. Number of Submatrices That Sum to Target
class Solution:
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
        m, n = len(matrix), len(matrix[0])
        prefix_sum = [n * [0] for _ in range(m)]
        prefix_sum[0][0] = matrix[0][0]
        for i in range(1, m):
            prefix_sum[i][0] = prefix_sum[i - 1][0] + matrix[i][0]
        for i in range(1, n):
            prefix_sum[0][i] = prefix_sum[0][i - 1] + matrix[0][i]
        for i in range(1, m):
            for j in range(1, n):
                prefix_sum[i][j] = prefix_sum[i - 1][j] + prefix_sum[i][j - 1] - prefix_sum[i - 1][j - 1] + matrix[i][j]
        count = 0
        for r1 in range(0, m):
            for r2 in range(r1, m):
                h = defaultdict(int)
                h[0] = 1
                for col in range(n):
                    curr_sum = prefix_sum[r2][col] - (prefix_sum[r1 - 1][col] if r1 > 0 else 0)
                    count += h[curr_sum - target]
                    h[curr_sum] += 1
        return count


#lc 1109
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        diff = [0]*n
        for first, last, seats in bookings:
            diff[first-1]+=seats
            if last==n:
                continue
            diff[last]-=seats
        ans = []
        s = 0
        for d in diff:
            s+=d
            ans.append(s)
        return ans