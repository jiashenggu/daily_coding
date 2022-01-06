from collections import deque
class Solution:
    def findDiagonalOrder(self, mat):
        flag = True
        q = deque([(0, 0)])
        m, n = len(mat), len(mat[0])
        vis = [n * [0] for _ in range(m)]
        vis[0][0] = 1
        ans = []
        while q:
            nq = len(q)
            print(q)
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


so = Solution()
so.findDiagonalOrder([[1,2,3],[4,5,6],[7,8,9]])

