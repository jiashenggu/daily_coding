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