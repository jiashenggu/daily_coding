# DFS
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        edges = collections.defaultdict(list)
        vis = [0]*numCourses
        self.valid = True
        res = []
        for info in prerequisites:
            edges[info[1]].append(info[0])
        def dfs(u):
            vis[u] = 1
            for v in edges[u]:
                if vis[v]==0:
                    dfs(v)
                    if not self.valid:
                        return
                elif vis[v]==1:
                    self.valid = False
                    return
            vis[u]=2
            res.append(u)
        for i in range(numCourses):
            if self.valid and vis[i]:
                dfs(i)
        return self.valid
# BFS
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        edges = collections.defaultdict(list)
        indeg = [0] * numCourses

        for info in prerequisites:
            edges[info[1]].append(info[0])
            indeg[info[0]] += 1

        q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])
        visited = 0
        res = [u for u in range(numCourses) if indeg[u] == 0]
        while q:
            visited += 1
            u = q.popleft()
            for v in edges[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
                    res.append(v)

        return visited == numCourses

#III
# maximum heap
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses.sort(key = lambda x: x[1])
        sumTime = 0
        maxTime = []
        for cost, deadline in courses:
            sumTime += cost
            heapq.heappush(maxTime, -cost)
            if sumTime>deadline:
                sumTime += heapq.heappop(maxTime)
        return len(maxTime)

# IV
# floyd
class Solution:
    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        dp = [ numCourses*[False] for _ in range(numCourses)]
        for info in prerequisites:
            dp[info[0]][info[1]]  = True
        for k in range(numCourses):
            for i in range(numCourses):
                for j in range(numCourses):
                    if dp[i][k] and dp[k][j]:
                        dp[i][j]=True
        ans = []
        for u,v in queries:
            ans.append(dp[u][v])
        return ans
