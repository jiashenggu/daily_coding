class UnionFind:  # 手写并查集模板
    def __init__(self, n):
        self.parent = [x for x in range(n)]
        self.size = [1 for x in range(n)]
        self.part = 0  # 不同之处。因为点还没有真正加进来

    def Find(self, x: int) -> int:  # 扁平化
        if self.parent[x] != x:
            self.parent[x] = self.Find(self.parent[x])
        return self.parent[x]

    def Union(self, x: int, y: int) -> bool:
        root_x = self.Find(x)
        root_y = self.Find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] > self.size[root_y]:  # size策略
            root_x, root_y = root_y, root_x
        self.parent[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        self.part -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.Find(x) == self.Find(y)


class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        Row, Col = m, n
        UF = UnionFind(Row * Col)
        res = []
        seen = set()  # 在并查中的，已经是陆地的位置
        for r, c in positions:
            ID = r * Col + c
            if ID in seen:  # 样例中有重复
                res.append(UF.part)
                continue
            seen.add(ID)
            UF.part += 1  # 先加上，再考虑合并
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < Row and 0 <= nc < Col:  # 在版图内
                    id2 = nr * Col + nc
                    if id2 in seen:  # 已经是陆地了
                        UF.Union(ID, id2)
            res.append(UF.part)

        return res


class UnionFind:  # 手写并查集模板
    def __init__(self, n):
        self.parent = [x for x in range(n)]
        self.size = [1 for x in range(n)]
        self.part = n  # 不同之处。因为点还没有真正加进来

    def Find(self, x: int) -> int:  # 扁平化
        if self.parent[x] != x:
            self.parent[x] = self.Find(self.parent[x])
        return self.parent[x]

    def Union(self, x: int, y: int) -> bool:
        root_x = self.Find(x)
        root_y = self.Find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] > self.size[root_y]:  # size策略
            root_x, root_y = root_y, root_x
        self.parent[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        self.part -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.Find(x) == self.Find(y)


# 5866. GCD Sort of an Array
# prime unionfind
class Solution:
    def gcdSort(self, nums: List[int]) -> bool:
        nums_c = nums.copy()
        n = len(nums)
        s = 100001 * [0]
        vis = 100001 * [0]
        # vis = defaultdict(int)
        for num in nums:
            s[num] = 1
        uf = UnionFind(100001)
        for i in range(2, 100001):
            if vis[i] == 0:
                for j in range(i, 100001, i):
                    vis[j] = 1
                    if s[j] > 0:
                        uf.Union(i, j)

        nums_c.sort()
        print(nums_c)
        for i in range(n):

            if uf.Find(nums[i]) != uf.Find(nums_c[i]):
                # print(uf.Find(nums[i]), uf.Find(nums_c[i]))
                # print(nums[i], nums_c[i])
                return False
        return True




