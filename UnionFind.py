# 952. Largest Component Size by Common Factor
class Solution:
    """ slower than the enumeration of all factors ?!
    """

    def largestComponentSize(self, A: List[int]) -> int:

        dsu = DisjointSetUnion(max(A))
        num_factor_map = {}

        for num in A:
            prime_factors = list(set(self.primeDecompose(num)))
            # map a number to its first prime factor
            num_factor_map[num] = prime_factors[0]
            # merge all groups that contain the prime factors.
            for i in range(0, len(prime_factors) - 1):
                dsu.union(prime_factors[i], prime_factors[i + 1])

        max_size = 0
        group_count = defaultdict(int)
        for num in A:
            group_id = dsu.find(num_factor_map[num])
            group_count[group_id] += 1
            max_size = max(max_size, group_count[group_id])

        return max_size

    def primeDecompose(self, num):
        """ decompose any positive number into
                a series of prime factors.
            e.g. 12 = 2 * 2 * 3
        """
        factor = 2
        prime_factors = []
        while num >= factor * factor:
            if num % factor == 0:
                prime_factors.append(factor)
                num = num // factor
            else:
                factor += 1
        prime_factors.append(num)
        return prime_factors


class DisjointSetUnion(object):

    def __init__(self, size):
        # initially, each node is an independent component
        self.parent = [i for i in range(size + 1)]
        # keep the size of each component
        self.size = [1] * (size + 1)

    def find(self, x):
        """ return the component id that the element x belongs to. """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """ merge the two components that x, y belongs to respectively,
              and return the merged component id as the result.
        """
        px, py = self.find(x), self.find(y)

        # the two nodes share the same set
        if px == py:
            return px

        # otherwise, connect the two sets (components)
        if self.size[px] > self.size[py]:
            # add the node to the union with less members.
            # keeping px as the index of the smaller component
            px, py = py, px
        # add the smaller component to the larger one
        self.parent[px] = py
        self.size[py] += self.size[px]
        # return the final (merged) group
        return py

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




