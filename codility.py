class Solution:
    def alienOrder(self, words: List[str]) -> str:
        wn = len(words)

        indegree = defaultdict(int)  # 入度
        adjvex = defaultdict(set)  # 邻接表
        points = set()  # 所有的点

        # ---- 所有的字母
        for word in words:
            for c in word:
                points.add(c)

        # ---- 建图，构建邻接表
        for i in range(wn):
            wordi = words[i]  # 前面的单词
            for j in range(i + 1, wn):
                wordj = words[j]  # 后面的单词
                for k in range(min(len(wordi), len(wordj))):
                    if wordi[k] != wordj[k]:  # 第一个不同的字母
                        adjvex[wordi[k]].add(wordj[k])
                        break
        # ---- 统计入度
        for x, ys in adjvex.items():
            for y in ys:
                indegree[y] += 1

        res = []
        # ------ 拓扑排序 topsort
        Q = collections.deque()
        for x in points:
            if indegree[x] == 0:
                Q.append(x)
        while Q:
            x = Q.popleft()
            res.append(x)
            for y in adjvex[x]:
                indegree[y] -= 1
                if indegree[y] == 0:
                    Q.append(y)

        # print(adjvex)
        # ----- 看是否输出了所有的点==有没有环
        if len(res) != len(points):  # 有环
            return ""

        # ----- 反过来找。列表的顺序有没有问题
        has_reverse = False
        for i in range(wn):
            if has_reverse == True:
                break
            wordi = words[i]
            for j in range(i + 1, wn):
                wordj = words[j]
                win = len(wordi)
                wjn = len(wordj)
                if win > wjn and wordi[:wjn] == wordj:  # 内容相同时，短的小
                    has_reverse = True
                    break
        if has_reverse == True:
            return ""

        # ----- 最后才通过
        return ''.join(res)


