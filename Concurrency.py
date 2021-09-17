import threading
from collections import deque


class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        def addChildren(url, q):
            urls = htmlParser.getUrls(url)
            q += urls

        def bfs(url, parser):
            q = deque()
            visited = set()
            hostname = url.split('/')[2]
            q.append(url)
            res = []
            while q:
                threads = []
                # The current queue length is the same level
                for i in range(len(q)):
                    current = q.popleft()
                    if current in visited:
                        continue
                    visited.add(current)
                    if hostname != current.split('/')[2]:
                        continue
                    res.append(current)
                    t = threading.Thread(target=addChildren, args=(current, q))
                    threads.append(t)
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            return res

        res = bfs(startUrl, htmlParser)
        return res