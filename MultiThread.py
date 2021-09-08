1242. Web Crawler Multithreaded
生产者和消费者模式
# """
# This is HtmlParser's API interface.
# You should not implement it, or speculate about its implementation
# """
# class HtmlParser(object):
#    def getUrls(self, url):
#        """
#        :type url: str
#        :rtype List[str]
#        """
import queue
import threading
from urllib.parse import urlsplit


class Solution:

    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        domain = urlsplit(startUrl).netloc

        requestQueue = queue.Queue()
        resultQueue = queue.Queue()
        requestQueue.put(startUrl)
        for _ in range(5):
            t = threading.Thread(target=self._crawl,
                                 args=(domain, htmlParser, requestQueue, resultQueue))
            t.daemon = True
            t.start()
        running = 1
        vis = set([startUrl])

        while running > 0:
            urls = resultQueue.get()
            for url in urls:
                if url in vis:
                    continue
                vis.add(url)
                requestQueue.put(url)
                running += 1
            running -= 1
        return list(vis)

    def _crawl(self, domain, htmlParser, requestQueue, resultQueue):
        while True:
            url = requestQueue.get()
            urls = htmlParser.getUrls(url)
            newUrls = []
            for url in urls:
                u = urlsplit(url).netloc
                if u == domain:
                    newUrls.append(url)
            resultQueue.put(newUrls)

class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        from threading import Lock, Thread

        def get_hostname(url: str) -> str:
            return url.split('//', 1)[1].split('/', 1)[0]

        def fetch(url: str) -> None:
            for url in htmlParser.getUrls(url):
                if get_hostname(url) == hostname:
                    with lock:
                        if url in visited:
                            continue
                        visited.add(url)
                    thread = Thread(target=fetch, args=(url,))
                    thread.start()
                    queue.append(thread)

        hostname = get_hostname(startUrl)
        lock = Lock()
        visited = {startUrl}
        main_thread = Thread(target=fetch, args=(startUrl,))
        main_thread.start()
        queue = deque([main_thread])
        while queue:
            queue.popleft().join()
        return list(visited)
