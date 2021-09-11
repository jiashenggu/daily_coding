# 528. Random Pick with Weight
class Solution:
    def __init__(self, w: List[int]):
        n = len(w)
        s = 0
        self.prefix = []
        for i in range(n):
            s+=w[i]
            self.prefix.append(s)

    def pickIndex(self) -> int:
        r = random.randint(1, self.prefix[-1])
        idx = bisect.bisect_left(self.prefix, r)
        return idx


# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()



# O(1) 时间插入、删除和获取随机元素
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.l = []
        self.d = {}

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """

        if self.d.__contains__(val):
            return False
        else:
            self.l.append(val)
            self.d[val] = len(self.l) - 1
            return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if self.d.__contains__(val):
            t = self.d[val]
            self.d[self.l[-1]] = t
            self.l[t], self.l[-1] = self.l[-1], self.l[t]
            self.l.pop(-1)
            del self.d[val]
            return True
        else:
            return False

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        r = random.randint(0, len(self.l) - 1)
        return self.l[r]

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()