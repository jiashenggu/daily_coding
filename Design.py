# 251. Flatten 2D Vector
class Vector2D:

    def __init__(self, vec: List[List[int]]):
        self.vec = []
        for row in vec:
            for val in row:
                self.vec.append(val)
        self.index = 0

    def next(self) -> int:
        ret = self.vec[self.index]
        self.index += 1
        return ret

    def hasNext(self) -> bool:
        if len(self.vec) == self.index:
            return False
        return True


# Your Vector2D object will be instantiated and called as such:
# obj = Vector2D(vec)
# param_1 = obj.next()
# param_2 = obj.hasNext()
class Vector2D:

    def __init__(self, v: List[List[int]]):
        self.vector = v
        self.inner = 0
        self.outer = 0

    # If the current outer and inner point to an integer, this method does nothing.
    # Otherwise, inner and outer are advanced until they point to an integer.
    # If there are no more integers, then outer will be equal to vector.length
    # when this method terminates.
    def advance_to_next(self):
        # While outer is still within the vector, but inner is over the
        # end of the inner list pointed to by outer, we want to move
        # forward to the start of the next inner vector.
        while self.outer < len(self.vector) and self.inner == len(self.vector[self.outer]):
            self.outer += 1
            self.inner = 0

    def next(self) -> int:
        # Ensure the position pointers are moved such they point to an integer,
        # or put outer = vector.length.
        self.advance_to_next()
        # Return current element and move inner so that is after the current
        # element.
        result = self.vector[self.outer][self.inner]
        self.inner += 1
        return result


    def hasNext(self) -> bool:
        # Ensure the position pointers are moved such they point to an integer,
        # or put outer = vector.length.
        self.advance_to_next()
        # If outer = vector.length then there are no integers left, otherwise
        # we've stopped at an integer and so there's an integer left.
        return self.outer < len(self.vector)
# 1166. Design File System

# Dictionary
class FileSystem:

    def __init__(self):
        self.paths = defaultdict()

    def createPath(self, path: str, value: int) -> bool:

        # Step-1: basic path validations
        if path == "/" or len(path) == 0 or path in self.paths:
            return False

        # Step-2: if the parent doesn't exist. Note that "/" is a valid parent.
        parent = path[:path.rfind('/')]
        if len(parent) > 1 and parent not in self.paths:
            return False

        # Step-3: add this new path and return true.
        self.paths[path] = value
        return True

    def get(self, path: str) -> int:
        return self.paths.get(path, -1)
# Trie
class FileSystem:

    def __init__(self):
        self.root = {}

    def createPath(self, path: str, value: int) -> bool:
        components = path.split('/')

        cur = self.root
        for i in range(1, len(components)):
            name = components[i]
            # print(name, i)
            if name not in cur:
                if i == len(components) - 1:
                    cur[name] = {}
                else:
                    return False
            cur = cur[name]
        if "value" in cur:
            return False
        cur["value"] = value
        return True

    def get(self, path: str) -> int:
        components = path.split('/')

        cur = self.root
        for i in range(1, len(components)):
            name = components[i]

            if name not in cur:
                return -1
            cur = cur[name]
        return cur["value"]


# Your FileSystem object will be instantiated and called as such:
# obj = FileSystem()
# param_1 = obj.createPath(path,value)
# param_2 = obj.get(path)

# 380. Insert Delete GetRandom O(1)
class RandomizedSet:

    def __init__(self):
        self.dict = {}
        self.list = []

    def insert(self, val: int) -> bool:
        if val in self.dict:
            return False
        self.dict[val] = len(self.list)
        self.list.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.dict:
            return False
        idx = self.dict[val]
        self.dict[self.list[-1]] = idx
        del self.dict[val]
        self.list[idx], self.list[-1] = self.list[-1], self.list[idx]
        self.list.pop()
        return True

    def getRandom(self) -> int:
        return choice(self.list)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()