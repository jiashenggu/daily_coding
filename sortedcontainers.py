from sortedcontainers import SortedDict
class SummaryRanges:

    def __init__(self):
        self.intervals = SortedDict()


    def addNum(self, val: int) -> None:
        _intervals = self.intervals
        keys = _intervals.keys()
        values = _intervals.values()
        intervals1 = _intervals.bisect_right(val)

        intervals0 = len(_intervals) if intervals1==0 else intervals1-1

        if intervals0 != len(_intervals) and keys[intervals0]<=val<=values[intervals0]:
            return
        else:
            leftaside = (intervals0!=len(_intervals)) and (values[intervals0]+1==val)
            rightaside = (intervals1!=len(_intervals)) and (keys[intervals1]-1==val)
            if leftaside and rightaside:
                l, r = keys[intervals0], values[intervals1]
                _intervals.popitem(intervals1)
                _intervals.popitem(intervals0)
                _intervals[l] = r
            elif leftaside:
                _intervals[keys[intervals0]]+=1
            elif rightaside:
                right = values[intervals1]
                _intervals.popitem(intervals1)
                _intervals[val] = right
            else:
                _intervals[val] = val

    def getIntervals(self) -> List[List[int]]:
        return list(self.intervals.items())


# Your SummaryRanges object will be instantiated and called as such:
# obj = SummaryRanges()
# obj.addNum(val)
# param_2 = obj.getIntervals()
