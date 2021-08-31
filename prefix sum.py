#lc 1109
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        diff = [0]*n
        for first, last, seats in bookings:
            diff[first-1]+=seats
            if last==n:
                continue
            diff[last]-=seats
        ans = []
        s = 0
        for d in diff:
            s+=d
            ans.append(s)
        return ans