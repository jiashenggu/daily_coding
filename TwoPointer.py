# 443. String Compression
class Solution:
    def compress(self, chars: List[str]) -> int:
        n = len(chars)
        if n == 1:
            return n
        cnt = 1
        ans = 0
        start = 0
        for i in range(1, n):
            if chars[i - 1] == chars[i]:
                cnt += 1
            else:
                if cnt == 1:
                    ans += 1
                    chars[start] = chars[i - 1]
                    start += 1
                else:
                    ccnt = 0
                    chars[start] = chars[i - 1]
                    q = []
                    while cnt:
                        q.append(str(cnt % 10))
                        cnt //= 10
                        ccnt += 1
                    while q:
                        # print(q)
                        start += 1
                        chars[start] = q.pop(-1)
                    start += 1
                    ans += 1 + ccnt
                cnt = 1

        if cnt == 1:
            ans += 1
            chars[start] = chars[i]
            start += 1
        else:
            ccnt = 0
            chars[start] = chars[i]
            q = []
            while cnt:
                q.append(str(cnt % 10))
                cnt //= 10
                ccnt += 1
            while q:
                # print(q)
                start += 1
                chars[start] = q.pop(-1)
            start += 1
            ans += 1 + ccnt

        return ans


# 1868. Product of Two Run-Length Encoded Arrays
class Solution:
    def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
        # two pointers
        p1, p2 = 0, 0
        ans = []
        while p1 < len(encoded1) and p2 < len(encoded2):
            l1 = encoded1[p1][1]
            l2 = encoded2[p2][1]
            current_v = encoded1[p1][0] * encoded2[p2][0]

            if l1 < l2:
                current_f = l1
                p1 += 1
                encoded2[p2][1] = l2 - l1  # update frequency for uncounted encoded2
            elif l2 < l1:
                current_f = l2
                p2 += 1
                encoded1[p1][1] = l1 - l2  # update frequency for uncounted encoded1
            else:  # l1 == l2
                current_f = l2
                p1 += 1
                p2 += 1

            # update ans
            if ans and ans[-1][0] == current_v:  # check if current_v == previous value in ans
                ans[-1][1] += current_f
            else:
                ans.append([current_v, current_f])
        return ans