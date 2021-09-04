# 509. Fibonacci Number
import numpy as np
class Solution:
    def fib(self, n: int) -> int:
        MOD = 10 ** 9 + 7
        if n < 2:
            return n

        def multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
            lk = len(a[0])
            # c = [[0, 0], [0, 0]]
            c = [ [0]*lk for _ in range(lk) ]
            for k in range(2):
                for i in range(2):
                    for j in range(2):
                        c[i][j] = (c[i][j] + (a[i][k] * b[k][j]) % MOD)%MOD
            return c

        def matrix_pow(a: List[List[int]], n: int) -> List[List[int]]:
            ret = [[1, 0], [0, 1]]
            while n > 0:
                if n & 1:
                    ret = multiply(ret, a)
                n >>= 1
                a = multiply(a, a)
            return ret

        res = matrix_pow([[1, 1], [1, 0]], n - 1)
        return res[0][0]