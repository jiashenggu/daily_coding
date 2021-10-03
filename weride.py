# Enter your code here. Read input from STDIN. Print output to STDOUT
import copy

n, k = (int(x) for x in input().split())
# print(n, k)
maps = []
for i in range(n):
    row = input()
    # print(row)
    row = [int(x) for x in row]
    maps.append(row)

# print(maps)
ops = [[0 for __ in range(2)] for _ in range(10)]
# print(ops)
for j in range(k):
    operation = input().split()
    num = int(operation[0])
    d = operation[1]
    print(num, d)
    mapsc = [[0 for _ in range(n)] for __ in range(n)]


    def mov(num, d, maps):
        for r in range(n):
            for c in range(n):
                if maps[r][c] == num:
                    maps[r][c] = 0
                    if d == 'L':
                        if c - 1 >= 0 and maps[r][c - 1] in [0, num]:
                            mapsc[r][c - 1] = num
                        else:
                            return maps
                    elif d == 'R':
                        if c + 1 < n and maps[r][c + 1] in [0, num]:
                            mapsc[r][c + 1] = num
                        else:
                            return maps
                    elif d == 'U':
                        if r - 1 >= 0 and maps[r - 1][c] in [0, num]:
                            mapsc[r - 1][c] = num
                        else:
                            return maps
                    elif d == 'D':
                        if r + 1 < n and maps[r + 1][c] in [0, num]:
                            mapsc[r + 1][c] = num
                        else:
                            return maps

        for r in range(n):
            for c in range(n):
                if mapsc[r][c] != 0:
                    maps[r][c] = num
        return maps


    maps = mov(num, d, maps)
    for line in maps:
        l = [str(x) for x in line]
        l = "".join(l)
        print(l)
# for line in maps:
#     l = [str(x) for x in line]
#     l = "".join(l)
#     print(l)
# print(maps)


# 8 4
# 00000000
# 10044000
# 10040000
# 10040000
# 00440220
# 00040020
# 00044030
# 00000000
# 1 L
# 2 D
# 3 R
# 1 U

# 10 10
# 0000000000
# 0000000000
# 0000000000
# 0000000000
# 5555500000
# 0000003004
# 2220003333
# 2200000000
# 1100000000
# 0111000000
# 5 D
# 5 D
# 4 R
# 3 D
# 5 D
# 4 U
# 2 U
# 4 U
# 5 U
# 2 R                            



