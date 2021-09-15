# Python3 implementation of the approach
def getZarr(s, n, Z):
    L, R, k = 0, 0, 0

    # [L, R] make a window which matches
    # with prefix of s
    for i in range(n):
        # if i>R nothing matches so we will
        # calculate Z[i] using naive way.

        if i > R:
            L, R = i, i

            '''
            R-L = 0 in starting, so it will start
            checking from 0'th index. For example,
            for "ababab" and i = 1, the value of R
            remains 0 and Z[i] becomes 0. For string
            "aaaaaa" and i = 1, Z[i] and R become 5
            '''
            while R < n and s[R - L] == s[R]:
                R += 1
            print('L',L,'R',R)
            Z[i] = R - L
            R -= 1
        else:

            # k = i-L so k corresponds to number
            # which matches in [L, R] interval.
            k = i - L
            print("i", i, 'L', L, 'R', R)
            print(k, Z[k])

            # if Z[k] is less than remaining interval
            # then Z[i] will be equal to Z[k].
            # For example, str = "ababab", i = 3, R = 5
            # and L = 2
            if Z[k] < R - i + 1:
                Z[i] = Z[k]
            else:
                L = i
                while R < n and s[R - L] == s[R]:
                    R += 1
                Z[i] = R - L
                R -= 1
        print("Z", Z)


def sumSimilarities(s, n):
    Z = [0 for i in range(n)]

    # Compute the Z-array for the
    # given string
    getZarr(s, n, Z)

    total = n

    # summation of the Z-values
    for i in range(n):
        total += Z[i]
    return total


# Driver Code
s = "ababa"

n = len(s)

print(sumSimilarities(s, n))

# This code is contributed
# by Mohit kumar 29
