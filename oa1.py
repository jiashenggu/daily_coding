# Problem:
# This problem is a variant of closest pair sum. You'll be given two arrays
# arr1 = { {1, 2000}, {2, 3000}, {3, 4000} }
# arr2 = { { 1, 5000 }, {2, 3000} }
# the first element of every pair represents id and the second value represents the value.
# and a target x = 5000
# Find the pairs from both the arrays whose vaue add upto a sum which is less than given target and should be close to the target.
#
# Output for the above example:
# { {1, 2} } // Note that the output should be in id's

def find(F, B, T):
    ans = [0, 0, 0]
    F = sorted([x, i] for i, x in F)
    for idy, y in B:
        f = 0
        end = len(F)
        z = T - y
        while f != end:
            m = (f + end) // 2
            if F[m][0] <= z:
                f = m + 1
            else:
                end = m
        if f != 0 and y + F[f - 1][0] > ans[0]:
            ans = [y + F[f - 1][0], F[f - 1][1], idy]
    return ans[1:]


print(find([(1, 3000), (2, 5000), (3, 4000), (4, 10000)],
     [(1, 2000), (2, 3000), (3, 4000)], 11000))
Sort one of the arrays of length N. Iterate the other array of length M and do a binary search in the first array updating the global maximum. O(N*log(N) + M*log(N))