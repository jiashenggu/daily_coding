# Python program to print prime factors
import math
# A function to print all prime factors of
# a given number n
def primeFactors(n):
    # Print the number of two's that divide n
    while n % 2 == 0:
        print(2)
        n = n / 2
    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        # while i divides n, print i ad divide n
        while n % i == 0:
            print(i)
            n = n / i
    # Condition if n is a prime
    # number greater than 2
    if n > 2:
        print(n)
# Eradosse sieve
class Solution:
    def countPrimes(self, n: int) -> int:
        isPrime = n*[1]
        ans = 0
        for i in range(2,n):
            if isPrime[i]:
                ans += 1
                if i*i<n:
                    for j in range(i*i,n,i):
                        isPrime[j]=0
        return ans

# linear sieve
class Solution:
    def countPrimes(self, n: int) -> int:
        isPrime = n*[1]
        primes = []
        for i in range(2,n):
            if isPrime[i]:
                primes.append(i)
            for p in primes:
                if p*i>=n:
                    break
                isPrime[p*i]=0
                if i%p==0:
                    break
        return len(primes)

    [[4, 10], [2, 2], [8, 8], [10, 2], [5, 5], [9, 10], [2, 6]]