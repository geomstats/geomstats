import math as _math

from numpy import pi


def comb(n, k):
    return _math.factorial(n) // _math.factorial(k) // _math.factorial(n - k)
