import numpy as np

from scipy import integrate

from numba import jit

import time

dims = 4


@jit
def func(*args):
    x = np.array(args)
    return np.exp((-np.dot(x, x) / 2)) / np.sqrt((2 * np.pi) ** dims)



start = time.time()
integ = integrate.nquad(func, [[-np.inf, np.inf] for _ in range(dims)], full_output=True)
print(integ)
print(time.time() - start)
