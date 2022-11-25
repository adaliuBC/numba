import time
import numpy as np
from numba import jit

def timer(func):
    def timer_decorator(*args, **kargs):
        tStart = time.perf_counter()
        # print(args)
        func(*args, **kargs)
        print(f"Exec time of {func.__class__.__name__} is {time.perf_counter() - tStart}s")
    return timer_decorator

@timer
def pythagorean_triples(max_n=100):
    """Find Pythagorean triples with a <= max_n."""
    triples = []
    for a in range(1, max_n + 1):
        for b in range(a, max_n + 1):
            for c in range(b, max_n + 1):
                if a ** 2 + b ** 2 == c ** 2:
                    triples.append((a, b, c))
    return triples

@timer
@jit
def numba_pythagorean_triples(max_n=100):
    """Find Pythagorean triples with a <= max_n."""
    triples = []
    for a in range(1, max_n + 1):
        for b in range(a, max_n + 1):
            for c in range(b, max_n + 1):
                if a ** 2 + b ** 2 == c ** 2:
                    triples.append((a, b, c))
    return triples

numba_pythagorean_triples(1000)
# tStart = time.perf_counter()
# numba_pythagorean_triples(1000)
# print(f"Exec time of {numba_pythagorean_triples.__class__.__name__} is {time.perf_counter() - tStart}s")

# pythagorean_triples(100)

# @jit
# def add(a, b):
#     return a + b

# add(1, 3)