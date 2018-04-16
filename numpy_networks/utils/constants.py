import numpy as np

"""
    For use in efficient convolution backprop. See the report for a
   derivation of these.
"""

delta_1 = np.zeros([3, 3, 28, 28, 30, 30])
it = np.nditer(delta_1, flags=['multi_index'])
while not it.finished:
    a, b, j, k, s, t = it.multi_index
    if (j + a) == s and (k + b) == t:
        delta_1[a, b, j, k, s, t] = 1.0
    it.iternext()

delta_2 = np.zeros([3, 3, 14, 14, 16, 16])
it = np.nditer(delta_2, flags=['multi_index'])
while not it.finished:
    a, b, j, k, s, t = it.multi_index
    if (j + a) == s and (k + b) == t:
        delta_2[a, b, j, k, s, t] = 1.0
    it.iternext()
