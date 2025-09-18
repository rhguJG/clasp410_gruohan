#!/usr/bin/env python3

'''
Series of simple examples for Lecture 2 about turkeys
'''

import numpy as np
import matplotlib.pyplot as plt

dx = 0.5
x = np.arange(0, 6 * np.pi, dx)
sinx = np.sin(x)
cosx = np.cos(x) # Analytical solution!


# The hard way
# fwd_diff = np.zeros(x.size - 1)
# for i in range (x.size - 1):
#     fwd_diff[i] = x[i + 1] - x[i]


fwd_diff = (sinx[1:] - sinx[:-1]) / dx
bkw_diff = (sinx[1:] - sinx[:-1]) / dx

plt.plot(x, cosx, label=r'Analytical derivative of $\sin{x}$')
plt.plot(x, fwd_diff, label='Forward Diff Approx')
plt.plot(x, bkw_diff, label='Backward Diff Approx')
plt.legend(loc='best')

