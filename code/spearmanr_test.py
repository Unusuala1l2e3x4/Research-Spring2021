from logging import StrFormatStyle
import time
import datetime as dt
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import copy
import numpy as np
import random

import importlib
fc = importlib.import_module('functions')



from scipy import stats

x2n = [1,2,3,4,5]
y2n = [5,6,7,8,7]



print(stats.spearmanr(x2n, y2n))
# (0.82078268166812329, 0.088587005313543798))

rho, pval = stats.spearmanr(x2n, y2n)
print(rho)
print(pval)


print()
np.random.seed(1234321)
x2n = np.random.randn(100, 2)
y2n = np.random.randn(100, 2)

print(x2n)
print(y2n)



print(stats.spearmanr(x2n))
# (0.059969996999699973, 0.55338590803773591)
print(stats.spearmanr(x2n[:,0], x2n[:,1]))
# (0.059969996999699973, 0.55338590803773591)

rho, pval = stats.spearmanr(x2n, y2n)                   # If axis=0 (default), then each column represents a variable, with observations in the rows
print(rho)
# array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],
#        [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],
#        [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],
#        [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])
print(pval)
# array([[ 0.        ,  0.55338591,  0.06435364,  0.53617935],
#        [ 0.55338591,  0.        ,  0.27592895,  0.80234077],
#        [ 0.06435364,  0.27592895,  0.        ,  0.73039992],
#        [ 0.53617935,  0.80234077,  0.73039992,  0.        ]])

# rho, pval = stats.spearmanr(x2n.T, y2n.T, axis=1)   # If axis=1, the relationship is transposed: each row represents a variable, while the columns contain observations
# print(rho)
# # array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],
# #        [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],
# #        [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],
# #        [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])
# print(pval)
# # array([[ 0.        ,  0.55338591,  0.06435364,  0.53617935],
# #        [ 0.55338591,  0.        ,  0.27592895,  0.80234077],
# #        [ 0.06435364,  0.27592895,  0.        ,  0.73039992],
# #        [ 0.53617935,  0.80234077,  0.73039992,  0.        ]])


# print(stats.spearmanr(x2n, y2n, axis=None))           #  If axis=None, then both arrays will be raveled.
# # (0.10816770419260482, 0.1273562188027364)
# print(stats.spearmanr(x2n.ravel(), y2n.ravel()))
# # (0.10816770419260482, 0.1273562188027364)

# xint = np.random.randint(10, size=(100, 2))
# print(stats.spearmanr(xint))
# (0.052760927029710199, 0.60213045837062351)