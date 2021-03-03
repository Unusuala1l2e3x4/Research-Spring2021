import time
import datetime as dt
import numpy as np
import pandas as pd





# d = pd.Timestamp(year=2016, month=8)

# d = dt.date(2016, 8, 1)

# d = dt.datetime(2016, 8, 1)


# d = dt.datetime.utcnow().strftime('%Y.%m.%d-%H.%M.%S')
# print(d)


# l = [[1,2,3],[1,2,3],[1,2,3]]

# temp_val = [ sum(a) for a in zip(*l)  ]

# print(temp_val)

# print(list(range(1,12)))


a = np.matrix([[4, 5, 7],[9, 3, 2],[3, 9, 1]])
b = np.matrix([[5, 2, 9],[8, 4, 2],[1, 7, 4]])


print(a)
print(b)
ab = np.multiply(a, b)
print(ab)