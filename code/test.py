import time
import datetime as dt

import pandas as pd





# d = pd.Timestamp(year=2016, month=8)

# d = dt.date(2016, 8, 1)

# d = dt.datetime(2016, 8, 1)


d = dt.datetime.utcnow().strftime('%Y.%m.%d-%H.%M.%S')

print(d)

# for i in range(2, 3):
#   print(i)



# l = [[1,2,3],[1,2,3],[1,2,3]]

# temp_val = [ sum(a) for a in zip(*l)  ]

# print(temp_val)
