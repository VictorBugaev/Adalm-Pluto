import numpy as np
import matplotlib as plt
import random
import time

a = np.random.randint(0, 100, 3000000)
b = np.random.randint(0, 100, 3000000)

start = time.time()
np.sort(a)
end1 = time.time() - start
start = time.time()
b.sort()
end2 = time.time() - start


print("NumPY - ",end1)
print("Lists - ",end2)