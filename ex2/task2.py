import numpy as np
import matplotlib.pyplot as plt
import random
import time

k = []
j = []
l = []

for i in range (10, 3000000, 200000):
    a = np.random.randint(0, i, i) 
    start = time.time()
    np.sort(a)
    end = time.time() - start
    k.append(end)   
    l.append(i)



for i in range (10, 3000000, 200000):
    b = []
    for i2 in range (1, i):
        b.append(random.random())
    start = time.time()
    b.sort()
    end = time.time() - start
    j.append(end)

plt.plot(l, k, 'r')
plt.plot(l, j, 'b')
plt.show()