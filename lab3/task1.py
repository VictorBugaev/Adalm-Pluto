import math
import matplotlib.pyplot as plt  
a = []
b = []
tt = []
A = 111
w = 1
f = 2
t = 3

for i in range (0, 1000):
    tt.append(i)
    a.append(A *math.sin(w * f * i))
    b.append(A *math.cos(w * f * i))
plt.title("A * sin(w*f*t)")
plt.xlabel("Время")
plt.ylabel("Амплитуда")
plt.plot(tt, a)
plt.plot(tt, b)

plt.show()
