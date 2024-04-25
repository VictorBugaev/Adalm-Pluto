
from scipy.fftpack import fft, ifft,  fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt


fc=10 * 2 # Частота косинуса 
fs=32*fc # частота дискретизации, избыточная 
t=np.arange( 0, 2,  1/fs) # длительность сигнала 2 с
x=np.cos(2*np.pi*fc*t) # формирование временного сигнала


plt.xlabel('$t=nT_s$')
plt.ylabel('$x[n]$') 

N=512 # количество точек ДПФ
X = fft(x,N)/N # вычисление ДПФ и нормирование на N
k = np.arange(0, N)
plt.stem(k,abs(X)) # выводим модуль ДПФ в точках ДПФ

