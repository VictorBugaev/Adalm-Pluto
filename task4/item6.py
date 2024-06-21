
from scipy.fftpack import fft, ifft,  fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt


fc=10 * 2 # Частота косинуса 
fs=32*fc # частота дискретизации, избыточная 
t=np.arange( 0, 2,  1/fs) # длительность сигнала 2 с
x=np.cos(2*np.pi*fc*t) # формирование временного сигнала

N=256 # количество точек ДПФ

#X = fft(x,N)/N # вычисление ДПФ и нормирование на N
X = np.array([0,0,1])
x_ifft = N*ifft(X,N)
t = np.arange(0, len(x_ifft))/fs
plt.plot(t ,np.real(x_ifft ), 'y')
plt.xlabel('c')
plt.ylabel('$x[n]$')