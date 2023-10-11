
from scipy.fftpack import fft, ifft,  fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt


fc=10 * 2 # Частота косинуса 
fs=32*fc # частота дискретизации, избыточная 
t=np.arange( 0, 2,  1/fs) # длительность сигнала 2 с
x=np.cos(2*np.pi*fc*t) # формирование временного сигнала

N=256 # количество точек ДПФ
df=fs/N
k = np.arange(0, N)
kf = k*df
k2 = np.arange(-N/2, N/2)
kf2=k2*df
X = fft(x,N)/N # вычисление ДПФ и нормирование на N
X2 = fftshift(X) # сдвиг ДПФ на центр 
plt.figure(4)
plt.stem(kf2,abs(X2), 'r')


