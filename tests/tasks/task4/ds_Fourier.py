#-*-coding:utf-8-*-
"""
CreatedonWedOct417:41:212023

@author:Andrey
"""

from scipy.fftpack import fft,ifft,fftshift
import numpy as np
import matplotlib.pyplot as plt

fc=10#Частотакосинуса
fs=32*fc#частотадискретизации,избыточная
t=np.arange(0,2,1/fs)#длительностьсигнала2с
x=np.cos(2*np.pi*fc*t)#формированиевременногосигнала
plt.figure(1)
plt.plot(t,x)
#выбратьдлительность0.2сек
plt.xlabel('$t=nT_s$')
plt.ylabel('$x[n]$')

#ДалеевычисляетсяДПФдлинойN=256точеквинтервалечастот0−fs.
N=256#количествоточекДПФ
X=fft(x,N)/N#вычислениеДПФинормированиенаN
rt = fs / N
t = [rt * i for i in range(N)]
plt.figure(2)
plt.stem(t, X)