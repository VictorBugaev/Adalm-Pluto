#!/usr/bin/env python3

from scipy import signal
from scipy.signal import max_len_seq
import numpy as np
import matplotlib.pyplot as plt

import adi 
from scipy.fftpack import fft, ifft,  fftshift, ifftshift


#https://humble-ballcap-e09.notion.site/14-IQ-f05a7383ce384e51a13de2b04708bbc8
data=max_len_seq(5)[0]


print(data)

m=2*data-1

ft = 100e3 # кб в секунду
fs = 900e6 # частота дискретезации

ns = fs / ft # отчетов на символ
b = np.ones(int(ns)) #Коэффициенты фильтра интерполятора
ts1t =np.array([0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1])
ts1t = 2 * ts1t - 1
x_IQ = np.hstack((ts1t,m)) # формирование пакета 
#x_IQ = m


N_input = len(x_IQ)
xup = np.hstack((x_IQ.reshape(N_input,1),np.zeros((N_input, int(ns-1)))))
xup= xup.flatten()
print(xup)
x1 = signal.lfilter(b, 1,xup)
x=x1.astype(complex)  
xt=.5*(1+x) #комплексные отсчеты для adalm

triq=2**14*xt
n_frame= len(triq)


print("n_frame = ", n_frame)

fm = int(fs)
sdr = adi.Pluto("ip:192.168.3.1")
#sdr.sample_rate = int(fs)
sdr.rx_buffer_size = 5000
sdr.rx_lo = fm
sdr.tx_lo = fm


file_name = "output_det.txt"

#Отправка данных
def sendto(data, n):
    for i in range(n):
        sdr.tx(data)
#Чтение данных и их визуализация
def listen_data():
    
    data = sdr.rx()
    plt.ylim(-3000, 3000)
    plt.plot(data)
    #plt.scatter(data.real, data.imag)
    plt.show()
    return data

sdr.rx_rf_bandwidth = 200000
sdr.rx_destroy_buffer()
sdr.tx_hardwaregain_chan0 = -10
#sdr.rx_buffer_size = 2*n_frame
#Запись в файл принятого сигнала
def listen_data_and_write():
    xrec1=sdr.rx()
    np.savetxt(file_name, xrec1, delimiter = ","  )

    
#sendto(triq, 1)
#time.sleep(1)
#listen_data()
listen_data_and_write()

    
    














