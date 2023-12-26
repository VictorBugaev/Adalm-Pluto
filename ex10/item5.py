


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import max_len_seq
from scipy.fftpack import fft, ifft,  fftshift, ifftshift
import sys

SDR = False
sample_rate = 1000000

if SDR:
    import adi
    sdr = adi.Pluto('ip:192.168.3.1')
    sdr.sample_rate = sample_rate
    sdr.tx_destroy_buffer()
    
if SDR:
    sdr.rx_lo = 2000000000
    sdr.tx_lo = 2000000000
    sdr.tx_cyclic_buffer = True
    #sdr.tx_cyclic_buffer = False
    sdr.tx_hardwaregain_chan0 = -5
    sdr.gain_control_mode_chan0 = "slow_attack"


fs = sample_rate
rs=100000
ns=fs//rs
 

data=max_len_seq(8)[0] 
data = np.concatenate((data,np.zeros(1)))
 
 
x_ = np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1])
b7=np.array([1,-1,1,1,1,-1,1])
ts1 =np.array([0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1])
ts2 =[0,0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,1]
ts3 =[1,0,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,1]
ts4 =[1,1,1,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,1,0,0]
m=2*data-1
#ts1t=2*ts1-1
ts1t=b7
 

b = np.ones(int(ns))
 
#qpsk
filename_input = "input_qpsk_new_type.txt"
filename_output = "output_qosk_new_type.txt"
  
 
x=np.reshape(m,(2,128))
xi=x[0,:]
xq=x[1,:]
x_bb=(xi+1j*xq)/np.sqrt(2)
plt.figure(1)
plt.scatter(x_bb.real,x_bb.imag)
 
 
xiq=2**14*x_bb
 
n_frame= len(xiq)

if SDR:
    sdr.tx(xiq)


    np.savetxt(filename_input, xiq, delimiter=":")
    
    sdr.rx_rf_bandwidth = 1000000
    sdr.rx_destroy_buffer()
    sdr.rx_hardwaregain_chan0 = -5
    sdr.rx_buffer_size =2*n_frame*4

    xrec1=sdr.rx()
    sdr.tx_destroy_buffer()
else:
    xrec1 = np.loadtxt(filename_output, delimiter=":", dtype=complex)
#np.savetxt(filename_output, xrec1, delimiter=":")
#sys.exit()

xrec = xrec1/np.mean(xrec1**2)
plt.figure(2, figsize=(15, 15))
plt.subplot(2,2,1)
plt.scatter(xrec1.real,xrec1.imag)
plt.subplot(2,2,2)
plt.scatter(xrec.real,xrec.imag, c="g")

 

m = 4
xrec = xrec ** m
plt.figure(3, figsize=(10,10))
plt.scatter(xrec.real,xrec.imag, color = "r")

plt.figure(4, figsize=(10,10))

sig_fft = abs(fft(xrec, 2048))



plt.stem(sig_fft, "green")
# max_fft_sig = max(sig_fft)

# print("max_fft_sig =",max_fft_sig)

# #index_max_elem = sig_fft.index(max_fft_sig)
# index_max_elem = np.where(sig_fft == max_fft_sig)
# print("index_max_elem =", index_max_elem)
index_max_elem = np.argmax(sig_fft)
print("index_max_elem =", index_max_elem)
max_fft_sig = sig_fft[index_max_elem]
print("max_fft_sig =",max_fft_sig)


sig_fft_shift = fftshift(sig_fft)

plt.figure(5, figsize=(10,10))

plt.stem(sig_fft_shift, "black")

#plt.figure(6, figsize=(10,10))
#sig_fft_shift = sig_fft_shift/180


#plt.figure(7, figsize=(10,10))
#plt.stem(sig_fft_shift, "blue")

w = np.linspace(-np.pi, np.pi, len(sig_fft_shift))
index_max_elem = np.argmax(abs(sig_fft_shift))
print("index_max_elem =", index_max_elem)
max_fft_sig = sig_fft[index_max_elem]
print("max_fft_sig =",max_fft_sig)

plt.figure(6, figsize=(10,10))

plt.stem(w, sig_fft_shift, "red")
print("w[index_max_fft_shift] =", w[index_max_elem])

fax = w[index_max_elem] / m
print("fax = ", fax)
fax = abs(fax)

t2 = np.exp( -1j * fax)

xrec2 = xrec1 * t2
plt.figure(7, figsize=(10,10))
plt.scatter(xrec2.real,xrec2.imag, color = "g")





 

