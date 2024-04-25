import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import max_len_seq
import random
from scipy.fftpack import fft, ifft,  fftshift, ifftshift



sdr = adi.Pluto('ip:192.168.3.1')
sdr.sample_rate = 1000000
sdr.tx_destroy_buffer()
 

sdr.rx_lo = 2000000000
sdr.tx_lo = 2000000000
sdr.tx_cyclic_buffer = True
#sdr.tx_cyclic_buffer = False
sdr.tx_hardwaregain_chan0 = -5
sdr.gain_control_mode_chan0 = "fast_attack"#"slow_attack"


fs = sdr.sample_rate
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
filename_input = "input_qpsk_old_type.txt"
filename_output = "output_qosk_old_type.txt"

sinhr_bits = ""

n_sinhr_bits = 20
for i in range(n_sinhr_bits):
    sinhr_bits += "1"


def generate_random_bit(N):
    bits = ""
    for i in range(N):
        #bits += str(np.random.randint(0, 2, 1))
        bits += str(random.randint(0, 1))
    return bits

def add_sinhr_bits(bits):
    return sinhr_bits + bits + sinhr_bits
def bits_to_qpsk(bits):
    qpsk_array = []
    for i in range(0, len(bits), 2):
        print(bits[i:i+2])
        if(bits[i : i+2] == "00"):
            qpsk_array.append(0)
        elif(bits[i : i+2] == "01"):
            qpsk_array.append(1)
        elif(bits[i : i+2] == "10"):
            qpsk_array.append(2)
        elif(bits[i : i+2] == "11"):
            qpsk_array.append(3)
            
        #qpsk_array.append()
    return qpsk_array
def char_to_bit(char):
    return ''.join(format(ord(i), '08b') for i in char)

N = 50
n_qpsk = 4

bits = list(data)
bits = list(map(int, bits))
bits = list(map(str, bits))

bits = "".join(bits)

bits = generate_random_bit(N)


print("Битовая последовательность:", bits)

bits = add_sinhr_bits(bits)
#x_int = np.random.randint(0, n_qpsk, N)
x_int = np.array(bits_to_qpsk(bits))
x_degrees = x_int*360/float(n_qpsk) + 135 # 45, 135, 225, 315 град.
x_radians = x_degrees*np.pi/180.0 # sin() и cos() в рад.
x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) #генерируем комплексные числа
samples = np.repeat(x_symbols, N) # 16 сэмплов на символ
samples *= 2**14 #Повысим значения для наших сэмплов




xiq = samples
n_frame= len(xiq)
plt.figure(1, figsize=(10, 10))
plt.scatter(xiq.real,xiq.imag)
 
sdr.tx(xiq)
np.savetxt(filename_input, xiq, delimiter=":")
sdr.rx_rf_bandwidth = 1000000
sdr.rx_destroy_buffer()
sdr.rx_hardwaregain_chan0 = -5
sdr.rx_buffer_size =2*n_frame
xrec1=sdr.rx()
sdr.tx_destroy_buffer()
np.savetxt(filename_output, xrec1, delimiter=":")
xrec = xrec1/np.mean(xrec1**2)
plt.figure(2, figsize=(10,10))
plt.scatter(xrec.real,xrec.imag)


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




