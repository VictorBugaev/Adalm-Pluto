import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import adi 
import time

from scipy.fftpack import fft, ifft,  fftshift, ifftshift
import random


N = 50
Fs = 5000000
Ns = 50
F = 1213e6
n_sinhr_bits = 20
sinhr_bits = ""

fm = int(F)
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(Fs)
sdr.rx_buffer_size = 5000
sdr.rx_lo = fm
sdr.tx_lo = fm

sdr.tx_cyclic_buffer = False


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
    return qpsk_array
def char_to_bit(char):
    return ''.join(format(ord(i), '08b') for i in char)

def bit_to_char(bits, encoding='utf-8', errors='surrogatepass'):                    # перевод из битов в текст
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

def listen_sinhro_msg(data):
    c_s = 0
    for i in range(len(data)):
        pass

def DecodeData(data):
    pass
    

fig = plt.figure(1, figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1)

#непрерывное чтение
def ListenData(e):
    rx_data = sdr.rx()

    ax1.clear()
    
    plt.subplot(2, 2, 1)
    plt.ylim(-3000, 3000)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.scatter(rx_data.real, rx_data.imag)
    
    
    plt.subplot(2, 2, 2)
    plt.title("Временное представление")
    plt.plot(rx_data)

    data_f = fft(rx_data, Ns)
    plt.subplot(2, 2, 3)
    plt.title("Частотное представление")
    plt.plot(data_f)

CON = 3#выбор состояния работы
#Передатчик
if(CON == 1):

    bits = generate_random_bit(N)
    print("Битовая последовательность:", bits)
    bits = add_sinhr_bits(bits)
    print(bits)
    #OPSK
    n_qpsk = 4
    x_int = np.array(bits_to_qpsk(bits))
    x_degrees = x_int*360/float(n_qpsk) + 135 # 45, 135, 225, 315 град.
    x_radians = x_degrees*np.pi/180.0 # sin() и cos() в рад.
    x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) #генерируем комплексные числа
    samples = np.repeat(x_symbols, Ns) # 16 сэмплов на символ
    samples *= 2**14 #Повысим значения для наших сэмплов
    
    
    plt.figure(1, figsize=(10, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("Временное представление")
    plt.plot(samples)
    
    samples_f = fft(samples, Fs)
    
    plt.subplot(2, 2, 2)
    plt.title("Частотное представление")
    plt.plot(samples_f)
    
    
    plt.subplot(2, 2, 3)
    
    plt.scatter(samples.real, samples.imag)
    it = 0
    while(1):
        
        sdr.tx(samples)
        print(it)
        #it += 1

#Приемник(1 раз принять)
if(CON == 2):
    ListenData(1)
#Непрерывное прослуушивание
if(CON == 3):
    ani = animation.FuncAnimation(fig, ListenData, interval=100)
    plt.show()
