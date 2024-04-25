import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
f = 10 # Аналоговая частота сигнала
fs = 100 # Частота дискретизации

t = np.linspace(0, 1, int(fs))
sgnl_64 = np.cos(2 * np.pi * f * t[:64])
sgnl_128 = np.cos(2 * np.pi * f * t[:128])
sgnl_256 = np.cos(2 * np.pi * f * t[:256])

# Визуализация выборки отсчетов
plt.figure(1, figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.stem(sgnl_64)
plt.title('64 отсчета')
plt.subplot(3, 1, 2)
plt.stem(sgnl_128)
plt.title('128 отсчетов')
plt.subplot(3, 1, 3)
plt.stem(sgnl_256)
plt.title('256 отсчетов')

# Задание 2: Расчет значения аналоговой частоты сигнала
w = 0.5 * np.pi # Нормированная частота рад
f_analog = w * fs / (2 * np.pi)
print(f'Аналоговая частота сигнала при fs={fs} равна {f_analog} Гц')

# Задание 3: Вычисление ДПФ сигнала
fft_64 = np.fft.fft(sgnl_64)
fft_128 = np.fft.fft(sgnl_128)
fft_256 = np.fft.fft(sgnl_256)

freq64 = np.fft.fftfreq(len(sgnl_64), 1 / fs)
freq128 = np.fft.fftfreq(len(sgnl_128), 1 / fs)
freq256 = np.fft.fftfreq(len(sgnl_256), 1 / fs)

# Визуализация модуля спектра ДПФ
plt.figure(2, figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.stem(freq64[:len(sgnl_64)], np.abs(fft_64))
plt.xlabel('Частота, Гц')
plt.title('64 отсчета')
plt.subplot(3, 1, 2)

plt.stem(freq128[:len(sgnl_128)], np.abs(fft_128))
plt.xlabel('Частота, Гц')
plt.title('128 отсчетов')

plt.subplot(3, 1, 3)
plt.stem(freq256[:len(sgnl_256)], np.abs(fft_256))
plt.xlabel('Частота, Гц')
plt.title('256 отсчетов')

# Генерация сигнала из двух гармонических колебаний
f1 = 10
f2 = 25
sgnl = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)
# Вычисление ДПФ полученных отсчетов
fft_sgnl = np.fft.ifft(sgnl)

fft_sgnl = np.fft.fftshift(fft_sgnl)
# Визуализация спектра ДПФ
plt.figure(3, figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.stem(freq64[:len(sgnl_64)], np.abs(fft_64))
plt.xlabel('Частота, Гц')
plt.ylabel('Модуль спектра')
plt.title('Спектр ДПФ сигнала')
#plt.show()

# Расчет отсчетов цифрового фильтра ФНЧ
cutoff_freq = 10 # Частота среза
impulse_response = np.sinc(2 * cutoff_freq * (np.arange(len(sgnl)) - len(sgnl) / 2))

# Визуализация импульсной характеристики ФНЧ
plt.subplot(2, 2, 2)
#plt.stem(np.arange(len(sgnl)), impulse_response)
plt.stem(fft_sgnl)
plt.xlabel('Отсчеты')
plt.ylabel('Значение импульсной характеристики')
plt.title('Импульсная характеристика ФНЧ')
#plt.show()

# Применение импульсной характеристики фильтра к входному сигналу
filtered_sgnl = np.convolve(sgnl, impulse_response, mode='same')

# фильтрация

plt.figure(4, figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(t, sgnl, label='Входной сигнал')
plt.subplot(2, 2, 2)
plt.plot(t, filtered_sgnl, label='Отфильтрованный сигнал')
plt.xlabel('Время, сек')
plt.title('Сигнал после фильтрации')
plt.legend()


# Вычисление ДПФ сигнала после фильтрации
fft_filtered_sgnl = np.fft.fft(filtered_sgnl)

# Визуализация спектра ДПФ сигнала после фильтрации
plt.subplot(2, 2, 3)
plt.stem(freq128[:len(sgnl)], np.abs(fft_filtered_sgnl))
plt.xlabel('Частота, Гц')
plt.ylabel('Модуль спектра')
plt.title('Спектр ДПФ сигнала после фильтрации')





