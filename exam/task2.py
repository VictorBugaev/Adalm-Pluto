
import numpy as np
import matplotlib.pyplot as plt

# Задание 1: Генерация сигнала и выборка отсчетов
f = 10 # Аналоговая частота сигнала
fs = 100 # Частота дискретизации

t = np.linspace(0, 1, int(fs))
signal_64 = np.cos(2 * np.pi * f * t[:64])
signal_128 = np.cos(2 * np.pi * f * t[:128])
signal_256 = np.cos(2 * np.pi * f * t[:256])

# Визуализация выборки отсчетов
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.stem(signal_64)
plt.title('64 отсчета')
plt.subplot(3, 1, 2)
plt.stem(signal_128)
plt.title('128 отсчетов')
plt.subplot(3, 1, 3)
plt.stem(signal_256)
plt.title('256 отсчетов')

#plt.tight_layout()
plt.show()

# Задание 2: Расчет значения аналоговой частоты сигнала
w = 0.5 * np.pi # Нормированная частота рад
f_analog = w * fs / (2 * np.pi)
print(f'Аналоговая частота сигнала при fs={fs} равна {f_analog} Гц')

# Задание 3: Вычисление ДПФ сигнала
fft_64 = np.fft.fft(signal_64)
fft_128 = np.fft.fft(signal_128)
fft_256 = np.fft.fft(signal_256)

freq64 = np.fft.fftfreq(len(signal_64), 1 / fs)
freq128 = np.fft.fftfreq(len(signal_128), 1 / fs)
freq256 = np.fft.fftfreq(len(signal_256), 1 / fs)

# Визуализация модуля спектра ДПФ
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.stem(freq64[:len(signal_64)], np.abs(fft_64))
plt.xlabel('Частота, Гц')
plt.title('64 отсчета')
plt.subplot(3, 1, 2)

plt.stem(freq128[:len(signal_128)], np.abs(fft_128))
plt.xlabel('Частота, Гц')
plt.title('128 отсчетов')

plt.subplot(3, 1, 3)
plt.stem(freq256[:len(signal_256)], np.abs(fft_256))
plt.xlabel('Частота, Гц')
plt.title('256 отсчетов')


#plt.tight_layout()
plt.show()

# Генерация сигнала из двух гармонических колебаний
f1 = 5
f2 = 15

signal = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)

# Вычисление ДПФ полученных отсчетов
fft_signal = np.fft.fft(signal)

# Визуализация спектра ДПФ
plt.stem(freq64[:len(signal_64)], np.abs(fft_64))
plt.xlabel('Частота, Гц')
plt.ylabel('Модуль спектра')
plt.title('Спектр ДПФ сигнала')
plt.show()

# Расчет отсчетов цифрового фильтра ФНЧ
cutoff_freq = 10 # Частота среза
impulse_response = np.sinc(2 * cutoff_freq * (np.arange(len(signal)) - len(signal) / 2))

# Визуализация импульсной характеристики ФНЧ
plt.stem(np.arange(len(signal)), impulse_response)
plt.xlabel('Отсчеты')
plt.ylabel('Значение импульсной характеристики')
plt.title('Импульсная характеристика ФНЧ')
plt.show()

# Применение импульсной характеристики фильтра к входному сигналу
filtered_signal = np.convolve(signal, impulse_response, mode='same')

# Визуализация сигнала после фильтрации
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label='Входной сигнал')
plt.plot(t, filtered_signal, label='Отфильтрованный сигнал')
plt.xlabel('Время, сек')
plt.title('Сигнал после фильтрации')
plt.legend()
plt.show()

# Вычисление ДПФ сигнала после фильтрации
fft_filtered_signal = np.fft.fft(filtered_signal)

# Визуализация спектра ДПФ сигнала после фильтрации
plt.stem(freq128[:len(signal)], np.abs(fft_filtered_signal))
plt.xlabel('Частота, Гц')
plt.ylabel('Модуль спектра')
plt.title('Спектр ДПФ сигнала после фильтрации')
plt.show()