import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from reject50 import reject50Hz

TWO_PI = 2 * np.pi
SAMPLING_FREQ = 1000 #частота выборки в Гц
T = 1                #длительность выборки в сек

t = np.arange(0, T, 1.0/SAMPLING_FREQ)

sin = {
    1 : np.sin(TWO_PI*t),
    2 : np.sin(2*TWO_PI*t),
    10 : 0.5 * np.sin(10*TWO_PI*t),
    50 : 0.5 * np.sin(50*TWO_PI*t),
    55 : 0.25 * np.sin(55*TWO_PI*t)
    }

# тестовый сигнал, содержащий 50 Гц
signal = sin[1] + sin[2] + sin[10] + sin[50] + sin[55]

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = scipy.signal.butter(order, [low, high], btype='bandstop')
    y = scipy.signal.lfilter(i, u, data)
    return y

#filtered = butter_bandstop_filter(signal, 49, 51, SAMPLING_FREQ, order=5)
filtered = reject50Hz(signal, SAMPLING_FREQ)


def butter_bandstop_filter_response(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = scipy.signal.butter(order, [low, high], btype='bandstop')
    w, h = scipy.signal.freqz(b, a, fs=fs)
    plt.plot(w, abs(h))
    plt.show()

#butter_bandstop_filter_response(49, 51, SAMPLING_FREQ, 3)

plt.magnitude_spectrum(signal, Fs=SAMPLING_FREQ, label='original')
plt.magnitude_spectrum(filtered, Fs=SAMPLING_FREQ, label='filtered')
plt.show()
