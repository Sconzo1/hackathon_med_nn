import numpy as np
import scipy.signal

'''
Подавление частоты 50 Hz.
data - входной сигнал
fs - частота сэмплирования(Гц)
lowcut - нижняя частота среза(Гц)
highcut - верхняя частота среза(Гц)
order - порядок фильтра Баттерворта
'''
def reject50Hz(data, fs, lowcut = 49, highcut = 51, order = 3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = scipy.signal.butter(order, [low, high], btype='bandstop')
    y = scipy.signal.lfilter(i, u, data)
    return y
