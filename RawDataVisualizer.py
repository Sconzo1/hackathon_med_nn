import numpy as np
import argparse
from matplotlib import pyplot as plt
from RawDataOperation import RawDataReader

parser = argparse.ArgumentParser()
parser.add_argument("rawfile", help="Путь до файла с сырыми данными")
args = parser.parse_args()

reader = RawDataReader(fname=args.rawfile)
rawUshorts = reader.read()

dataLen = len(rawUshorts)//8

#сформировать сырые данные в numpy array
data = np.array(rawUshorts).reshape((dataLen, 8)).transpose()
print(data.shape)
print(data)

for i in range(len(data)):
    plt.plot(data[i])
plt.show()
