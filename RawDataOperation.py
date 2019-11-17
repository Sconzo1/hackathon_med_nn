import struct
import time

def get_curloc_time_UTC():
    return time.strftime("%d_%H%M%S", time.localtime())

class RawDataWriter:
    def __init__(self, fname):
        self.fname = fname

    #   data - массив uint16
    def writeToFile(self, data):
        with open(self.fname, "wb") as f:
            dataToWrite = struct.pack("<" + "H" * len(data), *data)
            f.write(dataToWrite)

class RawDataReader:
    def __init__(self, fname):
        self.fname = fname

    def read(self):
        with open(self.fname, "rb") as f:
            rawBytes = f.read()
            rawUShorts = struct.unpack("<" + "HHHHHHHH" * (len(rawBytes)//16), rawBytes)
            return rawUShorts
