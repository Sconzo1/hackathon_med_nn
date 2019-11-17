from RawDataOperation import RawDataReader, RawDataWriter, get_curloc_time_UTC
import serial
import struct
import time
from Labels import Labels

#   Текущая метка данных - куда поворачиваем кисть
CURRENT_LABEL = Labels.TurnDown

#   директория, куда будут складываться сырые данные
RAW_DATA_DIR = "./rawdata/"
#   шаблон имени файла
FILENAME_TEMPLATE = "data_{label}_{utc}.dat"

def ComposeFilename():
    return FILENAME_TEMPLATE.format(label=CURRENT_LABEL,
    utc=get_curloc_time_UTC())

#   кол-во сэмплов на один замер
SAMPLES_PER_AQUISITION = 2000

print("Кол-во сэмплов на один замер: {}".format( SAMPLES_PER_AQUISITION ))

ser = serial.Serial(
port='COM4',
baudrate=921600,
parity=serial.PARITY_NONE,
stopbits=serial.STOPBITS_ONE,
bytesize=serial.EIGHTBITS,
timeout=None
)
# чтоб все данные влезали за одно считывание и не терялись
ser.set_buffer_size(rx_size=262144, tx_size=262144)

def ReadRawMeasurements(ser, samples = 2000):
    #чистим входной буфер, чтобы читать актуальные данные!
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    #читаем из COM порта сразу много данных
    print("START!" + str(time.localtime()))
    rawData = ser.read((samples + 32)*(16+2))
    print("END!" + str(time.localtime()))

    aaIdx = 0
    while aaIdx != -1:
        aaIdx = rawData.find(b'\xAA', aaIdx)
        bbIdx = aaIdx + 17
        if rawData[bbIdx] == 0xBB:
            print("Нашли корректный кадр данных! aaIdx = {}".format(aaIdx))
            break
        aaIdx += 1

    if aaIdx == -1:
        print("Не нашли 0xAA")
        raise Exception()

    rawData = rawData[aaIdx:]
    outData = []

    for i in range(samples):
        # проверить, что это корректный кадр данных
        aaIdx = i*18
        bbIdx = aaIdx + 17
        if rawData[aaIdx] == 0xAA and rawData[bbIdx] == 0xBB:
            #если это корректный кадр данных, то добавим его к уже найденным
            byteArray = rawData[aaIdx+1:bbIdx]
            #print(byteArray)
            unpacked = struct.unpack("<HHHHHHHH", byteArray)
            outData += unpacked
        else:
            print("Некорректный кадр данных обнаружен! Пропускаем его...")
            print("Предыдущий кадр " + str(rawData[aaIdx-18:bbIdx+1-18]))
            print("Текущий кадр " + str(rawData[aaIdx:bbIdx+1]))
            print("-----------------------------------------------------")
            break
    return outData

while(True):
    # Ждем сигнала пользователя для начала записи данных
    input("Нажмите Enter, чтобы начать запись нового фрагмента данных...")

    fname = RAW_DATA_DIR + ComposeFilename()
    fileWriter = RawDataWriter(fname)

    #Читаем данные
    data = ReadRawMeasurements(ser, SAMPLES_PER_AQUISITION)
    #print("Длина данных {}".format(len(data)))

    # Записываем массив в файл
    fileWriter.writeToFile(data)
    # Выводим сообщение об успешной записи
    print("{} сэмплов было записано в файл {}".format(
        len(data)//8,
        fname
    ))
