import argparse
import os
import pickle
import numpy as np
from RawDataOperation import RawDataReader
from HackatonDataset import HackatonDataset

DATASETS_DIR = "datasets/"

def readAll(dir):
    for root, dr, files in os.walk(dir):
        g_data = np.empty(shape=(len(files), 8, 2000))
        g_label = np.empty(shape=(len(files)))

        i = 0
        for file in files:
            path = dir + '\\' + file
            data = RawDataReader(fname=path).read()

            g_data[i] = np.array(data).reshape((len(data) // 8, 8)).transpose()
            g_label[i] = 0 if "turn_left" in os.path.splitext(file)[0] else 1
            i += 1

        data = {
            "data": g_data,
            "labels": g_label
        }
        return data

parser = argparse.ArgumentParser(description='Конвертация сырых данных в HackatonDataset')
parser.add_argument('file', type=str, metavar='path',
                    help='Директория с сырыми данными')
args = parser.parse_args()

data = readAll(args.file)
#   Перемешиваем датасет
permutationIndexes = np.random.permutation(len(data["data"]))
permutedData = {
    "data" : data["data"][permutationIndexes],
    "labels" : data["labels"][permutationIndexes]
}

dataset = HackatonDataset(permutedData)
dataset.save(DATASETS_DIR+"HackatonDataset.pickle")

#with open(DATASETS_DIR+"HackatonDataset.pickle", "wb") as f:
#    pickle.dump(dataset, f)