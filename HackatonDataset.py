import numpy as np
import torch
from torch import utils
import pickle

class HackatonDataset(utils.data.Dataset):
    """
    preparedData имеют структуру
    preparedData = {
        "data" : np.array[N, 8, 2000(возможно не 2000)],
        "labels" : np.array[N]
    }
    """
    def __init__(self, preparedData, transform=None):
        self.transform = transform
        self.data = preparedData["data"]
        self.targets = preparedData["labels"]

        assert self.data.shape[0] == self.targets.shape[0], "Первые размерности data и labels не совпадают!"

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def Load(path, transform=None):
        with open(path, "rb") as f:
            dataset = pickle.load(f)
            dataset.transform = transform
            return dataset

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = (
            torch.tensor( self.data[index], dtype=torch.float ),
            torch.tensor( self.targets[index], dtype=torch.long)
        )

        if self.transform:
            sample = self.transform(sample)
        return sample
