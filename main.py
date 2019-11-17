import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import utils
import numpy as np
import pickle
from matplotlib import pyplot as plt
from HackatonDataset import HackatonDataset

from networks.Conv1 import Conv1Net
from augmentations import Transforms

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).type(torch.long)

        optimizer.zero_grad()
        output = model(data)

        '''
        print("Target: {}".format(target))
        print("Output: {}".format(output))
        print("-------------------------")
        '''

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).type(torch.long)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def prepareDataset( dataset_path,
                    train_part=0.7):
    sourceDataset = dataset = HackatonDataset.Load(dataset_path)

    trainLen = int(train_part * len(sourceDataset))

    trainData = {
        "data" : sourceDataset.data[0:trainLen],
        "labels" : sourceDataset.targets[0:trainLen]
    }

    testData = {
        "data" : sourceDataset.data[trainLen:],
        "labels" : sourceDataset.targets[trainLen:]
    }

    trainDataset = HackatonDataset(trainData)
    testDataset =  HackatonDataset(testData)

    #рассчитать статистику по тренировочному! датасету
    reshapedTrain = np.transpose(trainDataset.data, (1, 0, 2)).reshape(8, -1)
    train_mean = np.mean(reshapedTrain, axis=-1)
    train_std = np.std(reshapedTrain, axis=-1)

    #   возвращаем два датасета и статистику по тренировочному
    return trainDataset, testDataset, train_mean, train_std

DATASET_PATH = "datasets/BigDataset5Class.pickle"
VALIDATION_DATASET_PATH = "datasets/BigDataset_Validation5Class.pickle"

def main():
    trainDataset, testDataset, train_mean, train_std = prepareDataset(DATASET_PATH, 0.7)
    trainDataset.transform = Transforms.Compose([
        Transforms.Normalize(train_mean, train_std)
        #Transforms.AddNoise(0.035),
        #Transforms.ShiftAndCrop(100,100)
    ])
    testDataset.transform = Transforms.Normalize(train_mean, train_std)
    validationDataset = HackatonDataset.Load(VALIDATION_DATASET_PATH)

    val_mean = np.mean(validationDataset.data, axis = -1)
    val_std = np.std(validationDataset.data, axis = -1)
    validationDataset.data -= np.expand_dims(val_mean, axis = -1)
    validationDataset.data /= np.expand_dims(val_std, axis = -1)

    device = torch.device('cuda')
    kwargs = {'num_workers': 1, 'pin_memory': True}

    trainLoader = utils.data.DataLoader(trainDataset, shuffle=True, batch_size=64, **kwargs)
    testLoader = utils.data.DataLoader(testDataset, shuffle=False, batch_size=64, **kwargs)
    validationLoader = utils.data.DataLoader(validationDataset, shuffle=False, batch_size=16, **kwargs)

    model = Conv1Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=4e-4)
    #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)

    for epoch in range(1, 16):
        train(model, device, trainLoader, optimizer, epoch)
        test(model, device, testLoader)
    print("TEST ON VALIDATION")
    test(model, device, validationLoader)

if __name__ == '__main__':
    main()
