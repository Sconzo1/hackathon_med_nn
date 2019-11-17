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

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).view(-1, 1)

        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
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
            data, target = data.to(device), target.to(device).view(-1, 1)
            output = model(data)
            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()
            print("Target: {}".format(target))
            pred = torch.sigmoid(output).ge(0.5)
            print("Pred: {}".format(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



DATASET1_PATH = "datasets/AndreyVladDatasetV1.pickle"
DATASET2_PATH = "datasets/VladDatasetV2.pickle"
TRAIN_PART = 0.7

def main():
    dataset = HackatonDataset.Load(DATASET1_PATH)
    dataset2 = HackatonDataset.Load(DATASET2_PATH)

    dataToAnalyze = np.transpose(dataset.data, (1, 0, 2)).reshape(8, -1)

    mean = np.mean(dataToAnalyze, axis=-1)
    std = np.std(dataToAnalyze, axis=-1)

    dataset.data -= np.expand_dims(mean, 1)
    dataset.data /= np.expand_dims(std, 1)
    dataset2.data -= np.expand_dims(mean, 1)
    dataset2.data /= np.expand_dims(std, 1)

    print( np.mean(dataset2.data, axis=-1) )
    print( np.std(dataset2.data, axis=-1) )

    trainDataset = utils.data.Subset(dataset, range(0, int(0.7 * len(dataset))) )
    testDataset = utils.data.Subset(dataset, range(int(0.7 * len(dataset)), len(dataset)))

    device = torch.device('cuda')
    kwargs = {'num_workers': 1, 'pin_memory': True}

    trainLoader = utils.data.DataLoader(trainDataset, shuffle=True, batch_size=16, **kwargs)
    testLoader = utils.data.DataLoader(testDataset, shuffle=False, batch_size=16, **kwargs)
    test2Loader = utils.data.DataLoader(dataset2, shuffle=False, batch_size=8, **kwargs)

    model = Conv1Net().to(device)
    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)

    for epoch in range(1, 3):
        train(model, device, trainLoader, optimizer, epoch)
        test(model, device, testLoader)
    print("TEST V2 --------------------")
    test(model, device, test2Loader)

if __name__ == '__main__':
    main()
