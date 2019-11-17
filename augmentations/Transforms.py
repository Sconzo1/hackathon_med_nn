import numpy as np
import torch
import torch.nn.functional as F
import random

class ShiftAndCrop(object):
    def __init__(self, max_left_shift = 100, max_right_shift = 100):
        self.max_left_shift = max_left_shift
        self.max_right_shift = max_right_shift
        pass

    def __call__(self, sample):
        (data, target) = sample

        padded = F.pad(data, (self.max_left_shift, self.max_right_shift), "constant", 0)
        shift = self.max_left_shift + random.randint(-self.max_left_shift, self.max_right_shift)
        cropped = padded[:, shift:shift + data.shape[1] ]
        '''
        print("Shift is {}. Cropped shape is {}".format(
            shift,
            cropped.shape
        ))
        '''
        return (cropped, target)

class AddNoise(object):
    def __init__(self, noise_power=0.1):
        super(AddNoise, self).__init__()
        self.noise_power = noise_power

    def __call__(self, sample):
        (data, target) = sample
        noised_data = self.noise_power * torch.randn_like(data) + data
        return (noised_data, target)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std, dtype=torch.float)

    def __call__(self, sample):
        (data, target) = sample
        transformed = (data - self.mean[:, None]) / self.std[:, None]
        return (transformed, target)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class RandomEraser(object):
    def __init__(self, max_erase_length=40, max_erase_count=10):
        super(RandomEraser, self).__init__()
        self.max_erase_length = max_erase_length
        self.max_erase_count = max_erase_count

    def __call__(self, sample):
        (data, target) = sample

        erase_length = randint(0, self.max_erase_length)
        count = randint(0, self.max_erase_count)

        for i in range(0, count):
            index = randint(0, data.shape[-1] - erase_length - 1)
            data[:, index:(index + erase_length)] = 0

        return (data, target)
