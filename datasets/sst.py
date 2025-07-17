import os
import numpy as np
from torch.utils.data import Dataset

class SST(Dataset):
    def __init__(self, directory, transform=None):
        self.data = np.load(directory)
        self.data = self.data[:, 1:551, :]
        # self.data2 = self.data[24:, 1:551, :]
        # self.data = np.concatenate((self.data1, self.data2), axis=0)


        self.target = np.load('/public/home/huanggang/Baidu/DDPG/test_data_normalized.npy')
        self.target = self.target[-106:-10,:,:]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_data = self.data[index]
        sample_data = sample_data.reshape((1, 550, 511))

        target = self.target[index]  # Placeholder for any target data you might have
        target = target.reshape((1, 550, 511))
        return sample_data, target