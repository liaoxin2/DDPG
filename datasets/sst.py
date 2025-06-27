import os
import numpy as np
import paddle

class SST(paddle.io.Dataset):
    def __init__(self, directory, target_path, transform=None):
        self.data = np.load(directory)

        if target_path is not None:
            self.target = np.load(target_path)
        else:
            self.target = 0
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_data = self.data[index]
        sample_data = sample_data.reshape((1, 550, 511))

        if (self.target != 0).any():
            target = self.target[index]  # Placeholder for any target data you might have
            target = target.reshape((1, 550, 511))
        else:
            target = 0
        return sample_data, target