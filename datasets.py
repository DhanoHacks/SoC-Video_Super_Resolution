# Dataset Objects as used by Pytorch

import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.npad = ((0,0),(6,6),(6,6))

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = np.pad(np.expand_dims(f['lr'][idx] / 255., 0), pad_width=self.npad, mode='constant', constant_values=0)
            hr = np.expand_dims(f['hr'][idx] / 255., 0)
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self.npad = ((0,0),(6,6),(6,6))

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = np.pad(np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), pad_width=self.npad, mode='constant', constant_values=0)
            hr = np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
