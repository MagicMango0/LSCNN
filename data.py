import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from sklearn import preprocessing


class LSCNNDataset(Dataset):
    def __init__(self, dataset_type, seed, path='./dataset/preliminary_data.csv'):
        super(LSCNNDataset, self).__init__()
        fiture, y = read_data(path)
        fiture_train, fiture_test, y_train, y_test = train_test_split(fiture, y, test_size=0.1667, random_state=seed)
        fiture_train, fiture_val, y_train, y_val = train_test_split(fiture_train, y_train, test_size=0.1,
                                                                    random_state=seed)
        fing_train, des_train = fiture_train[:, -1024:], fiture_train[:, :-1024]
        fing_val, des_val = fiture_val[:, -1024:], fiture_val[:, :-1024]
        fing_test, des_test = fiture_test[:, -1024:], fiture_test[:, :-1024]

        scaler = preprocessing.StandardScaler().fit(des_train)

        if dataset_type == 'train':
            des_train_scaled = scaler.transform(des_train)
            self.x_data = torch.from_numpy(cat(des_train_scaled, fing_train))
            self.y_data = y_train
            self.len = self.x_data.shape[0]

        elif dataset_type == 'val':
            val_des_scaled = scaler.transform(des_val)
            self.x_data = torch.from_numpy(cat(val_des_scaled, fing_val))
            self.y_data = y_val
            self.len = self.x_data.shape[0]

        else:
            test_des_scaled = scaler.transform(des_test)
            self.x_data = torch.from_numpy(cat(test_des_scaled, fing_test))
            self.y_data = y_test
            self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def read_data(path):
    data = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=1)
    fiture = data[:, 4:-1]
    target = data[:, -1:]

    return fiture, target


def cat(a, b):
    return np.concatenate((a, b), axis=1)


if __name__ == '__main__':
    data = LSCNNDataset('train', 0)
