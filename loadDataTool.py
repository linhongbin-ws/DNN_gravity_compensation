from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import os

class MTMDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, device='cpu'):
        input_mat = []
        output_mat = []
        # load .mat data to numpy
        for file_name in data_list:
            input = sio.loadmat(file_name)['input_mat']
            output = sio.loadmat(file_name)['output_mat']
            input_mat = input if len(input_mat)==0 else np.concatenate((input_mat, input), axis=0)
            output_mat = output if len(output_mat)==0 else np.concatenate((output_mat, output), axis=0)

        # _, s_size = input_mat.shape
        # indices = list(range(s_size))
        # np.random.shuffle(indices)
        # input_mat = input_mat[:, indices[:train_samples_num]]
        # output_mat = output_mat[:, indices[:train_samples_num]]

        # data pre-processing
            # scale output to zeroscore
        self.output_scaler = preprocessing.StandardScaler().fit(output_mat)
        output_mat = self.output_scaler.transform(output_mat)
        # input feature = [sin(q) cos(q)]
        input_mat = np.concatenate((np.sin(input_mat), np.cos(input_mat)), axis=1)
            # numpy to torch tensor
        self.x_data = torch.from_numpy(input_mat).to(device).float()
        self.y_data = torch.from_numpy(output_mat).to(device).float()
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def load_train_data(valid_size=0.2,batch_size=256, device='cpu'):
    data_list = []
    for root, dirs, files in os.walk(os.path.join("data", "data")):
        for file in files:
            if file.endswith(".mat"):
                print(os.path.join(root, file))
                data_list.append(os.path.join(root, file))

    full_dataset = MTMDataset(data_list, device=device)
    train_ratio = 1 -valid_size
    train_size = int(full_dataset.__len__() * train_ratio)
    test_size = full_dataset.__len__() - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=True
                              )
    valid_loader = DataLoader(validate_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=True)

    return train_loader, valid_loader, full_dataset.output_scaler