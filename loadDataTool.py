from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import os

class MTMDataset(Dataset):
    def __init__(self, data_list, device='cpu', is_scale=True):
        input_mat = []
        output_mat = []
        # load .mat file to numpy
        for file_name in data_list:
            input = sio.loadmat(file_name)['input_mat']
            output = sio.loadmat(file_name)['output_mat']
            input_mat = input if len(input_mat)==0 else np.concatenate((input_mat, input), axis=0)
            output_mat = output if len(output_mat)==0 else np.concatenate((output_mat, output), axis=0)

        # scale output to zeroscore
        if is_scale:
            self.input_scaler = preprocessing.StandardScaler().fit(input_mat)
            self.output_scaler = preprocessing.StandardScaler().fit(output_mat)
            input_mat = self.input_scaler.transform(input_mat)
            output_mat = self.output_scaler.transform(output_mat)

        # numpy to torch tensor
        self.x_data = torch.from_numpy(input_mat).to(device).float()
        self.y_data = torch.from_numpy(output_mat).to(device).float()

        # get length of pair CAD_sim_1e6
        self.len = self.x_data.shape[0]

        # get dimension of input and output CAD_sim_1e6
        self.input_dim = input_mat.shape[1]
        self.output_dim = output_mat.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def load_data_dir(data_dir, device='cpu', is_scale=True):
    data_list = []
    if not os.path.exists(data_dir):
        raise Exception('cannot find directory: {}'.format(os.getcwd()+data_dir))
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mat"):
                print(os.path.join(root, file))
                data_list.append(os.path.join(root, file))
    full_dataset = MTMDataset(data_list, device=device, is_scale=is_scale)
    return full_dataset

def load_train_data(data_dir, valid_ratio=0.2, batch_size=256, device='cpu'):
    full_dataset = load_data_dir(data_dir, device)
    train_ratio = 1 - valid_ratio
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

    return train_loader, valid_loader, full_dataset.input_scaler, full_dataset.output_scaler, full_dataset.input_dim, full_dataset.output_dim

