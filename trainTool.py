import torch
import numpy as np
import torch.utils.data as data
from sklearn import preprocessing
from os.path import join as pjoin
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def load_train_data(valid_size=0.2,batch_size=256, device='cpu'):
    data_list = [];
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

def train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch):
    train_losses = []
    valid_losses = []  # to track the validation loss as the model trains
    avg_train_losses = []  # to track the average training loss per epoch as the model trains
    avg_valid_losses = []  # to track the average validation loss per epoch as the model trains

    for t in range(max_training_epoch):
        train_losses = []
        valid_losses = []
        for label, target in train_loader:
            target_hat = model(label)
            loss = loss_fn(target_hat, target)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            train_losses.append(loss.item())
        for label, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            target_hat = model(label)
            loss = loss_fn(target_hat, target)
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print('Epoch', t, ': Train Loss is ', train_loss, 'Validate Loss is', valid_loss)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping at Epoch")
            break
    model.load_state_dict(torch.load('checkpoint.pt'))
    remove('checkpoint.pt')

    # plot
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, max(max(avg_valid_losses), max(avg_valid_losses)))  # consistent scale
    plt.xlim(0, len(avg_train_losses) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # fig.savefig(pjoin('model','LogNet',model_file_name+'.png'), bbox_inches='tight')

    return model

def test(model, test_input_file, test_output_file, output_scaler, device='cpu'):
    # test model
    test_input_mat = sio.loadmat(pjoin('data',test_input_file))['input_mat']
    test_output_mat = sio.loadmat(pjoin('data',test_output_file))['output_mat']
    test_input_mat = test_input_mat.T
    test_input_mat = test_input_mat[:, :-1]
    test_input_mat = np.concatenate((np.sin(test_input_mat), np.cos(test_input_mat)), axis=1)
    test_input_mat = torch.from_numpy(test_input_mat).to(device)
    test_input_mat = test_input_mat.float()
    y_pred = model(test_input_mat)
    test_output_mat_hat = output_scaler.inverse_transform(y_pred.data.numpy())
    test_output_mat = test_output_mat.T
    test_output_mat = test_output_mat[:, :-1]
    e_mat = np.sqrt(np.divide(np.sum(np.square(test_output_mat_hat - test_output_mat), axis=0),
                      np.sum(np.square(test_output_mat), axis=0)))
    print(e_mat)


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