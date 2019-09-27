from os.path import join as pjoin
import scipy.io as sio
import torch
from sklearn import preprocessing
import numpy as np
import _pickle as cPickle
from pytorchtools import EarlyStopping
import torch.utils.data as data
import matplotlib.pyplot as plt
from copy import deepcopy


# global configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
H = 1000 # number of hidden neurons
learning_rate = 0.01 # learning rate
max_training_epoch = 2000 # maximum training epoch
goal_loss = 1e-3 # goal loss
valid_size = 0.2
batch_size = 256
earlyStop_patience = 8
model_file_name = 'LogNet_CAD_sim_rand_1e5_Target'
train_input_file_list = ['CAD_sim_rand_1e5_pos_2.mat']
train_output_file_list = ['CAD_sim_rand_1e5_tor_2.mat']
source_model_file = 'LogNet_CAD_sim_rand_1e5_Source.pt'


# create Net architecture
class LogNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(LogNet, self).__init__()
        self._relu = torch.nn.ReLU()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        torch.log(x)
        h = self._input_linear(x)
        torch.exp(h)
        h = self._relu(h)
        y_pred = self._output_linear(h)
        return y_pred


input_mat = []
output_mat = []
# load .mat data to numpy
for train_input_file, train_output_file in zip(train_input_file_list, train_output_file_list):
    input = sio.loadmat(pjoin('data',train_input_file))['input_mat']
    output = sio.loadmat(pjoin('data',train_output_file))['output_mat']
    input_mat = input if len(input_mat)==0 else np.concatenate((input_mat, input), axis=1)
    output_mat = output if len(output_mat)==0 else np.concatenate((output_mat, output), axis=1)

# data pre-processing
input_mat = input_mat.T
output_mat = output_mat.T
input_mat = input_mat[:, :-1]
output_mat = output_mat[:, :-1]
    # scale output to zeroscore
output_scaler = preprocessing.StandardScaler().fit(output_mat)
output_mat = output_scaler.transform(output_mat)
# input feature = [sin(q) cos(q)]
input_mat = np.concatenate((np.sin(input_mat), np.cos(input_mat)), axis=1)
    # numpy to torch tensor
x = torch.from_numpy(input_mat).to(device)
y = torch.from_numpy(output_mat).to(device)
x = x.float()
y = y.float()
(N_in, D_in) = x.shape
(N_out, D_out) = y.shape
assert N_in == N_out

# split train set and vaidate set
indices = list(range(N_in))
np.random.shuffle(indices)
split = int(np.floor(valid_size * N_in))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = data.sampler.SubsetRandomSampler(train_idx)
valid_sampler = data.sampler.SubsetRandomSampler(valid_idx)
train_dataSet = data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(train_dataSet,
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           num_workers=0)
valid_loader = torch.utils.data.DataLoader(train_dataSet,
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           num_workers=0)



# configure network and optimizer
model = LogNet(D_in, H, D_out)
model.load_state_dict(torch.load(pjoin('model','LogNet',source_model_file)))
loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)
train_losses = []
valid_losses = []  # to track the validation loss as the model trains
avg_train_losses = []  # to track the average training loss per epoch as the model trains
avg_valid_losses = []  # to track the average validation loss per epoch as the model trains

# freeze the first layer
count = 0
for param in model.parameters():
    count += 1
    if count == 1:
        param.requires_grad = False


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






## plot

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')

# find position of lowest validation loss
minposs = avg_valid_losses.index(min(avg_valid_losses))+1
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, max(max(avg_valid_losses), max(avg_valid_losses))) # consistent scale
plt.xlim(0, len(avg_train_losses)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


fig.savefig(pjoin('model','LogNet',model_file_name+'.png'), bbox_inches='tight')
torch.save(model.state_dict(), pjoin('model','LogNet',model_file_name+'.pt'))
with open(pjoin('model','LogNet',model_file_name+'.pkl'), 'wb') as fid:
    cPickle.dump(output_scaler, fid)

