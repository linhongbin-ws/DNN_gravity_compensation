from os.path import join as pjoin
import scipy.io as sio
import torch
from sklearn import preprocessing
import numpy as np

input_mat = sio.loadmat(pjoin('data','Real_MTMR_pos_4096.mat'))['input_mat']
output_mat = sio.loadmat(pjoin('data','Real_MTMR_tor_4096.mat'))['output_mat']

device = torch.device('cpu')
H = 1000

input_mat = input_mat.T
output_mat = output_mat.T
input_scaler = preprocessing.StandardScaler().fit(input_mat)
output_scaler = preprocessing.StandardScaler().fit(output_mat)
input_mat = input_scaler.transform(input_mat)
output_mat = output_scaler.transform(output_mat)
input_mat = input_mat[:, :-1]
output_mat = output_mat[:, :-1]


x = torch.from_numpy(input_mat).to(device)
y = torch.from_numpy(output_mat).to(device)
x = x.float()
y = y.float()

(N_in, D_in) = x.shape
(N_out, D_out) = y.shape
assert N_in == N_out

model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(H, D_out)).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for t in range(3000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    loss_val = loss.item()
    print('Epoch', t, ': Loss is ', loss_val)

    if loss_val<1e-4:
        break

