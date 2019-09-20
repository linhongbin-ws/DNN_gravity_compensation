from os.path import join as pjoin
import scipy.io as sio
import torch
from torchvision import transforms as tf
import numpy as np

input_mat = sio.loadmat(pjoin('data','Real_MTMR_pos_4096.mat'))['input_mat']
output_mat = sio.loadmat(pjoin('data','Real_MTMR_tor_4096.mat'))['output_mat']

device =  torch.device('cpu')


H = 1000

x = torch.from_numpy(input_mat.T).to(device)
y = torch.from_numpy(output_mat.T).to(device)
x = x.float()
y = y.float()

norm_ = tf.Normalize(0,1)
x = norm_(x)
y = norm_(y)

(N_in, D_in) = x.shape
(N_out, D_out) = y.shape
assert N_in == N_out

model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out)).to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param.data -= learning_rate * param.grad

