import torch
import _pickle as cPickle
from regularizeTool import EarlyStopping
from trainTool import train
from Net import *
from loadDataTool import load_train_data
from os.path import join
from evaluateTool import *
import scipy.io as sio

# path
train_data_path = join("data", "acrobot_sim_64")
test_data_path = join("data", "acrobot_sim_1156")
save_result_path = join("figure","bp_train_8")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ReLuNet(2, [100, 100], 2).to(device)
model = LogNet(2,100,2).to(device)

# config hyper-parameters
H = 1000  # number of hidden neurons
learning_rate = 0.01 # learning rate
max_training_epoch = 2000 # stop train when reach maximum training epoch
goal_loss = 1e-4 # stop train when reach goal loss
valid_ratio = 0.2 # ratio of validation data set over train and validate data
batch_size = 256 # batch size for mini-batch gradient descent
earlyStop_patience = 15 # epoch number of looking ahead


# load data
print(device)
train_loader, valid_loader, input_scaler, output_scaler, input_dim, output_dim = load_train_data(data_dir=train_data_path,
                                                                                                 valid_ratio=valid_ratio,
                                                                                                 batch_size=batch_size,
                                                                                                 device=device)
# configure network and optimizer
# model = BPNet(input_dim, H, output_dim)

loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)


# train model
model = train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch)
# test model
model = model.to('cpu')
test_dataset = load_data_dir(test_data_path, device='cpu', is_scale=False)
test_input_mat = test_dataset.x_data
test_output_mat = predict(model, test_input_mat, input_scaler, output_scaler, 'cpu')
print(test_output_mat)

train_dataset = load_data_dir(train_data_path, device='cpu', is_scale=False)
train_input_mat = train_dataset.x_data
train_output_mat = train_dataset.y_data

sio.savemat(join(save_result_path,'result.mat'), {'test_input_mat': test_input_mat.numpy(),
                                                  'test_output_mat': test_output_mat.numpy(),
                                                  'train_input_mat': train_input_mat.numpy(),
                                                  'train_output_mat': train_output_mat.numpy()})
# save model
# torch.save(model.state_dict(), pjoin('model','LogNet',model_file_name+'.pt'))
# with open(pjoin('model','LogNet',model_file_name+'.pkl'), 'wb') as fid:
#     cPickle.dump(output_scaler, fid)

