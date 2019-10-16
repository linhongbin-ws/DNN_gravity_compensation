import torch
import _pickle as cPickle
from regularizeTool import EarlyStopping
from trainTool import train, train_lagrangian
from Net import *
from loadDataTool import load_train_data
from os.path import join
from evaluateTool import *
import scipy.io as sio
from os import mkdir

# path


def loop_func(train_file, use_net):
    train_data_path = join("data", "Acrobot", "sim", train_file)
    test_data_path = join("data", "Acrobot", "sim", "N34_std1")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #use_net = 'SinNet'
    #use_net = 'ReLuNet'
    #use_net = 'SigmoidNet'
    # use_net = 'Lagrangian_SinNet'

    if use_net == 'SinNet':
        model = SinNet(2, 100, 2).to(device)
    elif use_net == 'ReLuNet':
        model = ReLuNet(2, [100], 2).to(device)
    elif use_net == 'SigmoidNet':
        model = SigmoidNet(2, 100, 2).to(device)
    elif use_net == 'Lagrangian_SinNet':
        model = SinNet(2, 200, 1).to(device)
        delta_q = 1e-5
    else:
        raise Exception(use_net + 'is not support')


    #model = LogNet(2,100,2).to(device)

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
    train_loader, valid_loader, input_scaler, output_scaler, input_dim, output_dim = load_train_data(data_dir=join(train_data_path, "data"),
                                                                                                     valid_ratio=valid_ratio,
                                                                                                     batch_size=batch_size,
                                                                                                     device=device)
    # configure network and optimizer
    # model = BPNet(input_dim, H, output_dim)

    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)


    # train model
    if use_net=='Lagrangian_SinNet':
        model = train_lagrangian(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch,
                                 delta_q=delta_q, is_plot=False)
    else:
        model = train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch, is_plot=False)
    # test model
    model = model.to('cpu')
    test_dataset = load_data_dir(join(test_data_path,"data"), device='cpu', is_scale=False)
    test_input_mat = test_dataset.x_data
    if use_net=='Lagrangian_SinNet':
        test_output_mat = predict_lagrangian(model, test_input_mat, input_scaler, output_scaler, delta_q)
    else:
        test_output_mat = predict(model, test_input_mat, input_scaler, output_scaler, 'cpu')


    print(test_output_mat)

    train_dataset = load_data_dir(join(train_data_path,'data'), device='cpu', is_scale=False)
    train_input_mat = train_dataset.x_data
    train_output_mat = train_dataset.y_data

    try:
        mkdir(join(train_data_path,"result"))
    except:
        print(join(train_data_path,"result") + " already exist")

    sio.savemat(join(train_data_path, "result", use_net+'.mat'), {'test_input_mat': test_input_mat.numpy(),
                                                      'test_output_mat': test_output_mat.numpy(),
                                                      'train_input_mat': train_input_mat.numpy(),
                                                      'train_output_mat': train_output_mat.numpy()})
    # save model
    # torch.save(model.state_dict(), pjoin('model','LogNet',model_file_name+'.pt'))
    # with open(pjoin('model','LogNet',model_file_name+'.pkl'), 'wb') as fid:
    #     cPickle.dump(output_scaler, fid)

train_file_list = ['N5_std1', 'N5_std5', 'N5_std9','N8_std1', 'N8_std5', 'N8_std9','N15_std1', 'N15_std5', 'N15_std9']
use_net_list = ['SinNet', 'ReLuNet', 'SigmoidNet', 'Lagrangian_SinNet']

for train_file in train_file_list:
    for use_net in use_net_list:
        loop_func(train_file, use_net)