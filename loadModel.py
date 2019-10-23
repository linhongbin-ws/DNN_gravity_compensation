from Net import *
import numpy as np
from os import path, mkdir
import torch
def get_model(robot, use_net, D, device='cpu'):
    if robot=='MTM':
        if use_net == 'SinNet':
            model = SinNet(D, 100, D).to(device)
        elif use_net == 'ReLuNet':
            model = ReLuNet(D, [100], D).to(device)
        elif use_net == 'SigmoidNet':
            model = SigmoidNet(D, 100, D).to(device)
        elif use_net == 'Multi_SinNet':
            model = Multi_SinNet(D, 100, D).to(device)
        elif use_net == 'VanillaPolyNet':
            base_model = SinNet(D, 100, D).to(device)
            additon_modelPos = PolNet(D, 2).to(device)
            additon_modelNeg = PolNet(D, 2).to(device)
            modelPos = VanillaNet(base_model, additon_modelPos)
            modelNeg = VanillaNet(base_model, additon_modelNeg)
            modelList = [modelPos, modelNeg]
        elif use_net == 'VanillaBPNet':
            base_model = SinNet(D, 100, D).to(device)
            additon_modelPos = SigmoidNet(D, 20, D).to(device)
            additon_modelNeg = SigmoidNet(D, 20, D).to(device)
            modelPos = VanillaNet(base_model, additon_modelPos)
            modelNeg = VanillaNet(base_model, additon_modelNeg)
            model = [modelPos, modelNeg]
        elif use_net == 'Lagrangian_SinNet':
            model = SigmoidNet(D, 300, 1).to(device)
            delta_q = 1e-2
            w_list = [1, 1]
            w_vec = torch.from_numpy(np.array(w_list).T).to(device).float()
        else:
            raise Exception(use_net + 'is not support')
    else:
        raise Exception(robot + 'is not support')

    return model

def save_model(file_path, file_name, model):
    if not path.exists(file_path):
        mkdir(path)

    if isinstance(model, list):
        model_dict = {'model' + str(i + 1): model[i].state_dict() for i in range(len(model))}
        torch.save(model_dict, file_name+'.pt')
    else:
        torch.save(model.state_dict(), file_name+'.pt')

def save_model(file_path, file_name, model):
    if not path.exists(file_path):
        mkdir(path)

    if isinstance(model, list):
        model_dict = {'model' + str(i + 1): model[i].state_dict() for i in range(len(model))}
        torch.save(model_dict, path.join(file_path, file_name+'.pt'))
    else:
        torch.save(model.state_dict(), path.join(file_path, file_name+'.pt'))

def load_model(file_path, file_name, model):
    file = path.join(file_path, file_name)
    file = file+'.pt'
    if not path.isfile(file):
        raise Exception(file+ 'cannot not be found')

    if isinstance(model, list):
        checkpoint = torch.load(file)
        for i in range(len(model)):
            model[i].load_state_dict(checkpoint['model' + str(i + 1)])
    else:
        model.load_state_dict(torch.load(file))
    return model