import torch
from loadDataTool import load_data_dir
import numpy as np


def predict(model, input_mat, input_scaler, output_scaler):
    if input_scaler is None:
        feature_norm = input_mat
    else:
        feature_norm = torch.from_numpy(input_scaler.transform(input_mat.numpy())).to('cpu').float()

    target_norm_hat = model(feature_norm)
    if  isinstance(target_norm_hat, tuple):
        target_norm_hat = target_norm_hat[0]

    if output_scaler is None:
        target_hat_mat = target_norm_hat
    else:
        target_hat_mat = output_scaler.inverse_transform(target_norm_hat.detach().numpy())
    target_hat = torch.from_numpy(target_hat_mat)
    return target_hat

# predict from numpy input to numpy output
def predictNP(model, input_mat, input_scaler, output_scaler):
    if input_scaler is not None:
        feature_norm = torch.from_numpy(input_scaler.transform(input_mat)).to('cpu').float()
    else:
        feature_norm = torch.from_numpy(input_mat).to('cpu').float()

    target_norm_hat = model(feature_norm)
    if isinstance(target_norm_hat, tuple):
        target_norm_hat = target_norm_hat[0]

    if output_scaler is not None:
        target_hat_mat = output_scaler.inverse_transform(target_norm_hat.detach().numpy())
    else:
        target_hat_mat = target_norm_hat.detach().numpy()

    return target_hat_mat

def predictList(modelList, input_mat, input_scalerList, output_scalerList):
    target_hat = []
    for i in range(len(modelList)):
        feature_norm = torch.from_numpy(input_scalerList[i].transform(input_mat.numpy())).to('cpu').float()
        target_norm_hat = modelList[i](feature_norm)
        target_hat_mat = output_scalerList[i].inverse_transform(target_norm_hat.detach().numpy())
        if isinstance(target_hat, list):
            target_hat = torch.from_numpy(target_hat_mat)
        else:
            target_hat = target_hat + torch.from_numpy(target_hat_mat)
    target_hat = target_hat/len(modelList)
    return target_hat



def evaluate_rms_list(modelList, test_data_path, input_scalerList, output_scalerList, device, verbose=True):
    # test model
    test_dataset = load_data_dir(test_data_path, device=device, is_scale=False)
    feature = test_dataset.x_data
    target = test_dataset.y_data

    target_hat = predictList(modelList, feature, input_scalerList, output_scalerList)
    target_hat_mat = target_hat.numpy()

    target_mat = target.numpy()
    rel_rms_vec = np.sqrt(np.divide(np.mean(np.square(target_hat_mat - target_mat), axis=0),
                                    np.mean(np.square(target_mat), axis=0)))

    abs_rms_vec = np.sqrt(np.mean(np.square(target_hat_mat - target_mat), axis=0))

    if verbose:
        print('Absolute RMS for each joint are:', abs_rms_vec)
        print('Relative RMS for each joint are:', rel_rms_vec)

    return abs_rms_vec, rel_rms_vec
def evaluate_rms(model, loss_fn, test_data_path, input_scaler, output_scaler, device, verbose=True):
    # test model
    test_dataset = load_data_dir(test_data_path, device=device, input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False)
    feature = test_dataset.x_data
    target = test_dataset.y_data

    # scale input data
    if input_scaler is not None:
        feature_norm = torch.from_numpy(input_scaler.transform(feature.numpy())).to(device).float()
    else:
        feature_norm = feature
    # scale output data
    if output_scaler is not None:
        target_norm = torch.from_numpy(output_scaler.transform(target.numpy())).to(device).float()
    else:
        target_norm = target


    target_norm_hat = model(feature_norm)
    if  isinstance(target_norm_hat, tuple):
        target_norm_hat = target_norm_hat[0]


    loss = loss_fn(target_norm_hat, target_norm)
    test_loss = loss.item()

    # inverse scale of estimate target
    with torch.no_grad():
        target_hat_mat = output_scaler.inverse_transform(target_norm_hat.detach().numpy())
        target_mat = target.numpy()
        rel_rms_vec = np.sqrt(np.divide(np.mean(np.square(target_hat_mat - target_mat), axis=0),
                          np.mean(np.square(target_mat), axis=0)))

        abs_rms_vec = np.sqrt(np.mean(np.square(target_hat_mat-target_mat), axis=0))

    if verbose:
        print('Test Loss is ', test_loss)
        print('Absolute RMS for each joint are:', abs_rms_vec)
        print('Relative RMS for each joint are:', rel_rms_vec)

    return test_loss, abs_rms_vec, rel_rms_vec

