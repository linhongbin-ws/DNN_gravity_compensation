import torch
from loadDataTool import load_data_dir
import numpy as np
from Net import Lagrange_Net

def predict(type, model, input_mat, input_scaler, output_scaler, delta_q=None, w_vec=None):
    feature_norm = torch.from_numpy(input_scaler.transform(input_mat.numpy())).to('cpu').float()
    if type == 'Base':
        target_norm_hat = model(feature_norm)
    elif type == 'Lagrangian':
        target_norm_hat = Lagrange_Net(model, feature_norm, delta_q, w_vec, device='cpu')
    target_hat_mat = output_scaler.inverse_transform(target_norm_hat.detach().numpy())
    target_hat = torch.from_numpy(target_hat_mat)
    return target_hat

# def predict_base(model, input_mat, input_scaler, output_scaler):
#     feature_norm = torch.from_numpy(input_scaler.transform(input_mat.numpy())).to('cpu').float()
#     target_norm_hat = model(feature_norm)
#     target_hat_mat = output_scaler.inverse_transform(target_norm_hat.detach().numpy())
#     target_hat = torch.from_numpy(target_hat_mat)
#     return target_hat
#
# def predict_lagrangian(model, input_mat, input_scaler, output_scaler,delta_q,w_vec):
#     feature_norm = torch.from_numpy(input_scaler.transform(input_mat.numpy())).to('cpu').float()
#     target_norm_hat = Lagrange_Net(model, feature_norm, delta_q, w_vec, device='cpu')
#     target_hat_mat = output_scaler.inverse_transform(target_norm_hat.detach().numpy())
#     target_hat = torch.from_numpy(target_hat_mat)
#     return target_hat


def test(type, model, loss_fn, test_data_path, input_scaler, output_scaler, device, delta_q=None, w_vec=None, verbose=True):
    # test model
    test_dataset = load_data_dir(test_data_path, device=device, is_scale=False)
    feature = test_dataset.x_data
    target = test_dataset.y_data

    # scale input data
    feature_norm = torch.from_numpy(input_scaler.transform(feature.numpy())).to(device).float()
    # scale output data
    target_norm = torch.from_numpy(output_scaler.transform(target.numpy())).to(device).float()

    if type=='Base':
        target_norm_hat = model(feature_norm)
    elif type=='Lagrangian':
        target_norm_hat = Lagrange_Net(model, feature_norm, delta_q, w_vec, device='cpu')


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

# def test_lagrangian(model, loss_fn, test_data_path, input_scaler, output_scaler, delta_q, w_vec, device='cpu', verbose=True):
#     # test model
#     test_dataset = load_data_dir(test_data_path, device=device, is_scale=False)
#     feature = test_dataset.x_data
#     target = test_dataset.y_data
#
#     # scale input data
#     feature_norm = torch.from_numpy(input_scaler.transform(feature.numpy())).to(device).float()
#     # scale output data
#     target_norm = torch.from_numpy(output_scaler.transform(target.numpy())).to(device).float()
#
#     target_norm_hat = Lagrange_Net(model, feature_norm, delta_q, w_vec, device='cpu')
#     loss = loss_fn(target_norm_hat, target_norm)
#     test_loss = loss.item()
#
#     # inverse scale of estimate target
#     with torch.no_grad():
#         target_hat_mat = output_scaler.inverse_transform(target_norm_hat.detach().numpy())
#         target_mat = target.numpy()
#         rel_rms_vec = np.sqrt(np.divide(np.mean(np.square(target_hat_mat - target_mat), axis=0),
#                           np.mean(np.square(target_mat), axis=0)))
#
#         abs_rms_vec = np.sqrt(np.mean(np.square(target_hat_mat-target_mat), axis=0))
#
#     if verbose:
#         print('Test Loss is ', test_loss)
#         print('Absolute RMS for each joint are:', abs_rms_vec)
#         print('Relative RMS for each joint are:', rel_rms_vec)
#
#     return test_loss, abs_rms_vec, rel_rms_vec