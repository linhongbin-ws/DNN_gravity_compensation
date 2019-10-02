import torch
from loadDataTool import load_data_dir
import numpy as np

def test(model, loss_fn, test_data_path, input_scaler, output_scaler, device='cpu'):
    # test model
    test_dataset = load_data_dir(test_data_path, device=device, is_scale=False)
    feature = test_dataset.x_data
    target = test_dataset.y_data

    # scale input data
    feature_norm = torch.from_numpy(input_scaler.transform(feature.numpy())).to(device).float()
    # scale output data
    target_norm = torch.from_numpy(output_scaler.transform(target.numpy())).to(device).float()

    target_norm_hat = model(feature_norm)
    loss = loss_fn(target_norm_hat, target_norm)
    test_loss = loss.item()

    # inverse scale of estimate target
    target_hat_mat = output_scaler.inverse_transform(target_norm_hat.numpy())
    target_mat = target.numpy()
    rel_rms_vec = np.sqrt(np.divide(np.sum(np.square(target_hat_mat - target_mat), axis=0),
                      np.sum(np.square(target_mat), axis=0)))

    abs_rms_vec = np.sqrt(np.mean(np.square(target_hat_mat-target_mat)))

    return test_loss, abs_rms_vec, rel_rms_vec