import torch
import numpy as np
from os import remove
import matplotlib.pyplot as plt



def train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch):
    train_losses = []
    valid_losses = []  # to track the validation loss as the model trains
    avg_train_losses = []  # to track the average training loss per epoch as the model trains
    avg_valid_losses = []  # to track the average validation loss per epoch as the model trains

    for t in range(max_training_epoch):
        train_losses = []
        valid_losses = []
        for feature, target in train_loader:
            target_hat = model(feature)
            loss = loss_fn(target_hat, target)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            train_losses.append(loss.item())
        for feature, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            target_hat = model(feature)
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



def add_layer_to_autoencoder(base_model, add_layer_model):
    # n-1 layer
    layer_list = list(base_model.children())[:-1]
    # add new layer
    layer_list.append(add_layer_model)
    # add output layer
    layer_list.append(list(base_model.children())[-1])

    new_model = torch.nn.Sequential(*layer_list)

    layer_number = 0
    for _ in new_model.parameters():
        layer_number +=1

    # mark all first n-2 layer freeze
    count = 0
    for param in new_model.parameters():
        count +=1
        if count<=layer_number-2:
            param.requires_grad = False
        else:
            param.requires_grad = True

    return new_model

