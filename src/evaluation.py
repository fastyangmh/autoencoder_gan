# import
import torch
import numpy as np

# def


def encoder(model, x, y, USE_CUDA=False):
    model.eval()
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    if USE_CUDA:
        x_tensor = x_tensor.cuda()
        y_tensor = y_tensor.cuda()
    encoded = model.encoder(x_tensor, y_tensor).cpu().data.numpy()
    return encoded


def decoder(model, x, y, USE_CUDA=False):
    model.eval()
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    if USE_CUDA:
        x_tensor = x_tensor.cuda()
        y_tensor = y_tensor.cuda()
    decoded = model.decoder(x_tensor, y_tensor).cpu().data.numpy()
    return decoded
