# import
import torch
from os.path import isfile, join
from glob import glob
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np

#global parameters
USE_CUDA = torch.cuda.is_available()

# def


def train_loop(dataloader, model, optimizer, criterion, epochs):
    train_loader, test_loader = dataloader
    history = []
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = []
        for x, y in train_loader:
            if USE_CUDA:
                x, y = x.cuda(), y.cuda()
            xhat = model(x, y)
            loss = criterion(xhat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        history.append(np.mean(total_loss))
    return history
