
import sys
sys.path.append("..")

import os
import pickle 
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import  CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as tt


import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

from CIFAR10.CNN import *


from torchvision.transforms import ToTensor

# DATASET_DIR = "/nobackup/mazharul/"
# MODEL_DIR = os.getcwd() + "/../saved_models"

DATASET_DIR = "/home/rahul/nobackup/datasets"
# DATASET_DIR = "/home/rahul/pwdata/datasets/"
MODEL_DIR = os.getcwd() + "/../saved_models"

train_data = CIFAR10(download=True, root = DATASET_DIR, transform=ToTensor())
test_data = CIFAR10(root= DATASET_DIR, train=False, transform=ToTensor())


random_seed = 42
torch.manual_seed(random_seed);

val_size = 5000
train_size = len(train_data) - val_size

train_data, val_data = random_split(train_data, [train_size, val_size])
len(train_data), len(val_data)


BATCH_SIZE = 128
train_dl = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_data, BATCH_SIZE*2, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_data,BATCH_SIZE,num_workers=4, pin_memory=True)


device = get_default_device()
# device = 'cpu'
print(device)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


""" Training the models 
"""
num_epochs = 5
opt_func = torch.optim.Adam
lr = 0.001

# activationFuncs = [nn.SiLU, nn.GELU, nn.Mish]
activationFuncs = [nn.SiLU]
# activationFuncs = [nn.GELU]

histories = {}
saved_models = {}


for activationFunc in activationFuncs:
    
    act_name = get_name(activationFunc)
    print("Starting .... ", act_name)
    # mpcFunctions = NFGenApprox(ac_func=act_name, tol=tol)
    model = CnnModel(act_func=activationFunc)
    model_loc = f'{MODEL_DIR}/cifar10_cnn_{act_name}.pth'        
    model.load_state_dict(torch.load(model_loc))
    to_device(model, device)
    histories[act_name] = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    model_loc = f'{MODEL_DIR}/cifar10_cnn_{act_name}.pth'
    torch.save(model.state_dict(), model_loc)
    # saved_models[act_name] =  model                                  


