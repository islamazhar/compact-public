import sys; sys.path.append("..")
import os 

import torch
import torch.nn as nn


from torchvision.datasets import  CIFAR10
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor


from CIFAR10.CNN import (
    CnnModel, get_name, 
    to_device, 
    DeviceDataLoader, 
    get_default_device,
    secure_evaluate,
    evaluate
)



DATASET_DIR = "/home/rahul/nobackup/datasets"
MODEL_DIR = f"{os.getcwd()}/../saved_models"
device = get_default_device()

# activationFunc = nn.GELU
activationFunc  = nn.SiLU
act_name = get_name (activationFunc)


model = CnnModel(act_func = activationFunc)
model_loc = f'{MODEL_DIR}/cifar10_cnn_{act_name}.pth'
model.load_state_dict(torch.load(model_loc))
to_device(model, device)
model = torch.compile(model)


BATCH_SIZE =  128
SAMPLE_SIZE = 1000

test_data = CIFAR10 (root= DATASET_DIR, download = True, transform = ToTensor())
test_dl = DataLoader (test_data, BATCH_SIZE)
test_dl = DeviceDataLoader (test_dl, device)

print("Loaded model, test data")

from compact.main import (
    func_reciprocal, 
    func_exp 
)

f, n = 48, 96

def silu_func(x):
    return x * func_reciprocal((1 + func_exp(-x)))

silu_config = {
    "function": silu_func, # function config.
    "range": (-20, 20), # range to approximate
    "k_max": 30, # set the maximum order.
    "tol": 1e-3, # precision config.
    "ms": 10000, # maximum samples.
    "zero_mask": 1e-3,
    "max_breaks": 10000,
    "n": n, # <n, f> fixed-point config.
    "f": f
}
config = silu_config


# def gelu_func(x):
#     return x * func_reciprocal((1 + func_exp(-1.702 * x)))


from  utils.main_utils import GenerateMPCApproximation




#================================================================================
eta = evaluate (model, test_dl)['val_acc']
print (f"eta = {eta:.2f}")

hi_tol = 10
lo_tol = 1e-3


for _ in range(10):
    mid_tol = lo_tol + (hi_tol - lo_tol)/2.0
    config["tol"] = mid_tol 
    act_approx = GenerateMPCApproximation (config) # MPCApproximation class
    act_approx.to_device (device)
    history = secure_evaluate (model, test_dl, act_approx)
    eta_prime = history['val_acc']
    print(f"tol={mid_tol:.2f}\teta_prime={eta_prime:.2f}")
    
    acc_loss = (eta -eta_prime)/eta
    if  acc_loss < 1e-2:
        lo_tol = mid_tol # increase the tolerance level  
    else:
        hi_tol = mid_tol # not working decrease the tolerance level
            
print(lo_tol)
#===============================================================================
