import sys; sys.path.append("..")

import time
import os
import math


import sympy as sp
import numpy as np



from CIFAR10.CNN import get_default_device, secure_evaluate, evaluate
from  utils.main_utils import GenerateMPCApproximation

# device = 'cpu'
device = get_default_device()


def func_reciprocal(x):
        return 1 / x

def func_exp(x, lib=sp):
    return lib.exp(x)

def func_tanh(x, lib=sp):
    return lib.tanh(x)

def func_ln(x, lib=sp):
    return lib.ln(x)

def func_sqrt(x, lib=sp):
    return lib.sqrt(x)


""" perform binary search over tol
"""
def GenAccurateApprox (theta, model, config, test_dl, eta):
    
    
    # print(f"eta => {eta}")

    
    config["n"] = theta.n
    config["f"] = theta.f
    config["k_max"] = theta.k
    config["m_max"] = theta.m 
    
    best_act_approx = None 
    
    
    hi_tol = 1000
    lo_tol = 1e-6
    
    for epoch in range(50):
        # print(epoch)
        mid = (hi_tol + (hi_tol - lo_tol))/2.0
        config["tol"] = mid 
        act_approx = GenerateMPCApproximation (config) # MPCApproximation class
        act_approx.to_device(device)
        history = secure_evaluate (model, test_dl, act_approx)
        eta_prime = history['val_acc']
        print(f"epoch = {epoch}\tmid={mid}\t{eta_prime}\t{eta}")
        # exit(1)
        acc_loss = (eta -eta_prime)/eta
        
        if  acc_loss < 1e-1:
            hi_tol = mid
            best_act_approx = act_approx 
        else:
            lo_tol = mid 
            
    return best_act_approx
            
            
"""Global vars
"""

nu = 1e-2
chi = 0.2

class Theta:
    def __init__(self, m, k, n, f):
        self.m = m
        self.k = k
        self.n = n
        self.f = f
        
    def GenerateNeighbour(self):
        return Theta (self.m, self.k, self.n, self.f)
    
def GetInferenceTime (act_approx):
    return 1
            
def FindBestPiecePoly(model, config, test_dl):
    eta = evaluate (model, test_dl)['val_acc']
    
    print(f"eta={eta}")
    
    m0 = 1000
    k0 = 50
    n0, f0 = 128, 68
    
    theta_0 = Theta (m0, k0, n0, f0) 
    cur_act_approx = GenAccurateApprox (theta_0, model, config, test_dl, eta)
    
    if cur_act_approx is None:
        assert False 
        
        
    theta_cur = theta_0
    inference_time_best = GetInferenceTime (cur_act_approx)
    
    for i in range(10):
        temp = chi/np.log10(1+i)
        theta_i = theta_cur.GenerateNeighbour()
        act_approx = GenAccurateApprox (theta_i, model, config, test_dl, eta)
        
        if act_approx is None:
            continue 
        else:
            r = np.random.rand(1)[0]
            inference_time_cur = GetInferenceTime (cur_act_approx)
            diff = np.exp(inference_time_best - inference_time_cur) / temp
            if min(1, diff) > r:
                theta_cur = theta_i 
                inference_time_best = inference_time_cur
                cur_act_approx = act_approx
                
    return cur_act_approx    