import math

import numpy as np

import math
import matplotlib.pyplot as plt
import torch
from nflows.flows import MaskedAutoregressiveFlow, SimpleRealNVP, Flow
from statsmodels.tsa.arima.model import ARIMA
from torch import optim
from torch import nn
from torch.nn import KLDivLoss
from ARMAF import ARMaskedAutoregressiveFlow
from AR import sample_AR_p
np.random.seed(0)





p = 1
a = 3
b = 2
n_samples=20
# flow = SimpleRealNVP(features=p+1, hidden_features=30, num_layers=5, num_blocks_per_layer=2)

num_iter = 5000
traj_samples = 100

ar_2_params = [1.0, 0.4, 0.3]
sigma, phi1, phi2 = ar_2_params
variance = sigma**2/(1-phi1**2 - phi2**2)
KLD = KLDivLoss()


def train_reparametrization(flow: Flow, traj_samples=100):

    y_log_pred = torch.zeros((traj_samples, n_samples), requires_grad=True)
    y_log_target = torch.zeros_like(y_log_pred)
    for i in range(traj_samples):
        trajectory = sample_AR_p(20, params=ar_2_params)
        y_log_pred[i] = flow.log_prob(trajectory)
        y_log_target[i] = -n_samples/2 * math.log(2*math.pi*variance) - torch.sum(trajectory**2)/2
    loss = KLD(y_log_pred, y_log_target, log_target=True)
    return loss


def train_ar_1():
    data = sample_AR_p(100000, params=ar_2_params)
    model = ARIMA(list(data), order=(1,0,0))
    model_fit = model.fit()
    # print(model_fit.params)
    # print(model_fit.summary())
    return model_fit

ar_1_params = train_ar_1().params[2:0:-1]
flow = ARMaskedAutoregressiveFlow(AR_params=ar_1_params, features=n_samples, hidden_features=500, num_layers=7, num_blocks_per_layer=5)
# flow = MaskedAutoregressiveFlow(features=n_samples, hidden_features=500, num_layers=7, num_blocks_per_layer=5)
optimizer = optim.Adam(flow.parameters())
# print(ar_1_params)

for i in range(num_iter):
    train = torch.zeros((traj_samples, n_samples))
    for j in range(traj_samples):
        train[j] = sample_AR_p(n_samples, params=ar_2_params,p=p)
    # train = torch.stack([x[i:i+p+1] for i in range(len(x)-p)])
    optimizer.zero_grad()
    flow_loss = -flow.log_prob(inputs=train).mean()
    flow_loss.backward()
    optimizer.step()
    if (i+1) % 10 == 0:
        print(f"iteration: {i}, loss: {flow_loss.data}")






