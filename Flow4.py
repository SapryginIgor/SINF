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
from AR import sample_AR_p, AR

np.random.seed(0)
torch.manual_seed(0)

n_samples=20
num_iter = 5000
traj_samples = 100

q = 3
ar_q_params = torch.cat([torch.tensor([1.0]),(torch.rand(q) - 0.5)])
sigma, phi = ar_q_params[0], ar_q_params[1:]
variance = sigma**2/(1-(phi**2).sum())

def train_ar_p(ar_q_params, p=1):
    data = sample_AR_p(100000, params=ar_q_params)
    model = ARIMA(list(data), order=(p,0,0))
    model_fit = model.fit()
    # print(model_fit.params)
    # print(model_fit.summary())
    return model_fit

# def KLD(ar_q_params, flow, n_samples, trajectories):
#     AR_q = AR(ar_q_params, [n_samples])
#     Ps_log = torch.empty_like(trajectories)
#     Qs_log = torch.empty_like(trajectories)
#     for i,trajectory in enumerate(trajectories):
#         Ps_log[i] = AR_q.log_prob(trajectory.unsqueeze(0))
#         Qs_log[i] = flow.log_prob(trajectory.unsqueeze(0))
#     KLD = (Ps_log.exp() * (Ps_log - Qs_log)).sum()
#     return KLD



p = 5
model_fit = train_ar_p(ar_q_params, p=p)
print(model_fit.summary())
ar_p_params = model_fit.params[p+1:0:-1]
# flow = ARMaskedAutoregressiveFlow(AR_params=ar_p_params, features=n_samples, hidden_features=100, num_layers=3, num_blocks_per_layer=2)
flow = MaskedAutoregressiveFlow(features=n_samples, hidden_features=100, num_layers=3, num_blocks_per_layer=2)
optimizer = optim.Adam(flow.parameters())
KLD = KLDivLoss(log_target=True, reduction='batchmean')
# print(ar_1_params)
AR_q = AR(ar_q_params, [n_samples])
for i in range(num_iter):
    train = torch.zeros((traj_samples, n_samples))
    for j in range(traj_samples):
        train[j] = sample_AR_p(n_samples, params=ar_q_params,p=p)
    # train = torch.stack([x[i:i+p+1] for i in range(len(x)-p)])
    optimizer.zero_grad()
    y_true = AR_q.log_prob(inputs=train)
    y_pred = flow.log_prob(inputs=train)
    # flow_loss = -flow.log_prob(inputs=train).mean()
    flow_loss = KLD(y_pred, y_true)
    flow_loss.backward()
    optimizer.step()
    if (i+1) % 100 == 0:
        print(f"iteration: {i}, loss: {flow_loss.data}")
        # print(f"iteration: {i}, KLD loss: {KLD(ar_q_params, flow,n_samples,train)}")






