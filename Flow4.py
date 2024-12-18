import matplotlib.pyplot as plt
import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA
from torch import optim
from torch.nn import KLDivLoss

from AR import sample_AR_p, AR, compute_covariance
from ARMAF import ARMaskedAutoregressiveFlow
from scipy.stats import kstest
from pingouin import multivariate_normality


np.random.seed(0)
torch.manual_seed(0)

n_samples=20
num_iter = 200
traj_samples = 100

q = 2
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

def frobenius_norm(A,B):
    return np.linalg.norm(A-B)

def estimate_cov_matrix(flow, q, n_samples=50):
    samples = flow.sample(n_samples)
    gamma_est = np.zeros(q+1)
    for i in range(q+1):
        gamma_est[i] = (samples[:,0]*samples[:,i]).mean()
    return gamma_est





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
# model_fit = train_ar_p(ar_q_params, p=p)
# print(model_fit.summary())
# ar_p_params = model_fit.params[p+1:0:-1]
ar_p_params = torch.cat([torch.tensor([1.0]),(torch.rand(p) - 0.5)])
flow = ARMaskedAutoregressiveFlow(AR_params=ar_p_params, features=n_samples, hidden_features=100, num_layers=3, num_blocks_per_layer=2)
# flow = MaskedAutoregressiveFlow(features=n_samples, hidden_features=100, num_layers=3, num_blocks_per_layer=2)
optimizer = optim.Adam(flow.parameters())
KLD = KLDivLoss(log_target=True, reduction='batchmean')
# print(ar_1_params)
AR_q = AR(ar_q_params, [n_samples])
for i in range(num_iter):
    train = torch.zeros((traj_samples, n_samples))
    for j in range(traj_samples):
        train[j] = sample_AR_p(n_samples, params=ar_q_params,p=q)
    # train = torch.stack([x[i:i+p+1] for i in range(len(x)-p)])
    optimizer.zero_grad()
    y_true = AR_q.log_prob(inputs=train)
    y_pred = flow.log_prob(inputs=train)
    flow_loss = -flow.log_prob(inputs=train).mean()
    # flow_loss = KLD(y_pred, y_true)
    flow_loss.backward()
    optimizer.step()
    if (i+1) % 100 == 0:
        print(f"iteration: {i}, loss: {flow_loss.data}")
        # print(f"iteration: {i}, KLD loss: {KLD(ar_q_params, flow,n_samples,train)}")

final_estimated_q_params = np.zeros((100,q+1))

distributions = np.zeros((n_samples, 1000))
for k in range(1000):
    series = flow.sample(1).squeeze().detach().numpy()
    distributions[:,k] = series
    # model = ARIMA(list(series), order=(q,0,0))
    # model_fit = model.fit()
    # estimated_q_params = model_fit.params[q+1:0:-1]
    # final_estimated_q_params[k] = estimated_q_params
fig, ax = plt.subplots(3,1)
ax[0].hist(distributions[1])
ax[1].hist(distributions[10])
ax[2].hist(distributions[15])
plt.show()
# print(model_fit.summary())

real_variance = compute_covariance(ar_q_params)

print("Estimated parameters are:", *list(final_estimated_q_params.mean(axis=0)))
print("Real parameters are:", *list(ar_q_params.detach().numpy()))

estimated_cov = estimate_cov_matrix(flow,q)
real_cov = compute_covariance(ar_q_params)
print("Distance between covariance vectors is", np.linalg.norm(estimated_cov - real_cov.numpy()))

samples = flow.sample(100).detach().numpy()
print(multivariate_normality(samples, 0.05))

