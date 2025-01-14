import math
from importlib.metadata import distribution

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from torch import optim
from torch.nn import KLDivLoss
import seaborn as sns
import statistics
from AR import sample_AR_p, AR, compute_covariance, compute_big_cov_matrix
from nflows.flows import MaskedAutoregressiveFlow
from ARMAF import ARMaskedAutoregressiveFlow
from scipy.stats import kstest
from pingouin import multivariate_normality
from nflows.distributions.normal import StandardNormal
from myMAF import MyMaskedAutoregressiveFlow

np.random.seed(0)
torch.manual_seed(0)

n_samples = 2
num_iter = 100
traj_samples = 100

q = 1
ar_q_params = torch.cat([torch.tensor([1.0]),(torch.rand(q) - 0.5)/2])
print(compute_covariance(ar_q_params))
sigma, phi = ar_q_params[0], ar_q_params[1:]

def train_ar_p(ar_q_params, p=1):
    data = sample_AR_p(100000, params=ar_q_params)
    model = ARIMA(list(data), order=(p,0,0))
    model_fit = model.fit()
    return model_fit

def frobenius_norm(A,B):
    return np.linalg.norm(A-B)

def estimate_cov_matrix(flow, q, n_samples=50):
    samples = flow.sample(n_samples)
    gamma_est = np.zeros(q+1)
    for i in range(q+1):
        gamma_est[i] = (samples[:,0]*samples[:,i]).mean()
    return gamma_est


PLOT = True

p = 1
ar_p_params = torch.cat([torch.tensor([1.0]),(torch.rand(p) - 0.5)/2])

# flow = MyMaskedAutoregressiveFlow(features=n_samples, hidden_features=100, num_layers=3, num_blocks_per_layer=2)
flow = ARMaskedAutoregressiveFlow(AR_params=ar_p_params, features=n_samples, hidden_features=100, num_layers=3, num_blocks_per_layer=2)
optimizer = optim.Adam(flow.parameters())

KLD = KLDivLoss(log_target=True, reduction='batchmean')

AR_q = AR(ar_q_params, [n_samples])
for i in range(num_iter):
    train = torch.zeros((traj_samples, n_samples))
    for j in range(traj_samples):
        train[j] = sample_AR_p(n_samples, params=ar_q_params,p=q)
    optimizer.zero_grad()
    flow_loss = -flow.log_prob(inputs=train).mean()
    # y_true = AR_q.log_prob(inputs=train)
    # y_pred = flow.log_prob(inputs=train)
    # flow_loss = KLD(y_pred, y_true)
    flow_loss.backward()
    optimizer.step()
    if (i+1) % 100 == 0:
        print(f"iteration: {i}, loss: {flow_loss.data}")

test_iter = 1000
final_estimated_q_params = np.zeros((test_iter,q+1))

distributions = np.zeros((n_samples, test_iter))
ar_distributions = AR_q.sample(test_iter).detach().numpy().T
for k in range(test_iter):
    series = flow.sample(1).squeeze().detach().numpy()
    distributions[:,k] = series

    if n_samples > 10:
        model = ARIMA(list(series), order=(q,0,0))
        model_fit = model.fit()
        estimated_q_params = model_fit.params[q+1:0:-1]
        final_estimated_q_params[k] = estimated_q_params
fig, axs = plt.subplots(2,1)


real_variance = compute_big_cov_matrix(20, ar_q_params)[0][0]

sns.kdeplot(distributions[0], color='b', ax=axs[0])
sns.kdeplot(distributions[1], color='g', ax=axs[1])

axs[0].set(xlabel="variance=" + str(statistics.variance(distributions[0])))
axs[1].set(xlabel="variance=" + str(statistics.variance(distributions[1])))

plt.tight_layout()
plt.show()


#
if n_samples > 10:
    print("Estimated parameters are:", *list(final_estimated_q_params.mean(axis=0)))
    print("Real parameters are:", *list(ar_q_params.detach().numpy()))
#
estimated_cov = estimate_cov_matrix(flow,q)
real_cov = compute_covariance(ar_q_params)
print("Distance between covariance vectors is", np.linalg.norm(estimated_cov - real_cov.numpy()))
#
samples = flow.sample(100).detach().numpy()
print(multivariate_normality(samples, 0.05))

flow_kernel = stats.gaussian_kde(distributions)
ar_kernel = stats.gaussian_kde(ar_distributions)


mins = np.minimum(np.min(distributions, axis=1), np.min(ar_distributions, axis=1))
maxs =  np.maximum(np.max(distributions, axis=1), np.max(ar_distributions, axis=1))
fbounds = list(zip(mins, maxs))

N = 100
slices = tuple([slice(mi, ma, complex(N)) for mi, ma in zip(mins, maxs)])

grid = np.mgrid[slices]
points = grid.reshape(n_samples, -1)

flow_values = flow_kernel(points)
ar_values = ar_kernel(points)

diff = np.max(np.abs(flow_values - ar_values))

cell_area = np.prod((maxs-mins)/N)
integral = np.sum(np.square(flow_values - ar_values))*cell_area
print("max AR(q) density value:", np.max(ar_values))
print("max diff:", diff)
print("integral:", integral)

if PLOT:
    flow_z = np.reshape(flow_values.T, grid[0].shape)
    ar_z = np.reshape(ar_values.T, grid[0].shape)

    fig, axs = plt.subplots(2)

    axs[0].imshow(np.rot90(flow_z), cmap=plt.cm.gist_earth_r,
              extent=[mins[0], maxs[0], mins[1], maxs[1]])
    axs[0].set_xlim([mins[0], maxs[0]])
    axs[0].set_ylim([mins[1], maxs[1]])

    axs[1].imshow(np.rot90(ar_z), cmap=plt.cm.gist_earth_r,
                  extent=[mins[0], maxs[0], mins[1], maxs[1]])
    axs[1].set_xlim([mins[0], maxs[0]])
    axs[1].set_ylim([mins[1], maxs[1]])


    plt.show()



