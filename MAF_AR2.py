import math
from importlib.metadata import distribution

import matplotlib.pyplot as plt
import numpy as np
import torch
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

n_samples = 10
num_iter = 100
traj_samples = 100

q = 2
ar_q_params = torch.cat([torch.tensor([1.0]),(torch.rand(q) - 0.5)])
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


PLOT = False

flow = MyMaskedAutoregressiveFlow(features=n_samples, hidden_features=100, num_layers=3, num_blocks_per_layer=2)
optimizer = optim.Adam(flow.parameters())

AR_q = AR(ar_q_params, [n_samples])
for i in range(num_iter):
    train = torch.zeros((traj_samples, n_samples))
    for j in range(traj_samples):
        train[j] = sample_AR_p(n_samples, params=ar_q_params,p=q)
    optimizer.zero_grad()
    flow_loss = -flow.log_prob(inputs=train).mean()
    flow_loss.backward()
    optimizer.step()
    if (i+1) % 100 == 0:
        print(f"iteration: {i}, loss: {flow_loss.data}")
        if PLOT:
            N = 100
            xline = torch.linspace(-2, 2, N)
            yline = torch.linspace(-2, 2, N)
            xgrid, ygrid = torch.meshgrid(xline, yline)
            xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

            with torch.no_grad():
                tmp = flow.log_prob(xyinput).exp()
                zgrid = tmp.reshape(N, N)

            plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
            train_samples = train.detach().numpy()  # Convert tensor to NumPy array
            plt.xlim(xline.min().item(), xline.max().item())
            plt.ylim(yline.min().item(), yline.max().item())
            plt.scatter(train_samples[:, 0], train_samples[:, 1], color='red', s=5, label="Training Samples", alpha=0.8)
            plt.title('iteration {}'.format(i + 1))
            plt.show()

test_iter = 100
final_estimated_q_params = np.zeros((test_iter,q+1))

distributions = np.zeros((n_samples, test_iter))
ar_distributions = AR_q.sample(test_iter)
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

nbins = 100
#minimum value element wise from both arrays
flat_dist = distributions.flatten()
flat_ar_dist = ar_distributions.numpy().flatten()
min = math.floor(np.min(np.minimum(flat_dist, flat_ar_dist)))
#maximum value element wise from both arrays
max = math.ceil(np.max(np.maximum(flat_dist, flat_ar_dist)))
#histogram is build with fixed min and max values
bins = np.linspace(min, max, max-min+1)
hist1, _ = np.histogram(flat_dist, bins=bins, density=True)
hist2, _ = np.histogram(flat_ar_dist, bins=bins, density=True)

#makes sense to have only positive values
diff = np.square(hist1 - hist2)
# plt.bar(bins[:-1],hist1,width=1)
# plt.bar(bins[:-1],hist2,width=1)
# plt.bar(bins[:-1],diff,width=1)
fig2, axs2 = plt.subplots(3,1)
axs2[0].bar(bins[:-1],hist1,width=1)
axs2[1].bar(bins[:-1],hist2,width=1)
axs2[2].bar(bins[:-1],diff,width=1)
print(np.sqrt(np.sum(diff)))
plt.show()


