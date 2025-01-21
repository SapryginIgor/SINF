import numpy as np
import torch
import statsmodels.api as sm
from nflows.distributions.base import Distribution
from nflows.utils import torchutils

def compute_covariance(ma_params):
    sigma2 = ma_params[0]
    q = len(ma_params)-1
    theta = np.zeros(q+1)
    theta[0] = 1
    theta[1:] = ma_params[1:]
    gamma = np.zeros(q+1)
    for k in range(q+1):
        for s in range(k, q+1):
            gamma[k] += theta[s]*theta[s-k]
    # print(sigma2)
    gamma *= sigma2.item()
    return gamma


def compute_big_cov_matrix(n, ma_params):
    gamma = compute_covariance(ma_params)
    q = len(gamma)-1
    big_gamma = np.zeros(n)
    big_gamma[:q+1] = gamma
    arr = np.arange(n)
    dist_mtx = np.abs(arr[None] - arr[...,None])
    return torch.tensor(big_gamma)[dist_mtx]


def sample_MA_q(nsample, q=1,params=None, decay=0.9):
    if params is None:
        ma_coefs = [decay ** i for i in range(1, q + 1)]
        ma_params = np.array([1.0] + ma_coefs)
    else:
        ma_params = params
    arma_process = sm.tsa.ArmaProcess(ma=ma_params)
    simulated_data = arma_process.generate_sample(nsample=nsample, axis=-1) #time-series
    return torch.tensor(simulated_data, dtype=torch.float32)

class MA(Distribution):

    def __init__(self, shape, params):
        super().__init__()
        self.params = params
        self._shape = torch.Size(shape)
        cov_matrix = compute_big_cov_matrix(np.prod(shape), self.params)
        self.inv_cov_matrix = torch.linalg.inv(cov_matrix).to(torch.float32)
        self.det_cov_matrix = torch.linalg.det(cov_matrix)
        self.register_buffer("const_log",
                             0.5 * (np.prod(shape) * np.log(2 * np.pi) + torch.log(self.det_cov_matrix)).to(
                                 torch.float64), persistent=False)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = -0.5 * \
                     torch.matmul(torch.matmul(inputs[..., None, :], self.inv_cov_matrix), inputs[...,None])
        res = neg_energy - self.const_log
        return res.squeeze()

    def _sample(self, num_samples, context):
        if context is None:
            return sample_MA_q((num_samples, np.prod(self._shape)), params=self.params)
        else:
            raise NotImplementedError("context support is not implemented yet")

    def _mean(self, context):
        if context is None:
            return self.const_log.new_zeros(self._shape)
        else:
            return context.new_zeros(context.shape[0], *self._shape)

if __name__ == "__main__":
    ar_params = np.array([1.0, 0.5, 0.4, -0.2])
    cov = None
    print(cov)