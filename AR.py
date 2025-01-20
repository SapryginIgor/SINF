import numpy as np
import torch
import statsmodels.api as sm
from nflows.distributions.base import Distribution
from nflows.utils import torchutils


def sample_AR_p(nsample, p=1,params=None, decay=0.9):
    if params is None:
        ar_coefs = [decay ** i for i in range(1, p + 1)]
        ar_params = np.array([1.0] + ar_coefs)
    else:
        ar_params = params
    arma_process = sm.tsa.ArmaProcess(ar_params)
    simulated_data = arma_process.generate_sample(nsample=nsample, axis=-1) #time-series
    return torch.tensor(simulated_data, dtype=torch.float32)

def compute_covariance(ar_params):
    sigma2 = ar_params[0]
    phi = ar_params[1:]
    p = len(phi)
    A = torch.zeros((p+1, p+1))
    arr = np.arange(p)
    dist_mtx = abs(arr[None] - arr[...,None])
    for i in range(p):
        for j in range(p):
            A[i, dist_mtx[i,j]] += phi[j]
        A[i][i+1] -= 1
    tmp = [-1]
    tmp += list(phi)
    A[p] = torch.tensor(tmp)
    B = torch.zeros(p+1)
    B[p] = -sigma2
    gamma = torch.linalg.solve(A=A, B=B)
    return gamma

def compute_big_cov_matrix(n, ar_params):
    gamma = compute_covariance(ar_params).numpy()
    p = len(gamma)-1
    big_gamma = np.zeros(n)
    big_gamma[:p+1] = gamma
    phi = np.array(ar_params[1:])
    for i in range(p+1,n):
        big_gamma[i] = np.dot(phi, big_gamma[i-1:i-1-p:-1])
    arr = np.arange(n)
    dist_mtx = np.abs(arr[None] - arr[...,None])
    return torch.tensor(big_gamma)[dist_mtx]

def true_log_density(params, inputs):
    shape = inputs.shape[1:]
    sigma2 = params[0]
    phi = params[1:]
    const =  (0.5 * np.prod(shape) * np.log(2 * np.pi * sigma2))
    s = torch.zeros(inputs.shape[0])
    last = torch.zeros((inputs.shape[0], phi.shape[0]))
    for i in range(np.prod(shape)):
        mu = (phi[None] * last).sum(dim=1)
        s -= (inputs[..., i] - mu)**2/(2*sigma2)
        last =  torch.roll(last, 1, dims=1)
        last[:, 0] = inputs[:,i]
    return s - const






class AR(Distribution):

    def __init__(self, shape, params):
        super().__init__()
        self.params = params
        self._shape = torch.Size(shape)
        cov_matrix = compute_big_cov_matrix(np.prod(shape), self.params)
        self.inv_cov_matrix = torch.linalg.inv(cov_matrix).to(torch.float32)
        self.det_cov_matrix = torch.linalg.det(cov_matrix)
        self.register_buffer("const_log", 0.5*(np.prod(shape)*np.log(2*np.pi)+torch.log(self.det_cov_matrix)).to(torch.float64), persistent=False)

    def is_stationary(self):
        proc = sm.tsa.ArmaProcess(self.params)
        return proc.isstationary

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
        expected = true_log_density(self.params, inputs).reshape(inputs.shape[0],1,1)
        return res.squeeze()

    def _sample(self, num_samples, context):
        if context is None:
            return sample_AR_p((num_samples, np.prod(self._shape)), params=self.params)
        else:
            raise NotImplementedError("context support is not implemented yet")

    def _mean(self, context):
        if context is None:
            return self.const_log.new_zeros(self._shape)
        else:
            return context.new_zeros(context.shape[0], *self._shape)


if __name__ == "__main__":
    ar_params = np.array([1.0, 0.5, 0.4, -0.2])
    cov = compute_big_cov_matrix(10, ar_params)
    print(cov)