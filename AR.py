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
    simulated_data = arma_process.generate_sample(nsample=nsample) #time-series
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
    print(torch.tensor(tmp))
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
    # if p == 1:
    #     Vn = phi**(n-1)*gamma.item()
    #     return Vn
    # tmp = torch.concat([torch.eye(p-1), torch.zeros(p-1)], dim=1)
    # A = torch.concat([phi[None], tmp], dim=0)
    # An = torch.linalg.matrix_power(A, n)
    # Vn = torch.matmul(An, gamma)
    arr = np.arange(n)
    dist_mtx = np.abs(arr[None] - arr[...,None])

    return torch.tensor(big_gamma)[dist_mtx]





class AR(Distribution):

    def __init__(self, params, shape):
        super().__init__()
        # self.register_buffer("params", torch.tensor(params, dtype=torch.float64), persistent=False)
        self.params = params
        self._shape = torch.Size(shape)
        cov_matrix = compute_big_cov_matrix(np.prod(shape), params)
        self.inv_cov_matrix = torch.tensor(np.linalg.inv(cov_matrix), dtype=torch.float32)
        self.det_cov_matrix = np.linalg.det(cov_matrix)
        self.register_buffer("const_log", torch.tensor(0.5*(np.prod(shape)*np.log(2*np.pi)+np.log(self.det_cov_matrix)) , dtype=torch.float64), persistent=False)
        # self.register_buffer("_log_z",
        #                      torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
        #                                   dtype=torch.float64),
        #                      persistent=False)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        # print(inputs[..., None, :].dtype)
        # print(self.inv_cov_matrix.dtype)
        neg_energy = -0.5 * \
                     torch.matmul(torch.matmul(inputs[..., None, :], self.inv_cov_matrix), inputs[...,None])
                    # torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
        # print(f"log prob is {neg_energy - self.const_log}")
        return neg_energy - self.const_log

    def _sample(self, num_samples, context):
        if context is None:
            # return torch.randn(num_samples, *self._shape, device=self._log_z.device)
            return sample_AR_p(np.prod(self._shape), params=self.params)
        else:
            # # The value of the context is ignored, only its size and device are taken into account.
            # context_size = context.shape[0]
            # samples = torch.randn(context_size * num_samples, *self._shape,
            #                       device=context.device)
            # return torchutils.split_leading_dim(samples, [context_size, num_samples])
            raise NotImplementedError("No context is implied")

    def _mean(self, context):
        if context is None:
            return self.const_log.new_zeros(self._shape)
        else:
            return context.new_zeros(context.shape[0], *self._shape)


if __name__ == "__main__":
    ar_params = np.array([1.0, 0.5, 0.4, -0.2])
    cov = compute_big_cov_matrix(10, ar_params)
    print(cov)