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

class AR(Distribution):

    def __init__(self, params, shape):
        super().__init__()
        # self.register_buffer("params", torch.tensor(params, dtype=torch.float64), persistent=False)
        self.params = params
        sigma2 = self.params[0]
        phi = self.params[1:]
        self._shape = torch.Size(shape)
        self.variance = sigma2 / (1 - (phi**2).sum())
        if len(phi) > 1:
            raise NotImplementedError("No cov matrix")
        else:
            arr = np.arange(np.prod(shape))
            cov_matrix = self.variance * np.power(phi[0], np.abs(arr[None] - arr[:, None]))
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
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)