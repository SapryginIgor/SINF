import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import torch
from nflows import transforms, distributions, flows
from nflows.distributions.normal import StandardNormal
from nflows.flows import Flow
from nflows.transforms import ReversePermutation, MaskedAffineAutoregressiveTransform, CompositeTransform
from torch import optim
from torch.onnx.symbolic_helper import parse_args


# class AR_p(distributions.Distribution):
#
#     def forward(self, *args):
#         pass
#
#     def _log_prob(self, inputs, context=None):
#         pass
#
#     def _sample(self, num_samples, context=None, batch_size=None):
#         pass
#     def _mean(self, context):
#         pass

np.random.seed(0)

def sample_AR_p(nsample, p=1,decay=0.9):
    ar_coefs = [decay ** i for i in range(1, p + 1)]
    ar_params = np.array(ar_coefs)
    arma_process = sm.tsa.ArmaProcess(ar_params)
    simulated_data = arma_process.generate_sample(nsample=nsample) #time-series
    return torch.tensor(simulated_data, dtype=torch.float32)





num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                          hidden_features=4))
transform = CompositeTransform(transforms)
flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())
a = [1,2,3]
b = [5, 10, 2]
print(list(zip(a,b)))
num_iter = 5000
for i in range(num_iter):
    x = sample_AR_p(100, p=1)
    x = torch.stack((x, torch.arange(100)), dim=1)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()

    if (i + 1) % 500 == 0:
        xline = torch.linspace(-1.5, 2.5, 100)
        yline = torch.linspace(-.75, 1.25, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        plt.show()






