from pyexpat import features

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import torch
from nflows.flows import MaskedAutoregressiveFlow, SimpleRealNVP
from torch import optim
np.random.seed(0)

def sample_AR_p(nsample, p=1,decay=0.9):
    ar_coefs = [decay ** i for i in range(1, p + 1)]
    ar_params = np.array([1.0] + ar_coefs)
    arma_process = sm.tsa.ArmaProcess(ar_params)
    simulated_data = arma_process.generate_sample(nsample=nsample) #time-series
    return torch.tensor(simulated_data, dtype=torch.float32)

p = 1
# flow = MaskedAutoregressiveFlow(features=p+1, hidden_features=10, num_layers=5, num_blocks_per_layer=2)
flow = SimpleRealNVP(features=p+1, hidden_features=30, num_layers=5, num_blocks_per_layer=2)
optimizer = optim.Adam(flow.parameters())
num_iter = 5000
n_samples=50
for i in range(num_iter):
    x = sample_AR_p(n_samples, p=p)
    train = torch.stack([x[i:i+p+1] for i in range(len(x)-p)])
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=train).mean()
    loss.backward()
    optimizer.step()

    if (i + 1) % 500 == 0:
        xline = train[:,0]
        yline = train[:,1]
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(n_samples-1, n_samples-1)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        train_samples = train.detach().numpy()  # Convert tensor to NumPy array
        plt.xlim(xline.min().item(), xline.max().item())
        plt.ylim(yline.min().item(), yline.max().item())
        plt.scatter(train_samples[:, 0], train_samples[:, 1], color='red', s=5, label="Training Samples", alpha=0.8)
        plt.title('iteration {}'.format(i + 1))
        plt.show()






