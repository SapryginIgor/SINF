from scipy.stats import qmc
import numpy as np
import time
import torch
import mpmath as mp
from AR import AR, ConditionalAR
from MA import MA, ConditionalMA
from context_train import make_AR_p, make_MA_q
from myMAF import MyMaskedAutoregressiveFlow
from myNVP import MySimpleRealNVP
# dims = 20
scale = np.pi
from mpmath import mp, exp,sqrt, mpf
from nflows.distributions.normal import StandardNormal



# dims = 20
# sampler = qmc.Sobol(d=dims, scramble=True)
# mp.dps = 50
# mp.pretty=True
# t = sampler.random_base2(m=14)
# u = mp.pi * (t - 0.5)
# mptan = np.vectorize(mp.tan)
# x = mptan(u)
# mpcos = np.vectorize(mp.cos)
# jac = 1/np.prod(mpcos(u)**2, axis=-1)
# mpexp = np.vectorize(mp.exp)
#
# start = time.time()
# tmp = (mpexp((-x**2/2).sum(axis=1)) / mp.sqrt((2 * mp.pi) ** dims))*mp.pi**dims
# print(np.mean(jac*tmp))
# print("time is: {}".format(time.time()-start))

torch.manual_seed(0)
np.random.seed(42)
flow_type = ['MAF']
base_distributions = [(ConditionalAR, 2)]
target_distributions = [(AR, 2)]
n_samples = [20]
test_dist = AR(params=torch.cat([torch.tensor([1.0]), 4*(torch.rand(2) - 0.5)]), shape=n_samples)
# for bd in base_distributions:
#     for td in target_distributions:
#         for ft in flow_type:
#             for ns in n_samples:
#                 if bd[1] and ns <= bd[1]:
#                     continue
#                 elif td[1] and ns <= td[1]:
#                     continue
#                 if bd[0] is StandardNormal:
#                     base_dist = StandardNormal([ns])
#                 elif bd[0] is AR:
#                     base_dist = make_AR_p(p=bd[1], n=ns)
#                 elif bd[0] is MA:
#                     base_dist = make_MA_q(q=bd[1], n=ns)
#                 if ft == 'MAF':
#                     flow = MyMaskedAutoregressiveFlow(features=ns, hidden_features=20, num_layers=3,
#                                                       num_blocks_per_layer=2, distribution=base_dist)
#                 elif ft == 'RealNVP':
#                     flow = MySimpleRealNVP(features=ns, hidden_features=20, num_layers=3, num_blocks_per_layer=3,
#                                            distribution=base_dist)

dims = n_samples[0]
sampler = torch.quasirandom.SobolEngine(dimension=dims, scramble=True)
# mp.dps = 50
# mp.pretty = True
t = sampler.draw_base2(m=23, dtype=torch.float64)

u = torch.pi * (t - 0.5)
# mptan = np.vectorize(mp.tan)
x = torch.tan(u)
# mpcos = np.vectorize(np.cos)
jac = 1 / torch.prod(torch.cos(u) ** 2, dim=-1)


start = time.time()
# mpexp = np.vectorize(mp.exp)
# flow.double()
tmp = torch.exp(test_dist.log_prob(x)) * torch.pi ** dims
integral = torch.mean(jac*tmp)
var = ((jac*tmp - integral)**2).mean()
print(integral)
print("variance: ", var)
print("time is: {}".format(time.time() - start))