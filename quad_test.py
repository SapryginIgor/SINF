from scipy.stats import qmc
import numpy as np
import mpmath as mp
dims = 100
scale = 10

sampler = qmc.Sobol(d=dims, scramble=True)
sample = sampler.random_base2(m=12)
sample = qmc.scale(sample, l_bounds=[-scale / 2 for _ in range(dims)],  u_bounds=[scale / 2 for _ in range(dims)])
tmp = np.exp((-sample**2/2).sum(axis=1)) / mp.sqrt((2 * mp.pi) ** dims)
print((scale**dims) * np.mean(tmp))