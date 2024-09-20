import numpy as np
import matplotlib.pyplot as plt


def count_var(var, lam):
    return (var / (1 - lam ** 2)) ** 0.5


lambdas = [0.5, -0.1, 2, -3, 1, -1]

n_steps = 100
sigma = 10  # sigma for
seed = 42

fig, axes = plt.subplots(len(lambdas), figsize=(12, 10))
np.random.seed(seed)
rng = np.random.default_rng(seed)

for ax in axes.flat:
    ax.set(xlabel='Time', ylabel='X_T')
for (i, lam) in enumerate(lambdas):
    X = np.zeros(n_steps)
    eps = np.random.normal(0, sigma ** 2, n_steps)
    if abs(lam) < 1:
        mu_0 = 0
        sigma_0 = count_var(sigma ** 2, lam)
    else:
        if abs(lam) > 1:
            axes[i].set_yscale('log')
        mu_0 = rng.uniform(-100, 100)
        sigma_0 = rng.uniform(0, 10)
    X[0] = np.random.normal(mu_0, sigma_0)
    for step in range(1, n_steps):
        X[step] = lam * X[step - 1] + eps[step - 1]
    axes[i].title.set_text('\u039b' + " = " + str(lam) +
                           ", \u03bc = " + str(mu_0) +
                           ", \u03c3^2 = " + str(sigma_0))
    axes[i].plot(np.arange(n_steps), X)

fig.tight_layout()
plt.show()
