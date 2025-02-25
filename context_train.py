from datetime import datetime
import numpy as np
import pandas as pd
import torch
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.flows import Flow
from nflows.distributions import Distribution
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from scipy.stats.qmc import Sobol
from scipy import integrate
from AR import AR, ConditionalAR
from MA import MA, ConditionalMA
from myMAF import MyMaskedAutoregressiveFlow
from myNVP import MySimpleRealNVP
from mpmath import mp, exp,sqrt, mpf
from numba import jit
import time
from cubature import cubature

writer = SummaryWriter()

def create_static_dist(target_config, n_samples):
    if target_config[0] is AR:
        p = target_config[1]
        # while True:
        #     params = torch.cat([torch.tensor([1.0]), (torch.rand(p) - 0.5)])
        #     dist = AR([n_samples], params=params)
        #     if dist.is_stationary():
        #         break
        params = torch.tensor([2**power for power in range(0,-p,-1)])
        params = torch.cat([torch.tensor([1.0]), params])
        dist = AR([n_samples], params=params)
    elif target_config[0] is MA:
        q = target_config[1]
        params = torch.tensor([2 ** power for power in range(0, -q, -1)])
        params = torch.cat([torch.tensor([1.0]), params])
        # params = torch.cat([torch.tensor([1.0]), (torch.rand(q) - 0.5)])
        dist = MA([n_samples], params=params)
    else:
        dist = StandardNormal([n_samples])
    return dist 


def compute_context_metrics(target, flow, n_samples, include_context):
    # power = 20
    # mp.dps = 50
    # mp.pretty=True
    # target = create_static_dist(target_config, n_samples)
    # if include_context:
    #     context = torch.broadcast_to(target.params, (2**power, len(target.params)))
    # else:
    #     context = None
    # sob = Sobol(d=n_samples, scramble=True)
    # scale = 10
    # samples = scale*torch.tensor(sob.random_base2(m=power), requires_grad=False, dtype=torch.float32) - (scale//2)
    # exp_array = np.frompyfunc(exp, 1, 1)
    # jac = mpf(scale**n_samples)
    # flow_prob = exp_array(flow.log_prob(inputs=samples, context=context).detach().numpy())
    # target_prob = exp_array(target.log_prob(inputs=samples).detach().numpy())
    # # flow_l2 = np.sqrt(jac*(flow_prob**2).mean())
    # # target_l2 = np.sqrt(jac*(target_prob**2).mean())
    # diff = (np.abs(jac * (flow_prob - target_prob)).mean())
    # return  diff
    dims = n_samples
    sampler = Sobol(d=dims, scramble=True)
    mp.dps = 50
    mp.pretty = True
    m = 18
    t = sampler.random_base2(m=18)
    u = np.pi * (t - 0.5)
    x = np.tan(u)
    mpcos = np.vectorize(mp.cos)
    jac = 1 / np.prod(np.cos(u) ** 2, axis=-1)

    start = time.time()
    mpexp = np.vectorize(mp.exp)
    flow.double()

    if include_context:
        context = torch.broadcast_to(target.params, (2**m, len(target.params))).to(torch.float64)
    else:
        context = None

    flow_l1 = np.mean(jac*np.exp(flow.log_prob(inputs=torch.tensor(x, requires_grad=False, dtype=torch.float64),context=context).detach().numpy()) * mp.pi ** dims)
    target_l1 = np.mean(jac*np.exp(target.log_prob(torch.tensor(x, requires_grad=False, dtype=torch.float64)).detach().numpy()) * mp.pi ** dims)
    diff_l1 = np.mean(jac*np.abs(np.exp(target.log_prob(torch.tensor(x, requires_grad=False, dtype=torch.float64)).detach().numpy()) -
                          np.exp(flow.log_prob(inputs=torch.tensor(x, requires_grad=False, dtype=torch.float64),context=context).detach().numpy())) *   mp.pi ** dims)
    print("time is: {}".format(time.time() - start))
    return round(flow_l1,3), round(target_l1,3), round(diff_l1,3)


def compute_precise_context_metrics(target_config, flow, n_samples):
    power = 20
    mp.dps = 50
    mp.pretty=True
    target = create_static_dist(target_config, n_samples)
    # context = torch.broadcast_to(target.params, (2**power, len(target.params)))
    context = target.params[None]
    def flow_func(*args):
        x = args[0][None]
        flow_prob = flow.log_prob(inputs=x, context=context).exp().detach().numpy()
        return flow_prob

    def target_func(*args):
        x = args[0][None]
        target_prob = target.log_prob(inputs=x, context=context).exp().detach().numpy()
        return target_prob

    def flow_squared(*args):
        x = torch.tensor(args, requires_grad=False)
        return flow_func(x)**2

    def target_squared(*args):
        x = torch.tensor(args, requires_grad=False)
        return target_func(x)**2

    def diff_squared(*args):
        x = torch.tensor(args, requires_grad=False)
        return (flow_func(x) - target_func(x))**2

    flow_l2,_,_= integrate.nquad(flow_squared, [[-10, 10] for _ in range(n_samples)], full_output=True)
    target_l2,_,_ = integrate.nquad(target_squared, [[-10, 10] for _ in range(n_samples)], full_output=True)
    diff_l2,_,_= integrate.nquad(diff_squared, [[-10, 10] for _ in range(n_samples)], full_output=True)
    return  diff_l2, flow_l2, target_l2, diff_l2/target_l2



def compute_integral(flow, context):
    def density_func(x):
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=False)[None]
        return torch.exp(flow.log_prob(inputs=x_tensor,context=context)).detach().numpy()
    dims = 3
    integral, _ = cubature(density_func, ndim=dims, fdim=1, xmin=[-10 for _ in range(dims)], xmax=[10 for _ in range(dims)])

def importance_sampling(flow, dims, base_dist, con, num_samples=100000):
    """
    Estimate integral of the flow density over R^d using importance sampling.

    Args:
        flow: A trained normalizing flow model.
        num_samples: Number of Monte Carlo samples.

    Returns:
        Estimated integral value.
    """
     # Get the base distribution
      # Sample from base Gaussian

    # Compute importance weights: p(x) / q(x)
    if con is not None:
        context = torch.broadcast_to(con, (num_samples, dims))
    else:
        context = None
    samples = base_dist.sample(num_samples, context=context)
    log_prob_p = flow.log_prob(inputs = samples, context=context)  # Log probability under the flow model
    log_prob_q = base_dist.log_prob(samples, context=context)  # Log probability under base distribution

    weights = torch.exp(log_prob_p - log_prob_q)  # Importance weights

    return weights.mean().item()  # Monte Carlo estimate

# def compute_precise_context_metrics(target_config, flow, n_samples):
#
#     target = create_static_dist(target_config, n_samples)
#     context = target.params[None]
#     dims = len(target.params)
#     flow_l1 = importance_sampling(flow, dims,flow._distribution ,context)
#     target_l1 = importance_sampling(target, dims,StandardNormal([n_samples]),None )
#     return flow_l1, target_l1

metrics_data = []

def conditional_train_flow(target_config, target_dist: Distribution, flow: Flow, n_samples: int, num_epochs: int, batch_size: int, include_context: bool, base_name, target_name, flow_name, need_prefix=True):
    optimizer = optim.Adam(flow.parameters())
    final_target_dist = target_dist
    n_cases = 10
    best_diff = 10000
    best_metrics = []
    flow_loss = -1
    flow.double()
    for epoch in range(num_epochs):
        flow_loss = 0
        optimizer.zero_grad()
        n_finite = 0
        for k in range(n_cases):
            if include_context:
                target_dist = create_static_dist(target_config, n_samples)
                context = torch.broadcast_to(target_dist.params, (batch_size, len(target_dist.params))).to(
                    torch.float64)
            else:
                context = None
            train = target_dist.sample(num_samples=batch_size).to(torch.float64)
            prob = -flow.log_prob(inputs=train, context=context).mean()
            if prob.abs().item() != torch.inf and prob.item() != torch.nan:
                flow_loss += prob
                n_finite+=1
        if n_finite == 0:
            continue
        flow_loss /= n_finite
        if flow_loss.isnan().item() and hasattr(flow._distribution, '_context_encoder'):
            torch.nn.init.xavier_uniform_(flow._distribution._context_encoder.weight.data)
            print("GOT NAN")
            # raise Exception("SHIT!")
            continue
        writer.add_scalar("ContextLoss/train_" + base_name + '_' + target_name + '_' + flow_name + '_' + str(n_samples) + '_' + 'samples' + '_' + datetime.today().strftime('%Y-%m-%d'), flow_loss, epoch)
        flow_loss.backward()
        if hasattr(flow._distribution, 'context_encoder'):
            max_norm = 1000
            torch.nn.utils.clip_grad_norm_(flow._distribution.context_encoder.parameters(), max_norm)
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"iteration: {epoch}, loss: {flow_loss.data}")

    metrics = compute_context_metrics(final_target_dist, flow, n_samples, include_context)
    print('base: {}, target: {}, flow: {}, dim: {}, flow_l1: {}, target_l1: {}, diff_l1: {}'.format(base_name, target_name, flow_name, n_samples,  *metrics))
    if need_prefix:
        metrics_data.append([base_name, target_name, flow_name, n_samples,  *metrics])
    else:
        metrics_data.append([*metrics])
    writer.flush()

def make_AR_p(p, n):
    params = torch.cat([torch.tensor([1.0]), (torch.rand(p) - 0.5) / 2])
    params = torch.sign(params) * torch.maximum(torch.full_like(params, 0.1),
                                                              torch.abs(params))
    dist = AR([n], params=params)
    assert (dist.is_stationary())
    return dist

def make_MA_q(q, n):
    params = torch.cat([torch.tensor([1.0]), (torch.rand(q) - 0.5) / 2])
    params = torch.sign(params) * torch.maximum(torch.full_like(params, 0.1),
                                                torch.abs(params))
    dist = MA([n], params=params)
    return dist







if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(False):
        torch.manual_seed(81)
        np.random.seed(42)
        flow_type = ['MAF']
        base_distributions = [(ConditionalAR,2), (ConditionalMA,2), (AR,2), (MA,2),(StandardNormal, None)]
        target_distributions = [(AR, 2),(MA, 2)]
        metrics_frame = None
        fixed_base_distributions = {}
        fixed_target_distributions = {}
        n_samples = [10, 15]
        for seed in range(5):
            torch.manual_seed(seed)
            metrics_data.clear()
            for bd in base_distributions:
                for tc in target_distributions:
                    for ft in flow_type:
                        for ns in n_samples:
                            if bd[1] and ns <= bd[1]:
                                continue
                            elif tc[1] and ns <= tc[1]:
                                continue
                            if bd[0] is ConditionalAR or bd[0] is ConditionalMA:
                                base_dist = bd[0](shape=[ns], context_encoder=nn.Linear(tc[1]+1, bd[1]+1))
                                context_features = tc[1]+1
                                include_context=True
                                # base_dist = ConditionalDiagonalNormal(shape=[ns], context_encoder=nn.Linear(tc[1]+1, ns*2))
                            else:
                                fixed_base_distributions[(bd,ns)] = fixed_base_distributions.get(bd, create_static_dist(bd, n_samples=ns))
                                base_dist = fixed_base_distributions[(bd,ns)]
                                include_context=False
                                context_features = None
                            if ft == 'MAF':
                                flow = MyMaskedAutoregressiveFlow(features=ns, hidden_features=20, context_features=context_features, num_layers=3, num_blocks_per_layer=2, distribution=base_dist)
                            elif ft == 'RealNVP':
                                flow = MySimpleRealNVP(features=ns, hidden_features=20, context_features=context_features, num_layers=3, num_blocks_per_layer=3, distribution=base_dist)
                            if bd[0] is StandardNormal:
                                base_name = 'iid'
                            else:
                                base_name = bd[0].__name__+'_'+str(bd[1])
                            # if tc[0] is AR:
                            #     target_dist = make_AR_p(p=tc[1], n=ns)
                            # elif tc[0] is MA:
                            #     target_dist = make_MA_q(q=tc[1], n=ns)
                            target_name = tc[0].__name__ + '_' + str(tc[1])
                            fixed_target_distributions[(tc,ns)] = fixed_target_distributions.get(tc, create_static_dist(tc, n_samples=ns))
                            target_dist = fixed_target_distributions[(tc,ns)]
                            # metrics = compute_context_metrics(tc, flow, ns, include_context)
                            # print('n_samples: {}, epoch: {}, diff_l1: {}, flow_l1: {}, target_l1: {}, diff/target: {}'.format(
                            #         ns, 0,
                            #         *metrics))
                            need_prefix = True if seed==0 else False
                            conditional_train_flow(tc, target_dist, flow, ns, 300, 100, include_context,base_name, target_name, ft, need_prefix)
            if not metrics_frame:
                metrics_frame = pd.DataFrame(metrics_data, columns=['base', 'target', 'flow', 'dim', 'flow_l1_0', 'target_l1_0', 'diff_l1_0'])
            else:
                names = ['flow_l1', 'target_l1', 'diff_l1']
                names = [s + '_' + str(seed) for s in names]
                tmp = pd.DataFrame(metrics_data, columns=names)
                metrics_frame = pd.concat([metrics_frame, tmp], axis=1)


metrics_frame.to_csv("many_seeds_conditional_metrics.csv", index=False)
print(metrics_frame)



