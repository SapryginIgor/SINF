from datetime import datetime

import numpy as np
import torch
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.flows import Flow
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from scipy.stats.qmc import Sobol

from AR import AR, ConditionalAR
from MA import MA, ConditionalMA
from myMAF import MyMaskedAutoregressiveFlow
from myNVP import MySimpleRealNVP
from mpmath import mp, exp,sqrt

writer = SummaryWriter()

def create_static_dist(target_config, n_samples):
    if target_config[0] is AR:
        p = target_config[1]
        while True:
            params = torch.cat([torch.tensor([1.0]), (torch.rand(p) - 0.5)])
            dist = AR([n_samples], params=params)
            if dist.is_stationary():
                break
    elif target_config[0] is MA:
        q = target_config[1]
        params = torch.cat([torch.tensor([1.0]), (torch.rand(q) - 0.5)])
        dist = MA([n_samples], params=params)
    else:
        dist = StandardNormal([n_samples])
    return dist


def compute_context_metrics(target_config, flow, n_samples):
    power = 14
    mp.dps = 50
    mp.pretty=True
    target = create_static_dist(target_config, n_samples)
    context = torch.broadcast_to(target.params, (2**power, len(target.params)))
    sob = Sobol(d=n_samples, scramble=True)
    samples = 20*torch.tensor(sob.random_base2(m=power), requires_grad=False, dtype=torch.float32) - 10
    exp_array = np.frompyfunc(exp, 1, 1)
    flow_prob = exp_array(flow.log_prob(inputs=samples, context=context).detach().numpy())
    target_prob = exp_array(target.log_prob(inputs=samples).detach().numpy())
    flow_l2 = sqrt((flow_prob**2).mean())
    target_l2 = sqrt((target_prob**2).mean())
    diff = sqrt(((flow_prob - target_prob)**2).mean())
    return  diff, flow_l2, target_l2







def conditional_train_flow(target_config: tuple, flow: Flow, n_samples: int, num_epochs: int, batch_size: int, include_context: bool, base_name, target_name, flow_name):
    optimizer = optim.Adam(flow.parameters())
    n_cases = 10

    for epoch in range(num_epochs):
        flow_loss = 0
        optimizer.zero_grad()
        n_finite = 0
        for k in range(n_cases):
            target_dist = create_static_dist(target_config, n_samples)
            train = target_dist.sample(num_samples=batch_size)
            if include_context:
                context = torch.broadcast_to(target_dist.params, (batch_size, len(target_dist.params)))
            else:
                context = None
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
            continue
        writer.add_scalar("ContextLoss/train_" + flow_name + '_' + base_name + '_' + target_name + '_' + str(n_samples) + '_' + 'samples' + '_' + datetime.today().strftime('%Y-%m-%d'), flow_loss, epoch)
        flow_loss.backward()
        if hasattr(flow._distribution, 'context_encoder'):
            max_norm = 1000
            torch.nn.utils.clip_grad_norm_(flow._distribution.context_encoder.parameters(), max_norm)
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"iteration: {epoch}, loss: {flow_loss.data}")
    # print('n_samples: {}, diff: {}, integral: {}, flow_l2: {}, ar_l2: {}'.format(n_samples, *compute_metrics(target_dist, flow, n_samples)))
    print('n_samples: {}, diff: {}, flow_l2: {}, target_l2: {}'.format(n_samples, *compute_context_metrics(target_config, flow, n_samples)))
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
        torch.manual_seed(100)
        np.random.seed(42)
        flow_type = ['MAF']
        base_distributions = [(ConditionalMA, 1)]
        target_distributions = [(AR, 2)]
        n_samples = [100]
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
                            base_dist = create_static_dist(bd, n_samples=ns)
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
                        conditional_train_flow(tc, flow, ns, 300, 100, include_context,base_name, target_name, ft)


