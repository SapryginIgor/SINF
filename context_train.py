from datetime import datetime

import numpy as np
import torch
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.flows import Flow
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from AR import AR, ConditionalAR
from MA import MA
from myMAF import MyMaskedAutoregressiveFlow
from myNVP import MySimpleRealNVP

writer = SummaryWriter()

def create_target_dist(target_config, n_samples):
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

    return dist



def conditional_train_flow(target_config: tuple, flow: Flow, n_samples: int, num_epochs: int, batch_size: int, base_name, target_name, flow_name):
    optimizer = optim.Adam(flow.parameters())
    n_cases = 10
    for epoch in range(num_epochs):
        flow_loss = 0
        optimizer.zero_grad()
        for k in range(n_cases):
            target_dist = create_target_dist(target_config, n_samples)
            train = target_dist.sample(num_samples=batch_size)
            flow_loss += -flow.log_prob(inputs=train, context=torch.broadcast_to(target_dist.params, (batch_size, len(target_dist.params)))).mean()
        flow_loss /= n_cases
        writer.add_scalar("ContextLoss/train_" + flow_name + '_' + base_name + '_' + target_name + '_' + str(n_samples) + '_' + 'samples' + '_' + datetime.today().strftime('%Y-%m-%d'), flow_loss, epoch)
        flow_loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"iteration: {epoch}, loss: {flow_loss.data}")
    # print('n_samples: {}, diff: {}, integral: {}, flow_l2: {}, ar_l2: {}'.format(n_samples, *compute_metrics(target_dist, flow, n_samples)))
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

def compute_metrics(target, flow, n_samples):
    # di = {100: 1, 30: 2, 10: 5, 6}
    N = 100000
    samples = np.random.uniform(-1, 1, (N, n_samples)).astype(np.float32)
    # slices = [np.linspace(-10,10,N) for _ in range(n_samples)]
    # values = np.meshgrid(*slices)
    # stacked = np.stack(values, axis=-1).reshape(-1,n_samples).astype(np.float32)
    flow_prob = flow.log_prob(inputs=samples).exp()
    ar_prob = target.log_prob(inputs=samples).exp()
    diff = torch.max(torch.abs(flow_prob - ar_prob))
    cell_area = (20/N)**n_samples
    integral = (flow_prob - ar_prob).square().sum()
    flow_l2 = torch.sqrt(flow_prob.square().sum())
    ar_l2 = torch.sqrt(ar_prob.square().sum())
    return diff, integral, flow_l2, ar_l2







if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    flow_type = ['MAF']
    base_distributions = [(ConditionalAR, 1), (ConditionalAR, 2)]
    target_distributions = [(AR, 1), (AR, 2), (MA, 1), (MA, 2)]
    n_samples = [2, 20, 100]
    for bd in base_distributions:
        for tc in target_distributions:
            for ft in flow_type:
                for ns in n_samples:
                    if bd[1] and ns <= bd[1]:
                        continue
                    elif tc[1] and ns <= tc[1]:
                        continue
                    if bd[0] is StandardNormal:
                        base_dist = StandardNormal([ns])
                    elif bd[0] is ConditionalAR:
                        base_dist = ConditionalAR(shape=[ns], context_encoder=nn.Linear(tc[1]+1, bd[1]+1))
                        # base_dist = ConditionalDiagonalNormal(shape=[ns], context_encoder=nn.Linear(tc[1]+1, ns*2))
                    elif bd[0] is MA:
                        base_dist = make_MA_q(q=bd[1], n=ns)
                    if ft == 'MAF':
                        flow = MyMaskedAutoregressiveFlow(features=ns, hidden_features=20, context_features=tc[1]+1, num_layers=3, num_blocks_per_layer=2, distribution=base_dist)
                    elif ft == 'RealNVP':
                        flow = MySimpleRealNVP(features=ns, hidden_features=20, num_layers=3, num_blocks_per_layer=3, distribution=base_dist)
                    if bd[0] is StandardNormal:
                        base_name = 'iid'
                    else:
                        base_name = bd[0].__name__+'_'+str(bd[1])
                    # if tc[0] is AR:
                    #     target_dist = make_AR_p(p=tc[1], n=ns)
                    # elif tc[0] is MA:
                    #     target_dist = make_MA_q(q=tc[1], n=ns)
                    target_name = tc[0].__name__ + '_' + str(tc[1])
                    conditional_train_flow(tc, flow, ns, 200, 100, base_name, target_name, ft)


