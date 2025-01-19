import numpy as np
import torch
from torch import optim
from nflows.distributions import Distribution
from torch.utils.tensorboard import SummaryWriter

from nflows.flows import Flow
from myMAF import MyMaskedAutoregressiveFlow
from datetime import datetime
from nflows.distributions.normal import StandardNormal
from AR import AR

writer = SummaryWriter()

def train_flow(target_dist: Distribution, flow: Flow, n_samples: int, num_epochs: int, batch_size: int, base_name, target_name):
    optimizer = optim.Adam(flow.parameters())
    for epoch in range(num_epochs):
        train = target_dist.sample(num_samples=n_samples, batch_size=batch_size)
        optimizer.zero_grad()
        flow_loss = -flow.log_prob(inputs=train).mean()
        writer.add_scalar("Loss/train_" + base_name + '_' + target_name + '_' + str(n_samples) + '_' + 'samples' + '_' + datetime.today().strftime('%Y-%m-%d'), flow_loss, epoch)
        flow_loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"iteration: {epoch}, loss: {flow_loss.data}")
    print('n_samples: {}, diff: {}, integral: {}, flow_l2: {}, ar_l2: {}'.format(n_samples, *compute_metrics(target_dist, flow, n_samples)))
    writer.flush()

def make_AR_p(p, n):
    params = torch.cat([torch.tensor([1.0]), (torch.rand(p) - 0.5) / 2])
    params = torch.sign(params) * torch.maximum(torch.full_like(params, 0.1),
                                                              torch.abs(params))
    dist = AR([n], params=params)
    assert (dist.is_stationary())
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
    base_distributions = [(StandardNormal, None), (AR, 1), (AR, 2)]
    target_distributions = [(AR, 1), (AR, 2)]
    n_samples = [2, 5, 10, 20, 100]
    for bd in base_distributions:
        for td in target_distributions:
            for ft in flow_type:
                for ns in n_samples:
                    if bd[1] and ns <= bd[1]:
                        continue
                    elif td[1] and ns <= td[1]:
                        continue
                    if bd[0] is StandardNormal:
                        base_dist = StandardNormal([ns])
                    elif bd[0] is AR:
                        base_dist = make_AR_p(p=bd[1], n=ns)
                    if ft == 'MAF':
                        flow = MyMaskedAutoregressiveFlow(features=ns, hidden_features=100, num_layers=3, num_blocks_per_layer=2, distribution=base_dist)
                    if bd[0] is StandardNormal:
                        base_name = 'iid'
                    else:
                        base_name = bd[0].__name__+'_'+str(bd[1])
                    if td[0] is AR:
                        target_dist = make_AR_p(p=td[1], n=ns)
                    target_name = td[0].__name__ + '_' + str(td[1])

                    train_flow(target_dist, flow, ns, 100, 100, base_name, target_name)


