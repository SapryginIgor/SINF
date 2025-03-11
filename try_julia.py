from juliacall import Main as jl

import numpy as np
import torch
import time
from nflows.distributions.normal import StandardNormal

from AR import AR
from MA import MA
from context_train import make_AR_p, make_MA_q
from myMAF import MyMaskedAutoregressiveFlow
from myNVP import MySimpleRealNVP

text = """
    using HCubature
    dims = 8
    l = -100
    r = 100
    f(x) = exp(-sum(x.^2) / 2) / (sqrt(2 * pi)^dims)
    res = hcubature(f, [l for _ in 1:dims], [r for _ in 1:dims], atol = 0, maxevals=100000000, initdiv=1)
    print(res)
"""

l = -100
r = 100


jl.println("Hello from Julia!")
jl.seval('import Pkg; Pkg.add("HCubature")')

torch.manual_seed(42)
np.random.seed(42)
flow_type = ['MAF']
base_distributions = [(AR, 1), (AR, 2)]
target_distributions = [(AR, 1)]
n_samples = [8]
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
                elif bd[0] is MA:
                    base_dist = make_MA_q(q=bd[1], n=ns)
                if ft == 'MAF':
                    flow = MyMaskedAutoregressiveFlow(features=ns, hidden_features=20, num_layers=3,
                                                      num_blocks_per_layer=2, distribution=base_dist)
                elif ft == 'RealNVP':
                    flow = MySimpleRealNVP(features=ns, hidden_features=20, num_layers=3, num_blocks_per_layer=3,
                                           distribution=base_dist)


                def wrap_flow(x):
                    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=False).reshape(1, -1)
                    return np.float64(torch.exp(flow.log_prob(x_tensor)).item())

                jl.wflow = wrap_flow
                jl.dims = ns
                jl.l = l
                jl.r = r
                # print(type(wrap_flow([1.0, 2.0, 3.0])))

                command = """
                    using HCubature
                    using PythonCall
                    
                    function wrapper_flow(x)
                        val = wflow(x)
                        val_converted = pyconvert(Float64, val)
                        return val_converted
                    end
                    
                    res = hcubature(wrapper_flow, [l for _ in 1:dims], [r for _ in 1:dims], atol = 0, maxevals=100000, initdiv=1)
                    print(res)
                """
                start = time.time()
                jl.seval(command)
                print(time.time() - start)



# jl.seval(text)