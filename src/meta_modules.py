from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)
import torch.nn as nn

class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']
        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape)-2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class ReLUFC_(MetaModule):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features):
        super().__init__()

        self.net = [BatchLinear(in_features, hidden_features), nn.ReLU(inplace=True)]

        for i in range(num_hidden_layers):
            self.net.append(BatchLinear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))

        self.net.append(BatchLinear(hidden_features, out_features))

        self.net = MetaSequential(*self.net)
        
    def forward(self, coords, params=None, **kwargs):
        coords = coords.squeeze(1).transpose(2,1)
        output = self.net(coords, params=self.get_subdict(params, 'net'))
        return output.transpose(2, 1)


class MetaSDF(nn.Module):
    def __init__(self, hypo_module, loss, init_lr=1e-1, num_meta_steps=3, first_order=False, lr_type='per_parameter', inner_reg = False, gamma = None):
        super().__init__()
        self.inner_reg = inner_reg
        self.gamma = gamma
        self.hypo_module = hypo_module
        self.loss = loss
        self.num_meta_steps = num_meta_steps
        
        self.first_order = first_order

        self.lr_type = lr_type
        if self.lr_type == 'static':
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter':
            self.lr = nn.ModuleList([])
            hypo_parameters = hypo_module.parameters()
            for param in hypo_parameters:
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_meta_steps)]))
        elif self.lr_type == 'simple_per_parameter':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr])) for _ in hypo_module.parameters()])
         
        sigma = torch.ones(2)
        self.sigma = nn.Parameter(sigma)
        
        self.sigma_outer = nn.Parameter(torch.ones(2))
        #self.lmbda = nn.Parameter(torch.ones(1)*10.0)

        num_outputs = list(self.hypo_module.parameters())[-1].shape[0]

    def generate_params(self, context_x, context_y, num_meta_steps=None, **kwargs):
        meta_batch_size = context_x.shape[0]
        num_meta_steps = num_meta_steps if num_meta_steps != None else self.num_meta_steps
        with torch.enable_grad():
            adapted_parameters = OrderedDict()
            for name, param in self.hypo_module.meta_named_parameters():
                adapted_parameters[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))
            for j in range(num_meta_steps):
                #torch.cuda.empty_cache()
                context_x.requires_grad_()
                
                predictions = self.hypo_module(context_x, params=adapted_parameters)
                loss = self.loss(predictions, context_y, sigma=self.sigma,  **kwargs) 
                #loss = losses[0]+ dense_loss if isinstance(losses, tuple) else losses + dense_loss
                grads = torch.autograd.grad(loss, adapted_parameters.values(), allow_unused=False, create_graph=(True if (not self.first_order or j == num_meta_steps-1) else False))
                for i, ((name, param), grad) in enumerate(zip(adapted_parameters.items(), grads)):                    
                    if self.lr_type in ['static', 'global']:
                        lr = self.lr
                    elif self.lr_type in ['per_step']:
                        lr = self.lr[j]
                    elif self.lr_type in ['per_parameter']:
                        lr = self.lr[i][j] if num_meta_steps <= self.num_meta_steps else 1e-2
                    elif self.lr_type in ['simple_per_parameter']:
                        lr = self.lr[i]
                    else:
                        raise NotImplementedError
                    
                    adapted_parameters[name] = param - lr * grad
                    # TODO: Add proximal regularization from iMAML
                    # Add meta-regularization                
        return adapted_parameters
    
    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def forward(self, meta_batch, **kwargs):
        context_x, context_y = meta_batch['context']
        query_x = meta_batch['query'][0]
        fast_params = self.generate_params(context_x, context_y, **kwargs)
        return self.forward_with_params(query_x, fast_params), fast_params
