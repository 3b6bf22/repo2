import torch
import torch.nn as nn
import numpy as np
import os
import torch.autograd as autograd
import argparse

# Functions
class Mysin(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        indicator = torch.where((-1 <= input) & (input <= 1), torch.tensor(1.0), torch.tensor(0.0))
        result = torch.sin(input*torch.tensor(torch.pi)) * indicator
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        indicator = torch.where((-1 <= input) & (input <= 1), torch.tensor(1.0), torch.tensor(0.0))
        grad_input = grad_output * torch.cos(input*torch.tensor(torch.pi)) * torch.tensor(torch.pi) * indicator
        return grad_input

class Myfunc2(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        indicator = torch.where((0 <= input) & (input <= 1), torch.tensor(1.0), torch.tensor(0.0))
        result = torch.sin(input*torch.tensor(torch.pi)) * indicator
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        indicator = torch.where((0 <= input) & (input <= 1), torch.tensor(1.0), torch.tensor(0.0))
        grad_input = grad_output * torch.cos(input*torch.tensor(torch.pi)) * torch.tensor(torch.pi) * indicator
        return grad_input
    
class Myfunc3(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        indicator0 = torch.where((-1.5 <= input) & (input <= -0.5), torch.tensor(1.0), torch.tensor(0.0))
        indicator1 = torch.where((0.5 <= input) & (input <= 1.5), torch.tensor(1.0), torch.tensor(0.0))
        f0 = torch.tensor(-1.0) * torch.sin((input+torch.tensor(0.5))*torch.tensor(torch.pi))
        f1 = torch.tensor(1.0) * torch.sin((input-torch.tensor(0.5))*torch.tensor(torch.pi))
        result = f0 * indicator0 + f1 * indicator1
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        indicator0 = torch.where((-1.5 <= input) & (input <= -0.5), torch.tensor(1.0), torch.tensor(0.0))
        indicator1 = torch.where((0.5 <= input) & (input <= 1.5), torch.tensor(1.0), torch.tensor(0.0))
        f0 = torch.tensor(-torch.pi) * torch.cos((input+torch.tensor(0.5))*torch.tensor(torch.pi))
        f1 = torch.tensor(torch.pi) * torch.cos((input-torch.tensor(0.5))*torch.tensor(torch.pi))
        result = f0 * indicator0 + f1 * indicator1
        grad_input = grad_output * result
        return grad_input

class target_func(nn.Module):
    def __init__(self, M, input_dim, actfunc):
        super(target_func, self).__init__()
        self.M = torch.tensor(M*1.0)
        self.W = nn.Parameter(torch.randn(input_dim, M)) # [input_dim, M]
        self.W.requires_grad = False 

        self.b1 = torch.randn(input_dim)
        self.b2 = torch.randn(input_dim)
        
        if actfunc=='sin':
            self.func = Mysin.apply
            amplitude = torch.tensor(20.0)
        elif actfunc=='tru':
            self.func = Myfunc2.apply
            amplitude = torch.tensor(7.0)
        elif actfunc=='zoi':
            self.func = Myfunc3.apply
            amplitude = torch.tensor(5.0)

        self.v = nn.Parameter(torch.randn(M))
        self.v.requires_grad = False
        self.v.data = torch.max(torch.matmul(self.b1, self.W),torch.matmul(self.b2, self.W))*amplitude

    def forward(self, x):
        # x: [batch_size, input_dim]
        xW = torch.matmul(x, self.W)  # [batch_size, M]
        act_outputs = torch.matmul(self.func(xW),self.v)
        result = torch.div(act_outputs, self.M)
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='sin') # ‘zoi' 'tru'
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--M', type=int, default=100000)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--sample_size', type=int, default=15000)
    parser.add_argument('--bs', type=int, default=200)
    
    args = parser.parse_args()
    
    task_name = args.task_name
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # task_name = 'zoi' # 'sin' 'tru'
    # M = 100000
    # input_dim = 2
    # sample_size = 15000
    M = args.M
    input_dim = args.input_dim
    sample_size = args.sample_size

    X = torch.randn(sample_size, input_dim) 
    groundtruth = target_func(M, input_dim, task_name)

    bs = args.bs
    nmbs = sample_size//bs
    ylist=[]

    for i in range(nmbs):
        xbch = X[i*bs:(i+1)*bs]
        ybch = groundtruth(xbch)
        ylist.append(ybch)

    y = torch.cat(ylist, dim=0)

    os.makedirs('./data', exist_ok=True)
    np.save(f'./data/X_{task_name}.npy', X.numpy())
    np.save(f'./data/y_{task_name}.npy', y.detach().numpy())