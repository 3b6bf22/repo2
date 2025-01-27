import torch
import torch.nn as nn
import torch.autograd as autograd
from RFLAF_model import RBFLayer

class Cos(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.cos(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (-torch.sin(input))
    
class RFMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, actfunc='relu', seed=0, frozen=True):
        super(RFMLP, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.M = torch.tensor(hidden_dim*1.0, requires_grad=False)
        if frozen:
            self.weight = nn.Parameter(torch.randn(input_dim, hidden_dim), requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.randn(input_dim, hidden_dim))
        if output_dim==1:
            self.v = nn.Parameter(torch.randn(hidden_dim)/torch.sqrt(self.M))
        else:
            self.v = nn.Parameter(torch.randn(hidden_dim, output_dim)/torch.sqrt(self.M))
        if actfunc=='relu':
            self.actfunc = nn.ReLU()
        elif actfunc=='tanh':
            self.actfunc = nn.Tanh()
        elif actfunc=='cos':
            self.actfunc = Cos.apply
        elif actfunc=='silu':
            self.actfunc = nn.SiLU()
        elif actfunc=='gelu':
            self.actfunc = nn.GELU()
        elif actfunc=='softplus':
            self.actfunc = nn.Softplus()
        elif actfunc=='sigmoid':
            self.actfunc = nn.Sigmoid()
        elif actfunc=='elu':
            self.actfunc = nn.ELU()
        elif actfunc=='RBF1':
            self.actfunc = RBFLayer(0.0, 0.5)
        elif actfunc=='RBF2':
            self.actfunc = RBFLayer(1.5, 0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.actfunc(torch.matmul(x, self.weight))
        x = torch.matmul(x, self.v)
        return x