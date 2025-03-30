import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd

class RBFLayer(nn.Module):
    def __init__(self, center, gamma):
        super(RBFLayer, self).__init__()
        self.c = center
        self.h = gamma

    def forward(self, input):
        result = -(input-self.c)**2/(2*self.h**2)
        return torch.exp(result)

class RFLAF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, h=0.02, N=401, L=-2, R=2, seed=0):
        super(RFLAF, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        hlist=h * np.ones(N)
        clist=np.linspace(L, R, N)
        paralist = list(zip(clist, hlist))
        
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim)) 
        self.W.requires_grad = False
        if output_dim==1:
            self.v = nn.Parameter(torch.randn(hidden_dim)/torch.sqrt(torch.tensor(hidden_dim*1.0)))
        else:
            self.v = nn.Parameter(torch.randn(hidden_dim, output_dim)/torch.sqrt(torch.tensor(hidden_dim*1.0)))  
        self.a = nn.Parameter(torch.randn(N)/torch.sqrt(torch.tensor(N*1.0)))
        self.rbfs = nn.ModuleList([RBFLayer(torch.tensor(center), torch.tensor(gamma)) for center, gamma in paralist])

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = x.view(x.size(0), -1)  # Flatten input
        xW = torch.matmul(x, self.W)  # [batch_size, M]
        A = torch.stack([rbf(xW) for rbf in self.rbfs], dim=2) # [batch_size, M, N]
        Aa = torch.matmul(A, self.a) # [batch_size, M]
        vAa = torch.matmul(Aa, self.v) # [batch_size]
        return vAa

class CustomRegularizer(nn.Module):
    def __init__(self, lambda1, lambda2, N, M):
        super(CustomRegularizer, self).__init__()
        self.lambda1 = torch.tensor(lambda1)
        self.lambda2 = torch.tensor(lambda2)
        self.N = torch.tensor(N*1.0)
        self.M = torch.tensor(M*1.0)

    def forward(self, a, v):
        L1_term = self.lambda2 * torch.div(torch.abs(a).sum(), self.N)
        reg_term = L1_term + self.lambda1 * (torch.div(torch.sum(a**2),torch.sqrt(self.N)) - torch.div(torch.sum(v**2),torch.sqrt(self.M)))**2
        return reg_term

# For B-Splines as base functions:

class HardIndicatorFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b):
        return ((x >= a) & (x < b)).type_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_x = torch.zeros_like(grad_output)
        return grad_x, None, None

class IndicatorFunction(nn.Module):
    def __init__(self, a, b):
        super(IndicatorFunction, self).__init__()
        self.a = a
        self.b = b
        
    def forward(self, x):
        return HardIndicatorFunction.apply(x, self.a, self.b)
        
class BSplineLayer(nn.Module):
    def __init__(self, c, h):
        super(BSplineLayer, self).__init__()
        self.c = c
        self.h = h
        self.c1, self.c2, self.c3, self.c4 = c-h, c, c+h, c+2*h
        self.I1 = IndicatorFunction(c-h, c)
        self.I2 = IndicatorFunction(c, c+h)
        self.I3 = IndicatorFunction(c+h, c+2*h)
        
    def forward(self, x):
        result = (x - self.c1)**2 * self.I1(x) \
            + (1.5 * self.h**2 - 2 * (x - ((self.c2+self.c3)/2.0) * self.h)**2) * self.I2(x) \
            + (x - self.c4)**2 * self.I3(x)
        return 2 * result / (3*self.h**2)
    
class RFLAF_BSpline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, N=401, L=-2, R=2, seed=0):
        super(RFLAF_BSpline, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        hlist=(R-L)/(1.0*(N-1)) * np.ones(N)
        clist=np.linspace(L, R, N)
        paralist = list(zip(clist, hlist))
        
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim)) 
        self.W.requires_grad = False
        if output_dim==1:
            self.v = nn.Parameter(torch.randn(hidden_dim)/torch.sqrt(torch.tensor(hidden_dim*1.0)))
        else:
            self.v = nn.Parameter(torch.randn(hidden_dim, output_dim)/torch.sqrt(torch.tensor(hidden_dim*1.0)))  
        self.a = nn.Parameter(torch.randn(N)/torch.sqrt(torch.tensor(N*1.0)))
        self.rbfs = nn.ModuleList([BSplineLayer(torch.tensor(center), torch.tensor(gamma)) for center, gamma in paralist])

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = x.view(x.size(0), -1)  # Flatten input
        xW = torch.matmul(x, self.W)  # [batch_size, M]
        A = torch.stack([rbf(xW) for rbf in self.rbfs], dim=2) # [batch_size, M, N]
        Aa = torch.matmul(A, self.a) # [batch_size, M]
        vAa = torch.matmul(Aa, self.v) # [batch_size]
        return vAa
    
# For polynomials as base functions:

class PolyLayer(nn.Module):
    def __init__(self, n_order):
        super(PolyLayer, self).__init__()
        self.n_order = n_order
        
    def forward(self, x):
        return x**self.n_order

class RFLAF_Poly(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, N=401, seed=0):
        super(RFLAF_Poly, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim)) 
        self.W.requires_grad = False
        if output_dim==1:
            self.v = nn.Parameter(torch.randn(hidden_dim)/torch.sqrt(torch.tensor(hidden_dim*1.0)))
        else:
            self.v = nn.Parameter(torch.randn(hidden_dim, output_dim)/torch.sqrt(torch.tensor(hidden_dim*1.0)))  
        self.a = nn.Parameter(torch.randn(N)/torch.sqrt(torch.tensor(N*1.0)))
        self.rbfs = nn.ModuleList([PolyLayer(n_order) for n_order in range(N)])

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = x.view(x.size(0), -1)  # Flatten input
        xW = torch.matmul(x, self.W)  # [batch_size, M]
        A = torch.stack([rbf(xW) for rbf in self.rbfs], dim=2) # [batch_size, M, N]
        Aa = torch.matmul(A, self.a) # [batch_size, M]
        vAa = torch.matmul(Aa, self.v) # [batch_size]
        return vAa