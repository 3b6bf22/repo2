import torch
import torch.nn as nn
import numpy as np

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