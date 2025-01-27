import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.nn as nn
from RFLAF_model import RBFLayer
from gen_data import Mysin, Myfunc2, Myfunc3
from tqdm import tqdm
import os
import time

class testrbf(nn.Module):
    def __init__(self, x, h, N, L, R):
        super(testrbf, self).__init__()
        hlist=h * np.ones(N)
        clist=np.linspace(L, R, N)
        paralist = list(zip(clist, hlist))
        
        self.rbfs = nn.ModuleList([RBFLayer(torch.tensor(center), torch.tensor(gamma)) for center, gamma in paralist])
        self.A = torch.stack([rbf(x) for rbf in self.rbfs], dim=1) # [len(x), N]

    def forward(self, a):
        Aa = torch.matmul(self.A, a) # [len(x)]
        return Aa

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RFLAF')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--h', type=float, default=0.02)
    parser.add_argument('--N', type=int, default=401)
    parser.add_argument('--L', type=float, default=-2)
    parser.add_argument('--R', type=float, default=2)
    parser.add_argument('--M', type=int, default=1000)
    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--modelseed', type=int, default=402025)
    args=parser.parse_args()
    
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    actfunc = 'LAF'
    moreargs=f'_h={args.h}_N={args.N}_L={args.L}_R={args.R}_M={args.M}_lambda1={args.lambda1}_lambda2={args.lambda2}'
    moreargs=f'seed={args.modelseed}_epoch={args.epochs}{moreargs}'
    task_name = args.data + '_' + actfunc # taskname = 'dataset_modelname'
    
    h, N, L, R = args.h, args.N, args.L, args.R
    x_list = torch.tensor(np.linspace(L, R, 401), requires_grad=False).to(device)
    learned_func = testrbf(x_list, h, N, L, R).to(device)
    
    if args.data=='sin':
        mysinfunc = Mysin.apply
    elif args.data=='tru':
        mysinfunc = Myfunc2.apply
    elif args.data=='zoi':
        mysinfunc = Myfunc3.apply
    if args.data in ['sin', 'tru', 'zoi']:
        # mysinfunc = mysinfunc.to(device)
        y_list2 = mysinfunc(x_list).cpu().numpy() # Ground truth function
    
    cols, rows = 5, int((args.epochs+4)/5)
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    colors = ['#ff7f0e','#1f77b4']
    
    print('Start plotting the learned activation function...')
    for epoch in tqdm(range(args.epochs)):
        coef = torch.tensor(np.loadtxt(f"./coef/{task_name}_coef_{moreargs}_{epoch+1}.txt")).to(device)
        y_list = learned_func(coef).cpu().numpy()

        i = epoch % rows
        j = epoch // rows
        if rows==1:
            ax = axs[j]
        else:
            ax = axs[i, j]

        # Plot the learned activation function
        if args.data in ['sin', 'tru', 'zoi']:
            ax.plot(x_list.cpu().detach().numpy(), y_list2, label='Ground truth', ls='-', color=colors[0])
        ax.plot(x_list.cpu().detach().numpy(), y_list*(7.8), label='Learned curve', ls=':', color=colors[1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'No. {epoch+1}')

    if args.data=='sin':
        plt.title(r'$\sigma_1$')
    elif args.data=='tru':
        plt.title(r'$\sigma_2$')
    elif args.data=='zoi':
        plt.title(r'$\sigma_3$')
    plt.legend()
    plt.tight_layout()
    os.makedirs('./transform_figs', exist_ok=True)
    plt.savefig(f"./transform_figs/{task_name}_actfunc_{moreargs}.pdf", format='pdf')
    end_time = time.time()
    print(f'Elapsed time: {end_time-start_time:.2f} seconds')