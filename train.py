import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from RFLAF_model import RFLAF, CustomRegularizer, RFLAF_BSpline, RFLAF_Poly
from BaseRF_model import RFMLP
from efficient_kan import KAN
import os
from tqdm import tqdm
import time
from load_more_data import get_uci_loaders

def load_data(data, batch_size):
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if data in ['sin', 'tru', 'zoi']:
        task_name = data
        X = np.load(f'./data/X_{task_name}.npy') 
        y = np.load(f'./data/y_{task_name}.npy') 
        X = torch.tensor(X)
        y = torch.tensor(y)

        dataset = TensorDataset(X, y)

        test_size = int(0.2 * len(dataset))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    elif data=='mnist':
        # 1,28,28
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (45.0,))])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    elif data=='cifar10':
        # 3,32,32
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (65.0, 65.0, 65.0))])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    else:
        # data = 'adult', 'protein', 'ct', 'workloads'
        X, y = get_uci_loaders(data)
        if data=='adult':
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        else:
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        test_size = int(0.2 * len(dataset))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset

def train():
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    train_time, test_time = [], []
    for epoch in range(epochs):
        time1 = time.time()
        iter=0
        total_loss, correct = 0, 0
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False):
            iter+=1
            images, labels = images.view(-1, input_dim).to(device), labels.to(device)
            
            outputs = model(images)
            if args.model in ['RFLAF', 'RFLAFBS', 'RFLAFPL', 'LAN', 'LANBS', 'LANPL']:
                reg_value = regularizer(model.a,model.v)
            else:
                reg_value = torch.tensor(0.0, requires_grad=False)
            loss = criterion(outputs, labels) + reg_value
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iter <= warmup_iter:
                warmup_scheduler.step()
            else:
                optimizer.step()
            total_loss += loss.item() - reg_value.item()
            if classification:
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
            # print(f'Epoch [{epoch+1}/{epochs}], Iter [{iter}], Loss: {loss.item():.6f}\t={loss.item():.6f}')
        
        train_losses.append(total_loss / len(train_loader))
        if classification:
            train_accuracies.append(correct / len(train_dataset))
        time2 = time.time()
        train_time.append(time2-time1)

        time1 = time.time()
        total_loss, correct = 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", leave=False):
                images, labels = images.view(-1, input_dim).to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                if classification:
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
        
        test_losses.append(total_loss / len(test_loader))
        if classification:
            test_accuracies.append(correct / len(test_dataset))
        time2 = time.time()
        test_time.append(time2-time1)
        
        if args.model in ['RFLAF', 'LAN', 'RFLAFBS', 'RFLAFPL', 'LANBS', 'LANPL']:
            os.makedirs('./coef', exist_ok=True)
            coef = model.a.cpu().detach().numpy()
            np.savetxt(f"./coef/{task_name}_coef_{moreargs}_{epoch+1}.txt", coef)
        
        if classification:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}| "
                f"Train Accuracy: {train_accuracies[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
    
    return train_losses, test_losses, train_accuracies, test_accuracies, train_time, test_time

def save_losses():
    if classification:
        os.makedirs('./accuracies', exist_ok=True)
        np.savetxt(f"./accuracies/{task_name}_train_acc_{moreargs}.txt", train_accuracies)
        np.savetxt(f"./accuracies/{task_name}_test_acc_{moreargs}.txt", test_accuracies)
    
    os.makedirs('./losses', exist_ok=True)
    np.savetxt(f"./losses/{task_name}_train_loss_{moreargs}.txt", train_losses)
    np.savetxt(f"./losses/{task_name}_test_loss_{moreargs}.txt", test_losses)
    
    os.makedirs('./times', exist_ok=True)
    np.savetxt(f"./times/{task_name}_train_time_{moreargs}.txt", train_time)
    np.savetxt(f"./times/{task_name}_test_time_{moreargs}.txt", test_time)

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
    parser.add_argument('--actfunc', type=str, default='relu')
    parser.add_argument('--modelseed', type=int, default=402025)
    parser.add_argument('--batch_size', type=int, default=128)
    args=parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device==torch.device("cuda"):
        print(f"GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}")
    
    classification = False
    if args.data in ['sin', 'tru', 'zoi']:
        input_dim = 2
        hidden_dim = args.M
        output_dim = 1
        criterion = nn.MSELoss()
    elif args.data=='mnist':
        input_dim = 28 * 28
        hidden_dim = args.M
        output_dim = 10
        criterion = nn.CrossEntropyLoss()
        classification = True
    elif args.data=='cifar10':
        input_dim = 3 * 32 * 32
        hidden_dim = args.M
        output_dim = 10
        criterion = nn.CrossEntropyLoss()
        classification = True
    elif args.data=='adult':
        input_dim = 100
        hidden_dim = args.M
        output_dim = 2
        criterion = nn.CrossEntropyLoss()
        classification = True
    elif args.data=='protein':
        input_dim = 9
        hidden_dim = args.M
        output_dim = 1
        criterion = nn.MSELoss()
    elif args.data=='ct':
        input_dim = 384
        hidden_dim = args.M
        output_dim = 1
        criterion = nn.MSELoss()
    elif args.data=='workloads':
        input_dim = 6
        hidden_dim = args.M
        output_dim = 1
        criterion = nn.MSELoss()
    elif args.data=='msd':
        input_dim = 89
        hidden_dim = args.M
        output_dim = 1
        criterion = nn.MSELoss()
    else:
        pass
    
    print(f'Loading data {args.data}')
    train_loader, test_loader, train_dataset, test_dataset = load_data(args.data, args.batch_size)
    print(f'Finished loading data {args.data}')
    
    # Setting RFLAF model
    modelseed = args.modelseed
    if args.model=='RFLAF':
        # Setting parameters
        print(f'Using model RFLAF, modelseed={modelseed}')
        h, N, L, R = args.h, args.N, args.L, args.R
        print(f'RBF params:\th={h}\tN={N}\tL={L}\tR={R}')
        model = RFLAF(input_dim, hidden_dim, output_dim, h, N, L, R, modelseed).to(device)
        regularizer = CustomRegularizer(args.lambda1, args.lambda2, args.N, args.M)
    elif args.model=='RFLAFBS':
        # Setting parameters
        print(f'Using model RFLAF_BSpline, modelseed={modelseed}')
        N, L, R = args.N, args.L, args.R
        h = (R-L)/(N-1)
        print(f'BSpline params:\th={h}\tN={N}\tL={L}\tR={R}')
        model = RFLAF_BSpline(input_dim, hidden_dim, output_dim, N, L, R, modelseed).to(device)
        regularizer = CustomRegularizer(args.lambda1, args.lambda2, args.N, args.M)
    elif args.model=='RFLAFPL':
        print(f'Using model RFLAF_Poly, modelseed={modelseed}')
        N = args.N
        print(f'Poly params:\tN={N}')
        model = RFLAF_Poly(input_dim, hidden_dim, output_dim, N, modelseed).to(device)
        regularizer = CustomRegularizer(args.lambda1, args.lambda2, args.N, args.M)
    elif args.model=='RFMLP':
        print(f'Using model RFMLP, modelseed={modelseed}')
        model = RFMLP(input_dim, hidden_dim, output_dim, args.actfunc, args.modelseed).to(device)
    elif args.model=='MLP':
        print(f'Using model MLP, modelseed={modelseed}')
        model = RFMLP(input_dim, hidden_dim, output_dim, args.actfunc, args.modelseed, frozen=False).to(device)
    elif args.model=='LAN':
        print(f'Using model LAN, modelseed={modelseed}')
        h, N, L, R = args.h, args.N, args.L, args.R
        print(f'RBF params:\th={h}\tN={N}\tL={L}\tR={R}')
        model = RFLAF(input_dim, hidden_dim, output_dim, h, N, L, R, modelseed, frozen=False).to(device)
        regularizer = CustomRegularizer(args.lambda1, args.lambda2, args.N, args.M)
    elif args.model=='LANBS':
        print(f'Using model RFLAF_BSpline, modelseed={modelseed}')
        N, L, R = args.N, args.L, args.R
        h = (R-L)/(N-1)
        print(f'BSpline params:\th={h}\tN={N}\tL={L}\tR={R}')
        model = RFLAF_BSpline(input_dim, hidden_dim, output_dim, N, L, R, modelseed, frozen=False).to(device)
        regularizer = CustomRegularizer(args.lambda1, args.lambda2, args.N, args.M)
    elif args.model=='LANPL':
        print(f'Using model RFLAF_Poly, modelseed={modelseed}')
        N = args.N
        print(f'Poly params:\tN={N}')
        model = RFLAF_Poly(input_dim, hidden_dim, output_dim, N, modelseed, frozen=False).to(device)
        regularizer = CustomRegularizer(args.lambda1, args.lambda2, args.N, args.M)
    elif args.model=='KAN':
        print(f'Using model KAN, modelseed={modelseed}')
        N, L, R = args.N, args.L, args.R
        print(f'KAN params:\tN={args.N}\tL={args.L}\tR={args.R}')
        model = KAN([input_dim, max(hidden_dim//args.N, 10), output_dim], grid_size=args.N, spline_order=3, grid_range=[args.L, args.R]).to(device)
    
    # Setting logger file name
    if args.model=='RFLAF':
        actfunc = 'LAF'
        moreargs=f'_h={args.h}_N={args.N}_L={args.L}_R={args.R}_M={args.M}_lambda1={args.lambda1}_lambda2={args.lambda2}'
    elif args.model=='RFLAFBS':
        actfunc = 'BS'
        moreargs=f'_N={args.N}_L={args.L}_R={args.R}_M={args.M}_lambda1={args.lambda1}_lambda2={args.lambda2}'
    elif args.model=='RFLAFPL':
        actfunc = 'PL'
        moreargs=f'_N={args.N}_M={args.M}_lambda1={args.lambda1}_lambda2={args.lambda2}'
    elif args.model=='RFMLP':
        actfunc = args.actfunc
        moreargs=f'_M={args.M}'
    elif args.model=='MLP':
        actfunc = 'N'+args.actfunc
        moreargs=f'_M={args.M}'
    elif args.model=='LAN':
        actfunc = 'LAN'
        moreargs=f'_h={args.h}_N={args.N}_L={args.L}_R={args.R}_M={args.M}_lambda1={args.lambda1}_lambda2={args.lambda2}'
    elif args.model=='LANBS':
        actfunc = 'LANBS'
        moreargs=f'_N={args.N}_L={args.L}_R={args.R}_M={args.M}_lambda1={args.lambda1}_lambda2={args.lambda2}'
    elif args.model=='LANPL':
        actfunc = 'LANPL'
        moreargs=f'_N={args.N}_M={args.M}_lambda1={args.lambda1}_lambda2={args.lambda2}'
    elif args.model=='KAN':
        actfunc = 'KAN'
        moreargs=f'_N={args.N}_L={args.L}_R={args.R}_M={args.M}'
        
    moreargs=f'seed={modelseed}_epoch={args.epochs}{moreargs}'
    task_name = args.data + '_' + actfunc # taskname = 'dataset_modelname'
    
    # weight_decay=0.001 if args.model=='RFMLP' else 0.0
    if args.model in ['MLP', 'LAN', 'KAN', 'LANBS', 'LANPL']:
        weight_decay = 0.0001
    else:
        weight_decay = 0.0
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    warmup_iter = 8
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iter)
    epochs = args.epochs
    
    print(f'Start training model {args.model} on data {args.data}...')
    start_time = time.time()
    train_losses, test_losses, train_accuracies, test_accuracies, train_time, test_time = train()
    end_time = time.time()
    print(f'Finished in {end_time-start_time:.2f}s')
    
    os.makedirs('./times', exist_ok=True)
    np.savetxt(f"./times/{task_name}_time_{moreargs}.txt", [end_time-start_time])
    
    print(f'Saving losses...')
    save_losses()
    print(f'Finished saving losses')
    
    