"""
Time-Series Modeling with GeN Optimizer

This example demonstrates how to use GeN optimizer for time-series regression tasks
with 1D-CNN and Vanilla Transformer models. It shows how to replace AdamW with GeN
for models with approximately 5M parameters.

Usage:
    python time_series.py --model cnn1d --dataset synthetic --epochs 50 --lr_scheduler GeN
    python time_series.py --model transformer --dataset synthetic --epochs 50 --lr_scheduler GeN
"""

from GeN import setup, lr_parabola

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from time import time


class TimeSeriesDataset(Dataset):
    """Synthetic time-series dataset for regression"""
    def __init__(self, num_samples=10000, seq_len=100, num_features=1):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_features = num_features
        
        # Generate synthetic time-series data
        self.data = []
        self.labels = []
        for _ in range(num_samples):
            # Generate a sine wave with noise
            t = np.linspace(0, 4*np.pi, seq_len)
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            noise = np.random.normal(0, 0.1, seq_len)
            series = np.sin(freq * t + phase) + noise
            
            # Predict the next value
            label = np.sin(freq * (t[-1] + 4*np.pi/seq_len) + phase)
            
            self.data.append(series.reshape(-1, num_features))
            self.labels.append(label)
        
        self.data = torch.FloatTensor(np.array(self.data))
        self.labels = torch.FloatTensor(np.array(self.labels))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CNN1D(nn.Module):
    """1D-CNN for time-series regression (~5M parameters)"""
    def __init__(self, input_dim=1, seq_len=100, num_filters=256, num_layers=6):
        super(CNN1D, self).__init__()
        
        layers = []
        in_channels = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            in_channels = num_filters
            num_filters = min(num_filters * 2, 512)
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, seq_len)
            dummy_output = self.conv_layers(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class VanillaTransformer(nn.Module):
    """Vanilla Transformer for time-series regression (~5M parameters)"""
    def __init__(self, input_dim=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(VanillaTransformer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use the last timestep for prediction
        x = x[:, -1, :]
        x = self.fc(x)
        return x.squeeze(-1)


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    print('==> Preparing time-series data..')
    trainset = TimeSeriesDataset(num_samples=args.train_samples, seq_len=args.seq_len)
    testset = TimeSeriesDataset(num_samples=args.test_samples, seq_len=args.seq_len)
    
    trainloader = DataLoader(trainset, batch_size=args.mini_bs, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    n_acc_steps = args.bs // args.mini_bs  # gradient accumulation steps
    
    # Model
    print(f'==> Building {args.model} model..')
    if args.model == 'cnn1d':
        net = CNN1D(input_dim=1, seq_len=args.seq_len).to(device)
    elif args.model == 'transformer':
        net = VanillaTransformer(input_dim=1).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    num_params = count_parameters(net)
    print(f'Model has {num_params:,} trainable parameters')
    
    criterion = nn.MSELoss()
    
    # Optimizer
    if args.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print('Optimizer does not exist!!!')
        return
    
    # Learning rate scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    else:
        print('No lr scheduler...')
    
    def train(epoch):
        tr_iter = iter(trainloader)
        
        net.train()
        train_loss = 0
        total_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc=f'Epoch {epoch}')):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            train_loss += loss.item()
            loss /= n_acc_steps
            loss.backward()
            
            if (batch_idx + 1) % n_acc_steps == 0:
                if ((batch_idx + 1) / n_acc_steps) % args.lazy_freq == 0:
                    if args.lr_scheduler == 'GeN':
                        # Horizon-aware learning rate schedule (linear decay from 1 to 0)
                        scale_list = np.linspace(1, 0, args.epochs + 1)[:-1]
                        lr_parabola(net, optimizer, criterion, n_acc_steps, 
                                  tr_iter=tr_iter, device=device, task='time_series', scale=scale_list[epoch])
                
                optimizer.step()
                optimizer.zero_grad()
                total_batches += 1
            
            if (batch_idx + 1) % (len(trainloader) // 10) == 0:
                avg_loss = train_loss / (batch_idx + 1)
                print(f'Epoch: {epoch} | Train Loss: {avg_loss:.6f}')
    
    def test(epoch):
        net.eval()
        test_loss = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(testloader)
        print(f'Epoch: {epoch} | Test Loss: {avg_test_loss:.6f}')
        return avg_test_loss
    
    # Training loop
    start_time = time()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        train(epoch)
        test_loss = test(epoch)
        
        if args.lr_scheduler in ['cosine', 'multistep']:
            scheduler.step()
        
        if test_loss < best_loss:
            best_loss = test_loss
            print(f'New best test loss: {best_loss:.6f}')
    
    print(f"Total training time for {args.epochs} epochs = {time() - start_time:.2f}s")
    print(f"Best test loss: {best_loss:.6f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch Time-Series Training with GeN')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--bs', default=256, type=int, help='logical batch size')
    parser.add_argument('--mini_bs', type=int, default=64, help='physical batch size')
    parser.add_argument('--train_samples', type=int, default=10000, help='number of training samples')
    parser.add_argument('--test_samples', type=int, default=2000, help='number of test samples')
    parser.add_argument('--seq_len', type=int, default=100, help='time-series sequence length')
    parser.add_argument('--model', default='cnn1d', type=str, choices=['cnn1d', 'transformer'],
                       help='model architecture')
    parser.add_argument('--lazy_freq', default=4, type=int, help='lr update frequency for GeN')
    parser.add_argument('--optim', default='adamw', type=str, choices=['sgd', 'adamw'],
                       help='base optimizer')
    parser.add_argument('--lr_scheduler', default='none', type=str, choices=['none', 'cosine', 'multistep', 'GeN'],
                       help='learning rate scheduler')
    parser.add_argument('--dataset', default='synthetic', type=str, help='dataset name')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    args = parser.parse_args()
    
    setup(args.seed)
    print(vars(args))
    main(args)
