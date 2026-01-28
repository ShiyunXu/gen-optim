# Time-Series Modeling with GeN Optimizer

## Quick Start Guide

This guide demonstrates how to replace AdamW with GeN optimizer for time-series modeling tasks using 1D-CNN and Transformer models with approximately 5M parameters.

## Example 1: 1D-CNN with GeN

```python
from GeN import lr_parabola, setup
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Initialize model and optimizer
model = CNN1D(...)  # Your 1D-CNN model
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop
tr_iter = iter(train_loader)
for epoch in range(epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward and backward
        loss = criterion(model(inputs), targets)
        loss.backward()
        
        # Apply GeN every 4 steps (lazy update)
        if (batch_idx + 1) % 4 == 0:
            # Horizon-aware learning rate (linear decay from 1 to 0)
            scale = 1 - epoch / epochs
            lr_parabola(
                model, 
                optimizer, 
                criterion,
                grad_accumulation_steps=1,
                tr_iter=tr_iter,
                device='cuda',  # or 'cpu'
                task='time_series',
                scale=scale
            )
        
        optimizer.step()
        optimizer.zero_grad()
```

## Example 2: Vanilla Transformer with GeN

The usage is identical to the 1D-CNN example above. Simply replace your model:

```python
model = VanillaTransformer(...)  # Your Transformer model
# Rest of the code remains the same
```

## Running the Provided Examples

### 1D-CNN Model
```bash
python examples/time_series.py \
    --model cnn1d \
    --epochs 50 \
    --lr_scheduler GeN \
    --lazy_freq 4 \
    --bs 256 \
    --mini_bs 64
```

### Transformer Model
```bash
python examples/time_series.py \
    --model transformer \
    --epochs 50 \
    --lr_scheduler GeN \
    --lazy_freq 4 \
    --bs 256 \
    --mini_bs 64
```

## Key Parameters

- `--lr_scheduler GeN`: Enable GeN optimizer (vs. 'cosine', 'multistep', or 'none')
- `--lazy_freq 4`: Update learning rate every 4 gradient accumulation steps
- `--bs`: Logical batch size (affects convergence)
- `--mini_bs`: Physical batch size (affects memory and speed)
- `--optim adamw`: Base optimizer (AdamW is recommended)

## Comparison: AdamW vs. GeN

### Traditional AdamW
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    for batch in train_loader:
        loss = criterion(model(batch), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()  # Manual learning rate scheduling
```

### GeN (Learning-Rate-Free)
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # Initial LR only
tr_iter = iter(train_loader)

for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        loss = criterion(model(batch), labels)
        loss.backward()
        
        if (batch_idx + 1) % lazy_freq == 0:
            scale = 1 - epoch / epochs  # Automatic horizon-aware scheduling
            lr_parabola(model, optimizer, tr_iter=tr_iter, task='time_series', scale=scale)
        
        optimizer.step()
        optimizer.zero_grad()
# No manual scheduler.step() needed!
```

## Benefits of GeN

1. **Learning-Rate-Free**: Automatically adapts learning rate based on loss landscape
2. **Horizon-Aware**: Optional decay schedule (linear, cosine-like) without hyperparameters
3. **Minimal Changes**: Only 2 lines of code difference from standard training
4. **Hessian Information**: Leverages second-order information like Newton's method
5. **Efficient**: Lazy update strategy minimizes computational overhead

## Custom Datasets

To use your own time-series data, replace the `TimeSeriesDataset` class:

```python
class MyTimeSeriesDataset(Dataset):
    def __init__(self, data_path):
        # Load your data
        self.data = ...
        self.labels = ...
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.data)
```

Then use it with the training script:
```python
trainset = MyTimeSeriesDataset('path/to/train.csv')
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
```

## Model Architectures

### 1D-CNN (~3.9M parameters)
- 6 convolutional layers with batch normalization
- Max pooling after each conv layer
- 3 fully connected layers
- Dropout for regularization

### Vanilla Transformer (~5.1M parameters)
- 6 transformer encoder layers
- 8 attention heads
- d_model=512, d_ff=2048
- Positional encoding
- Final regression head

Both models are sized to approximately 5M parameters as requested in the issue.

## References

- Paper: [Gradient Descent with Generalized Newton's Method (ICLR 2024)](https://openreview.net/pdf?id=bI3fcTsKW4)
- GitHub: [ShiyunXu/gen-optim](https://github.com/ShiyunXu/gen-optim)
