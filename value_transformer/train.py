import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os
from tqdm import tqdm
import numpy as np
import gc
from model import ChessFormer

# ------------------- HYPERPARAMETERS -------------------
batch_size = 512  # Increased for larger data
gradient_accumulation_steps = 8  # Increased for stability on 50M data
block_size = 64

# 30M Parameter Configuration
n_embed = 512
n_head = 8
n_layer = 8
dropout = 0.1

max_iters = 10000       
eval_interval = 1000
save_interval = 5000     
learning_rate = 3e-4
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = learning_rate / 10
weight_decay = 0.01
grad_clip = 1.0

# Force float32 for stability if NaN persists (disable AMP for debugging)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = True # TEMPORARILY DISABLED TO PREVENT NANs
print(f"Using device: {device} | AMP: {use_amp}")

# ------------------- DATA LOADING (FROM .NPY) -------------------
class ChessValueDataset(Dataset):
    def __init__(self, x_path, y_path, split='train', val_ratio=0.1):
        print(f"Loading X from {x_path} and Y from {y_path}...")
        
        # Load with memory mapping for large files (50M+ positions)
        self.X = np.load(x_path, mmap_mode='r')  # [N, 64], int64 or uint8
        self.Y = np.load(y_path, mmap_mode='r')  # [N], float32 (tanh(cp/400))
        
        # Ensure shapes match
        assert len(self.X) == len(self.Y), "X and Y lengths don't match!"
        N = len(self.X)
        
        # Sanitize: Remove NaN/inf in Y
        valid_mask = ~np.isnan(self.Y) & ~np.isinf(self.Y)
        self.X = self.X[valid_mask]
        self.Y = self.Y[valid_mask]
        if len(self.X) < N:
            print(f"⚠️ REMOVED {N - len(self.X)} CORRUPT SAMPLES (NaN/Inf in Y)!")
        
        # Train/val split (random for simplicity)
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        val_size = int(len(indices) * val_ratio)
        if split == 'train':
            self.indices = indices[val_size:]
        else:
            self.indices = indices[:val_size]
        
        print(f"{split.capitalize()} Dataset Ready: {len(self.indices):,} positions.")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        x = torch.from_numpy(self.X[actual_idx].astype(np.int64))  # [64]
        y = torch.tensor(self.Y[actual_idx], dtype=torch.float32)  # scalar
        return x, y

# ------------------- SETUP DATASET -------------------
X_PATH = "/kaggle/input/cpboards/dataset_X_final.npy"  # Adjust path if needed
Y_PATH = "/kaggle/input/cpboards/dataset_Y_final.npy"  # Adjust path if needed

train_dataset = ChessValueDataset(X_PATH, Y_PATH, split='train', val_ratio=0.1)
val_dataset = ChessValueDataset(X_PATH, Y_PATH, split='val', val_ratio=0.1)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True, 
    num_workers=4,
    pin_memory=True,
    persistent_workers=True 
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True,
    persistent_workers=True 
)

# ------------------- MODEL & OPTIMIZER -------------------
model = ChessFormer(
    block_size=block_size,
    n_embed=n_embed,
    n_heads=n_head,
    n_layers=n_layer,
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scaler = torch.amp.GradScaler(enabled=use_amp)
criterion = nn.SmoothL1Loss(beta=1.0)  # Huber loss (smooth L1)

# ------------------- TRAINING LOOP -------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss(loader, num_batches=100):
    model.eval()
    losses = torch.zeros(num_batches)
    eval_iter = iter(loader)
    for k in range(num_batches):
        try:
            x, y = next(eval_iter)
        except StopIteration:
            eval_iter = iter(loader)
            x, y = next(eval_iter)
        x, y = x.to(device), y.to(device).unsqueeze(1)
        # Check for NaN in targets during eval
        if torch.isnan(y).any():
            print("WARNING: NaN found in validation targets!")
            continue
        pred = model(x)
        loss = criterion(pred, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

print(f"Starting Training | Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
best_loss = float('inf')
pbar = tqdm(range(max_iters), desc="Training")
train_iter = iter(train_loader)

for iter_num in pbar:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    accum_loss = 0
    for micro_step in range(gradient_accumulation_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x = x.to(device)
        y = y.to(device).unsqueeze(1)
        
        # Verify no NaNs in batch inputs
        if torch.isnan(y).any():
            print(f"SKIPPING BATCH {iter_num}: Found NaN in targets")
            continue
        
        with torch.amp.autocast(device_type='cuda', enabled=use_amp) if device=='cuda' else torch.no_grad():
            pred = model(x)
            # NaN check in pred
            if torch.isnan(pred).any():
                print(f"WARNING: NaN in predictions at iter {iter_num}! Skipping.")
                continue
            loss = criterion(pred, y) / gradient_accumulation_steps
        
        accum_loss += loss.item()
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    
    if use_amp:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
    optimizer.zero_grad(set_to_none=True)

    if iter_num % 10 == 0:
        real_loss = accum_loss * gradient_accumulation_steps
        pbar.set_description(f"Iter {iter_num} | Huber {real_loss:.5f} | LR {lr:.2e}")

    if (iter_num + 1) % eval_interval == 0:
        val_loss = estimate_loss(val_loader, num_batches=100)  # Use val_loader
        train_loss = estimate_loss(train_loader, num_batches=100)  # Quick train estimate
        print(f"\nStep {iter_num}: Train Huber {train_loss:.5f}, Val Huber {val_loss:.5f}")
        
        if val_loss < best_loss and not math.isnan(val_loss):
            best_loss = val_loss
            torch.save(model.state_dict(), "best_value_model.pt")
            print("--> Best model saved.")

    if (iter_num + 1) % save_interval == 0:
        checkpoint = {
            'iter_num': iter_num,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, f"value_ckpt_{iter_num}.pt")

torch.save(model.state_dict(), "final_value_model.pt")
print("Training complete!")