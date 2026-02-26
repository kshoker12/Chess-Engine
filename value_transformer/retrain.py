import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import os
from tqdm import tqdm
import numpy as np
from model import ChessFormer
import random

# ------------------- FINE-TUNE SETTINGS -------------------
CHECKPOINT_PATH = "/kaggle/input/latest-value-ft3/mini_value_2o6.pt"
FREEZE_LAYERS = 3

# ------------------- HYPERPARAMETERS -------------------
batch_size = 4096
gradient_accumulation_steps = 1
block_size = 64
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.1 
max_iters = 10000 
eval_interval = 500
save_interval = 1000
learning_rate = 2e-5
warmup_iters = 1000
lr_decay_iters = max_iters
min_lr = learning_rate / 10
weight_decay = 0.01
grad_clip = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = True # Match baseline

early_stop_patience = 8

print(f"Using device: {device} | AMP: {use_amp}")
print(f"Fine-Tuning | 39% old + 1% new + 60% latest | LR: {learning_rate}")

# ------------------- NECESSARY CHANGE: FAST DATA LOADING -------------------

def load_to_ram(x_path, y_path, name):
    print(f"Loading {name} to RAM...")
    x = np.load(x_path)
    y = np.load(y_path)
    # Validation for mismatched lengths
    valid = ~np.isnan(y) & ~np.isinf(y)
    # Load as uint8 to save memory, keep targets exactly as provided (-1 to 1)
    x_tensor = torch.from_numpy(x[valid]).to(torch.uint8)
    y_tensor = torch.from_numpy(y[valid]).to(torch.float32)
    return TensorDataset(x_tensor, y_tensor)

# Paths
OLD_X_PATH = "/kaggle/input/training-data/dataset_X_final.npy"
OLD_Y_PATH = "/kaggle/input/training-data/dataset_Y_final.npy"
# 'New' now refers to the self-play dataset we want to mix in 50:50
NEW_X_PATH = "/kaggle/input/training-data/dataset_X_sp.npy"
NEW_Y_PATH = "/kaggle/input/training-data/dataset_Y_sp.npy"

ds_old = load_to_ram(OLD_X_PATH, OLD_Y_PATH, "Old")
ds_new = load_to_ram(NEW_X_PATH, NEW_Y_PATH, "New")

# --- REPLACED: Efficient Vectorized Loader ---
class FastInfiniteLoader:
    def __init__(self, ds_old, ds_new, batch_size):
        self.old_x, self.old_y = ds_old.tensors
        self.new_x, self.new_y = ds_new.tensors
        self.batch_size = batch_size
        self.half_size = batch_size // 2
        print(f"FastInfiniteLoader initialized. Batch size: {batch_size} (50% Old / 50% New)")

    def __iter__(self):
        while True:
            # 1. Randomly sample indices (Vectorized)
            idx_old = torch.randint(len(self.old_x), (self.half_size,))
            idx_new = torch.randint(len(self.new_x), (self.batch_size - self.half_size,))
            
            # 2. Slice tensors directly (Fast C++ op, no Python loop)
            # Use copy() to ensure memory is contiguous for transfer
            x_batch = torch.cat([self.old_x[idx_old], self.new_x[idx_new]])
            y_batch = torch.cat([self.old_y[idx_old], self.new_y[idx_new]])
            
            yield x_batch.long(), y_batch

# Initialize Fast Loader
# Note: Use pin_memory=True if using CUDA, but here we do it manually in loop
# Since tensors are in RAM, we just yield them. 
# Pre-fetching is less critical as generation is now nearly instant.
train_loader = FastInfiniteLoader(ds_old, ds_new, batch_size)

# --- RE-LOAD VALIDATION DATA (Baseline Style) ---
class ValDataset(Dataset):
    def __init__(self, x_path, y_path, val_ratio=0.1, name="Val"):
        X = np.load(x_path, mmap_mode='r')
        Y = np.load(y_path, mmap_mode='r')
        valid = ~np.isnan(Y) & ~np.isinf(Y)
        X, Y = X[valid], Y[valid]
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        self.X, self.Y = X[indices[:int(len(indices)*val_ratio)]], Y[indices[:int(len(indices)*val_ratio)]]
        print(f"{name} Dataset: {len(self.X):,} positions")
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].astype(np.int64)), torch.tensor(self.Y[idx], dtype=torch.float32)

old_val_loader = DataLoader(ValDataset(OLD_X_PATH, OLD_Y_PATH, name="Old Val"), batch_size=batch_size)
new_val_loader = DataLoader(ValDataset(NEW_X_PATH, NEW_Y_PATH, name="New Val"), batch_size=batch_size)

# ------------------- MODEL & OPTIMIZER -------------------
model = ChessFormer(block_size=block_size, n_embed=n_embed, n_heads=n_head, n_layers=n_layer, dropout=dropout).to(device)
model = torch.compile(model, mode="default", dynamic=True)
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device), strict=False)

if FREEZE_LAYERS > 0:
    if hasattr(model, 'token_embedding_table'):
        for p in model.token_embedding_table.parameters(): p.requires_grad = False
    for i in range(FREEZE_LAYERS):
        for p in model.blocks[i].parameters(): p.requires_grad = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
scaler = torch.amp.GradScaler(enabled=use_amp)
criterion = nn.SmoothL1Loss(beta=1.0)

# ------------------- TRAINING LOOP -------------------
def get_lr(it):
    if it < warmup_iters: return learning_rate * (it + 1) / warmup_iters
    decay_ratio = min(1.0, (it - warmup_iters) / (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss(old_loader, new_loader, num_batches=40):
    model.eval()
    losses = {'old': [], 'new': []}
    for name, loader in [('old', old_loader), ('new', new_loader)]:
        it = iter(loader)
        for _ in range(num_batches):
            try: x, y = next(it)
            except StopIteration: break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).view(-1, 1) # Force strict [B, 1] shape
            pred = model(x)
            losses[name].append(criterion(pred, y).item())
    model.train()
    return np.mean(losses['old']), np.mean(losses['new'])

best_new_loss = float('inf')
no_improve_count = 0
train_iter = iter(train_loader)
pbar = tqdm(range(max_iters), desc="Fine-Tuning")

for iter_num in pbar:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    
    # Fast Infinite Loader never stops, but we can just call next()
    # It handles batch generation internally
    x, y = next(train_iter)
    
    # CRITICAL FIX: Ensure y is [batch_size, 1] to match pred shape exactly
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).view(-1, 1)
    
    with torch.amp.autocast(device_type='cuda', enabled=use_amp) if use_amp else torch.inference_mode(False):
        pred = model(x)
        loss = criterion(pred, y)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % 10 == 0:
        pbar.set_description(f"Iter {iter_num} | Huber {loss.item():.5f} | LR {lr:.2e}")
    
    if (iter_num + 1) % eval_interval == 0:
        old_val_loss, new_val_loss = estimate_loss(old_val_loader, new_val_loader)
        print(f"\nStep {iter_num}: Old={old_val_loss:.5f} | New={new_val_loss:.5f}")
        if new_val_loss < best_new_loss:
            best_new_loss = new_val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= early_stop_patience: break

torch.save(model.state_dict(), "mini_value_3o1.pt")