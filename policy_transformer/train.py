import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import pickle
import time
from tqdm import tqdm
from torch.amp import autocast, GradScaler   # Updated import (non-deprecated)
from model import PolicyHead
import gc

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# ──────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────
batch_size = 475
grad_accum = 1              # effective batch size = 32 × 8 = 256
block_size = 128
max_iters = 100001
run_iters = 10000
eval_interval = 1000
save_interval = 1000
learning_rate = 1e-5
min_lr = learning_rate / 10
warmup_iters = 1000
lr_decay_iters = 100000
n_embd = 500
n_head = 10
n_layer = 8
dropout = 0.1
vocab_size = 4033
checkpoint_path = 'ultra_3o8.pt'
eval_iters = 50
weight_decay = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler(enabled=(device.type == 'cuda'))

with open('vocab1.pkl', 'rb') as f:
    stoi = pickle.load(f)
itos = {i: s for s, i in stoi.items()}
pad_id = stoi["|"]

# ──────────────────────────────────────────────────────────────
# Data loading (unchanged except pin_memory for faster transfer)
# ──────────────────────────────────────────────────────────────
def load_and_split(train_ratio=0.9, x_file = "full_X.pt", y_file = "full_Y.pt"):
    print("Loading preprocessed data...")
    full_X = torch.load(x_file)
    full_Y = torch.load(y_file)
    gc.collect()

    print("Corpus shapes:", full_X.shape, full_Y.shape)
    print("X min/max:", full_X.min().item(), full_X.max().item())
    print("Any negative in X?", (full_X < 0).any().item())
    print("Tokens >= vocab_size?", (full_X >= len(stoi)).any().item())  # safety check
    
    # Optional: simple train/val split by position (temporal-ish, avoids shuffle leakage)
    split_idx = int(0.9 * len(full_X))
    X_train, Y_train = full_X[:split_idx], full_Y[:split_idx]
    X_val,   Y_val   = full_X[split_idx:],  full_Y[split_idx:]
    
    del full_X, full_Y
    gc.collect()
    
    print(f"Train tokens: {len(X_train):,}, Val tokens: {len(X_val):,}")
    return X_train, Y_train, X_val, Y_val

X_train, Y_train, X_val, Y_val = load_and_split(x_file = "full_X.pt", y_file = "full_Y.pt")
X_train_weak, Y_train_weak, X_val_weak, Y_val_weak  = load_and_split(x_file = "weak_X.pt", y_file = "weak_Y.pt")
X_train_elite, Y_train_elite, X_val_elite, Y_val_elite = load_and_split(x_file = "elite_X.pt", y_file = "elite_Y.pt")


def get_batch(split):
    if split == 'train':
        data_x, data_y = X_train, Y_train
        elite_x, elite_y = X_train_elite, Y_train_elite
        weak_x,  weak_y  = X_train_weak,  Y_train_weak
    else:
        data_x, data_y = X_val, Y_val
        elite_x, elite_y = X_val_elite, Y_val_elite
        weak_x,  weak_y  = X_val_weak,  Y_val_weak

    n_normal = 445
    n_elite  = 25
    n_weak   = 5

    ix  = torch.randint(0, len(data_x)  - block_size + 1, (n_normal,))
    ixe = torch.randint(0, len(elite_x) - block_size + 1, (n_elite,))
    ixw = torch.randint(0, len(weak_x)  - block_size + 1, (n_weak,))

    xb  = torch.stack([data_x[i:i+block_size]  for i in ix])
    yb  = torch.stack([data_y[i:i+block_size]  for i in ix])

    xbe = torch.stack([elite_x[i:i+block_size] for i in ixe])
    ybe = torch.stack([elite_y[i:i+block_size] for i in ixe])

    xbw = torch.stack([weak_x[i:i+block_size]  for i in ixw])
    ybw = torch.stack([weak_y[i:i+block_size]  for i in ixw])

    x = torch.cat([xb, xbe, xbw], dim=0)
    y = torch.cat([yb, ybe, ybw], dim=0)

    x = x.to(device, dtype=torch.long, non_blocking=True)
    y = y.to(device, dtype=torch.long, non_blocking=True)

    return x, y

# ──────────────────────────────────────────────────────────────
# Loss estimation (unchanged)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            Xb, Yb = get_batch(split)
            with autocast(device_type='cuda'):
                _, loss = model(Xb, Yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# LR schedule (unchanged)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ──────────────────────────────────────────────────────────────
# Main Training Loop with Gradient Accumulation
# ──────────────────────────────────────────────────────────────
def main():
    torch.set_float32_matmul_precision('high') 
    model = PolicyHead(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)
    model = torch.compile(model, mode="default", dynamic=True)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    iter_num = 0
    best_val_loss = 1e9

    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint.get('best_val_loss', 1e9)
            print(f"Loaded at iter {iter_num}, best val loss {best_val_loss:.4f}")
        except Exception as e:
            print(f"Checkpoint load failed: {e}. Starting fresh.")

    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Effective batch size: {batch_size * grad_accum}")

    model.train()
    target_end_iter = min(max_iters, iter_num + run_iters)

    accum_loss = 0.0   # for printing averaged loss

    pbar = tqdm(total=target_end_iter - iter_num, desc="Training", unit="iter", dynamic_ncols=True)

    while iter_num < target_end_iter:
        lr = get_lr(iter_num)
        for g in optimizer.param_groups:
            g['lr'] = lr

        if iter_num > 0 and (iter_num % eval_interval == 0):
            losses = estimate_loss(model)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

        # Step optimizer every grad_accum micro-steps
        for i in range(grad_accum):
            xb, yb = get_batch('train')
            torch.cuda.empty_cache()  # helps avoid fragmentation
    
            with autocast(device_type='cuda'):
                _, loss = model(xb, yb)
                loss = loss / grad_accum   # scale for accumulation
    
            scaler.scale(loss).backward()
            accum_loss += loss.item()
            
                      
        scaler.unscale_(optimizer)                  # required before clipping with AMP
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Print averaged loss over accumulation steps
        print(f"step {iter_num}: train loss {accum_loss:.4f}, lr {lr:.2e}")
        accum_loss = 0.0      
        
        iter_num += 1
        pbar.update(1)

    pbar.close()

    # Final save
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'scaler': scaler.state_dict(),
    }
    torch.save(checkpoint, "ultra_3o9.pt")
    losses = estimate_loss(model)
    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    print(f"Finished at iter {iter_num}. Saved.")

if __name__ == "__main__":
    main()