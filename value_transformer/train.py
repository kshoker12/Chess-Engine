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

# ------------------- MODE SWITCH (CHANGE THESE FOR FINE-TUNE / NORMAL) -------------------
IS_FINE_TUNE = True          # Set to True for fine-tuning on new data
CHECKPOINT_PATH = "/kaggle/input/datasets/kshoker/latest-value-ft2/mini_value_3o6.pt"  # Path to pre-trained checkpoint (ignored if not fine-tuning)
FREEZE_LAYERS = 0             # Number of early layers to freeze (0 = none)

# ------------------- HYPERPARAMETERS -------------------
batch_size = 2048
gradient_accumulation_steps = 1
block_size = 64

# Model config (keep as-is or scale down if desired)
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.1

max_iters = 5000       
eval_interval = 1000
save_interval = 5000     

# Learning rate: lower for fine-tuning
learning_rate = 3e-5 if IS_FINE_TUNE else 3e-4
warmup_iters = 1000 if IS_FINE_TUNE else 2000  # shorter warmup in fine-tune
lr_decay_iters = 20000
min_lr = learning_rate / 10
weight_decay = 0.01
grad_clip = 1.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = True  # Can keep enabled unless NaN issues persist
print(f"Using device: {device} | AMP: {use_amp}")
print(f"Mode: {'Fine-Tuning' if IS_FINE_TUNE else 'Normal Training'}")
if IS_FINE_TUNE:
    print(f"Loading checkpoint: {CHECKPOINT_PATH} | Freezing first {FREEZE_LAYERS} layers")

# ------------------- DATA LOADING (FROM .NPY) -------------------
class ChessValueDataset(Dataset):
    def __init__(self, x_path, y_path, split='train', val_ratio=0.1):
        print(f"Loading X from {x_path} and Y from {y_path}...")
        
        self.X = np.load(x_path, mmap_mode='r')
        self.Y = np.load(y_path, mmap_mode='r')
        
        assert len(self.X) == len(self.Y), "X and Y lengths don't match!"
        N = len(self.X)
        
        valid_mask = ~np.isnan(self.Y) & ~np.isinf(self.Y)
        self.X = self.X[valid_mask]
        self.Y = self.Y[valid_mask]
        if len(self.X) < N:
            print(f"⚠️ REMOVED {N - len(self.X)} CORRUPT SAMPLES!")
        
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
        x = torch.from_numpy(self.X[actual_idx].astype(np.int64))
        y = torch.tensor(self.Y[actual_idx], dtype=torch.float32)
        return x, y

def main():
    # ------------------- SETUP DATASET -------------------
    X_PATH = "/kaggle/input/datasets/kshoker/cp-eval-data2/dataset_X_final.npy"  # Adjust if needed
    Y_PATH = "/kaggle/input/datasets/kshoker/cp-eval-data2/dataset_Y_final.npy"  # Adjust if needed
    
    train_dataset = ChessValueDataset(X_PATH, Y_PATH, split='train', val_ratio=0.1)
    val_dataset = ChessValueDataset(X_PATH, Y_PATH, split='val', val_ratio=0.1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    # ------------------- MODEL & OPTIMIZER -------------------
    model = ChessFormer(
        block_size=block_size,
        n_embed=n_embed,
        n_heads=n_head,
        n_layers=n_layer,
        dropout=dropout
    ).to(device)
    model = torch.compile(model, mode="default", dynamic=True)
    
    # Load pre-trained checkpoint if fine-tuning
    if IS_FINE_TUNE and CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading pre-trained weights from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    
    # Freeze early layers if requested (fine-tuning only)
    if IS_FINE_TUNE and FREEZE_LAYERS > 0:
        print(f"Freezing first {FREEZE_LAYERS} transformer blocks...")
        for i in range(FREEZE_LAYERS):
            for param in model.blocks[i].parameters():
                param.requires_grad = False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    criterion = nn.SmoothL1Loss(beta = 1.0)  # Huber loss
    
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
            
            if torch.isnan(y).any():
                print(f"SKIPPING BATCH {iter_num}: Found NaN in targets")
                continue
            
            with torch.amp.autocast(device_type='cuda', enabled=use_amp) if device=='cuda' else torch.no_grad():
                pred = model(x)
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
        
        real_loss = accum_loss
        pbar.set_description(f"Iter {iter_num} | Huber {real_loss:.5f} | LR {lr:.2e}")
        
        if (iter_num + 1) % eval_interval == 0:
            val_loss = estimate_loss(val_loader, num_batches=100)
            train_loss = estimate_loss(train_loader, num_batches=100)
            print(f"\nStep {iter_num}: Train Huber {train_loss:.5f}, Val Huber {val_loss:.5f}")
    
    torch.save(model.state_dict(), "mini_value_3o7.pt")
    print("Training complete!")

if __name__ == "__main__":
    main()