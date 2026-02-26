import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os
from tqdm import tqdm
import pickle
import glob
import numpy as np  # Added numpy for memory efficiency
from model import ChessFormer

# ------------------- HYPERPARAMETERS -------------------
# OPTIMIZED FOR 16GB VRAM (Kaggle/Colab) & 8GB RAM
batch_size = 32          # Reduced from 96 to save VRAM
gradient_accumulation_steps = 8  # Increased to maintain effective batch size ~256
block_size = 256
n_embd = 768
n_head = 12
n_layer = 10
dropout = 0.1

max_iters = 10000
eval_interval = 500
save_interval = 500
learning_rate = 3e-4
warmup_iters = 1000
lr_decay_iters = max_iters
min_lr = learning_rate / 10
weight_decay = 0.01
grad_clip = 1.0

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps' 
else:
    device = 'cpu'

use_amp = True if device == 'cuda' else False
print(f"Using device: {device} | AMP: {use_amp}")

# ------------------- DATA LOADING -------------------
class ChessDataset(Dataset):
    def __init__(self, data_path, vocab_path, block_size):
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        self.vocab_size = len(self.vocab)
        self.stoi = self.vocab
        self.itos = {i: s for s, i in self.vocab.items()}
        
        print(f"Loading games from {data_path}...")
        
        # Optimize: Use intermediate lists but convert to numpy immediately
        data_ids = []
        target_ids = []
        
        # To avoid massive RAM usage reading file lines, we process line by line
        # and append to a list, then compact to numpy.
        with open(data_path, 'r') as f:
            for line in tqdm(f, desc="Processing text"):
                line = line.strip()
                if not line: continue
                
                tokens = line.split()
                if not tokens: continue
                
                ids = [self.stoi.get(t, self.stoi.get("[UNK]", 0)) for t in tokens]
                
                # --- LOGIC PRESERVED FROM YOUR SCRIPT ---
                first_token_str = self.itos.get(ids[0], "")
                is_black_start = False
                if len(first_token_str) >= 2 and first_token_str[1].isdigit():
                    rank = int(first_token_str[1])
                    if rank >= 5:
                        is_black_start = True
                
                curr_targets = []
                for i, token_id in enumerate(ids):
                    if is_black_start:
                        if i % 2 != 0:
                            curr_targets.append(token_id) 
                        else:
                            curr_targets.append(-100) 
                    else:
                        if i % 2 == 0:
                            curr_targets.append(token_id)
                        else:
                            curr_targets.append(-100) 
                # ----------------------------------------

                data_ids.extend(ids)
                target_ids.extend(curr_targets)
        
        # CRITICAL OPTIMIZATION: Convert to numpy uint16 (2 bytes per int vs 28 bytes)
        # Assuming vocab_size < 65535. If larger, use int32.
        self.data = np.array(data_ids, dtype=np.uint16)
        gc.collect()
        # For targets, we need signed int because of -100. int16 supports -32768 to 32767.
        self.targets = np.array(target_ids, dtype=np.int16)
        
        # Free memory explicitly
        del data_ids
        del target_ids
        
        self.block_size = block_size
        print(f"Loaded {len(self.data):,} tokens. RAM Optimized.")

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        # Numpy slicing is fast
        chunk_x = self.data[idx : idx + self.block_size + 1]
        chunk_y = self.targets[idx : idx + self.block_size + 1]
        
        # Convert to tensor at the last moment
        x = torch.from_numpy(chunk_x[:-1].astype(np.int64))
        y = torch.from_numpy(chunk_y[1:].astype(np.int64))
        
        return x, y

# Instantiate
train_dataset = ChessDataset("input.txt", "vocab.pkl", block_size)

# num_workers > 0 on Mac/MPS can sometimes cause issues, but on Kaggle/Linux 4 is fine.
# persistent_workers=True helps speed up epoch transitions.
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True, 
    num_workers=2, 
    pin_memory=True,
    persistent_workers=True 
)



# ------------------- MODEL & OPTIMIZER -------------------
model = ChessFormer(
    vocab_size=train_dataset.vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device
).to(device)

# Enable Flash Attention optimization hints
torch.set_float32_matmul_precision('high') 

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scaler = torch.amp.GradScaler(enabled=use_amp)

# ------------------- CHECKPOINTING & RESUME -------------------
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
latest_ckpt = None

ckpts = glob.glob(os.path.join(checkpoint_dir, "ckpt_*.pt"))
if ckpts:
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print(f"Found latest checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    iter_num = checkpoint['iter_num'] + 1
    print(f"Resuming from iteration {iter_num}")
else:
    iter_num = 0
    print("No checkpoint found — starting from scratch")

# ------------------- LR SCHEDULER -------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ------------------- EVALUATION -------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(50)
    # Very important: dataloaders are stateful. 
    # To avoid messing up the main iterator, we create a temporary one for eval 
    # or just take a few batches if we don't care about precise order.
    # Simpler: just loop the loader.
    eval_iter = iter(train_loader)
    for k in range(50):
        try:
            x, y = next(eval_iter)
        except StopIteration:
            eval_iter = iter(train_loader)
            x, y = next(eval_iter)
            
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

# ------------------- TRAINING LOOP -------------------
print(f"Starting training... ({len(train_loader)} batches/epoch)")
print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

best_loss = float('inf')
pbar = tqdm(range(iter_num, max_iters), initial=iter_num, total=max_iters, desc="Training")

train_iter = iter(train_loader)

# Accumulation Loop
optimizer.zero_grad(set_to_none=True)

for iter_num in pbar:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    accum_loss = 0
    
    # Gradient Accumulation Micro-Steps
    for micro_step in range(gradient_accumulation_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        
        # Context manager for mixed precision
        with torch.amp.autocast(device_type='cuda', enabled=use_amp) if device=='cuda' else torch.no_grad():
            # If using MPS, autocast is not yet fully supported for all ops, 
            # but we left 'use_amp' False for MPS in the setup block.
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps # Scale loss
        
        accum_loss += loss.item()
        
        # Backward
        scaler.scale(loss).backward()
    
    # Step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    if iter_num % 10 == 0:
        pbar.set_description(f"Iter {iter_num} | Loss {accum_loss:.4f} | LR {lr:.2e}")

    # Save & Eval
    if (iter_num + 1) % save_interval == 0:
        checkpoint = {
            'iter_num': iter_num,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'loss': accum_loss
        }
        torch.save(checkpoint, f"chessformer_ckpt_{iter_num}.pt")
    
    if (iter_num + 1) % eval_interval == 0:
        val_loss = estimate_loss()
        print(f"\nStep {iter_num}: Train Loss {accum_loss:.4f}, Val Loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

torch.save(model.state_dict(), "final_model.pt")
print("Training complete!")