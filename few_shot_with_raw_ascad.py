"""
Few-Shot SCA using FULL RAW ASCAD Traces
=========================================
Uses ATMega8515_raw_traces.h5 with 100,000 time points
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Hyperparameters
K_SHOT = 10
N_FEATURES = 700  # Will select best 700 from 100k points
META_ITERS = 200
INNER_STEPS = 10
INNER_LR = 0.01
META_LR = 0.001
ATTACK_BATCH_SIZE = 100

# S-box
sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)

# SNR computation
def compute_snr(X, y, sample_points=10000):
    """Compute SNR - sample points to speed up"""
    print(f"  Computing SNR on {sample_points} sample points...")

    # Sample evenly across trace
    indices = np.linspace(0, X.shape[1]-1, sample_points, dtype=int)
    snr = np.zeros(sample_points)

    for idx, i in enumerate(indices):
        if idx % 1000 == 0:
            print(f"    Progress: {idx}/{sample_points}")

        class_means = []
        class_vars = []

        for c in range(256):
            mask = (y == c)
            if mask.sum() > 2:
                class_means.append(np.mean(X[mask, i]))
                class_vars.append(np.var(X[mask, i]))

        if len(class_means) > 1:
            snr[idx] = np.var(class_means) / (np.mean(class_vars) + 1e-12)

    return indices, snr

print("Loading RAW ASCAD traces...")
file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ATMEGA_AES_v1\ATM_AES_v1_fixed_key\ASCAD_data\ASCAD_databases\ATMega8515_raw_traces.h5'

with h5py.File(file_path, 'r') as f:
    traces = f['traces']
    metadata = f['metadata']

    print(f"  Total traces: {traces.shape[0]}, Length: {traces.shape[1]:,} points")

    # Split into profiling and attack
    n_prof = 50000
    n_attack = 10000

    X_prof = np.array(traces[:n_prof], dtype=np.float32)
    X_attack = np.array(traces[n_prof:n_prof+n_attack], dtype=np.float32)

    # Get metadata (trying different byte - byte 3 often has good leakage)
    meta_prof = metadata[:n_prof]
    meta_attack = metadata[n_prof:n_prof+n_attack]

    # Extract plaintext, key, masks for byte 3
    plaintext_prof = np.array([m[0][3] for m in meta_prof])  # plaintext byte 3
    key_byte = meta_prof[0][2][3]  # key byte 3
    masks_prof = np.array([m[3][3] for m in meta_prof])  # mask byte 3

    y_prof = sbox[plaintext_prof ^ key_byte ^ masks_prof]

    plaintext_attack = np.array([m[0][3] for m in meta_attack])
    masks_attack = np.array([m[3][3] for m in meta_attack])
    correct_key = meta_attack[0][2][3]

print(f"✓ Loaded {n_prof} profiling, {n_attack} attack traces")
print(f"✓ Using byte 3, correct key: {hex(correct_key)}")

# CRITICAL: SNR-based feature selection on FULL 100k points!
print(f"\nComputing SNR on 100,000-point traces...")
sampled_indices, snr_values = compute_snr(X_prof, y_prof, sample_points=10000)

print(f"\n  SNR Statistics:")
print(f"    Min: {snr_values.min():.6f}")
print(f"    Max: {snr_values.max():.6f}")
print(f"    Mean: {snr_values.mean():.6f}")
print(f"    Points with SNR > 0.1: {(snr_values > 0.1).sum()}")
print(f"    Points with SNR > 1.0: {(snr_values > 1.0).sum()}")

# Select top N_FEATURES by SNR
top_snr_idx = np.argsort(snr_values)[-N_FEATURES:]
selected_points = sampled_indices[top_snr_idx]
selected_points = np.sort(selected_points)  # Keep time order

print(f"\n✓ Selected top {N_FEATURES} points by SNR")
print(f"  Selected SNR range: {snr_values[top_snr_idx].min():.6f} to {snr_values[top_snr_idx].max():.6f}")

X_prof_sel = X_prof[:, selected_points]
X_attack_sel = X_attack[:, selected_points]

# Normalize
mean = X_prof_sel.mean(axis=0)
std = X_prof_sel.std(axis=0) + 1e-8
X_prof_sel = (X_prof_sel - mean) / std
X_attack_sel = (X_attack_sel - mean) / std

# Split
split = int(0.8 * len(X_prof_sel))
X_train, y_train = X_prof_sel[:split], y_prof[:split]
X_test, y_test = X_prof_sel[split:], y_prof[split:]

print(f"✓ Meta-train: {len(X_train)}, Meta-test: {len(X_test)}\n")

# Rest of the code (model, MAML, etc.) - same as before
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 11, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, 11, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.AvgPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, 11, padding=5)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TaskSampler:
    def __init__(self, X, y, k_shot=10):
        self.X = X
        self.y = y
        self.k_shot = k_shot
        self.class_idx = defaultdict(list)
        for i, label in enumerate(y):
            self.class_idx[label].append(i)
        self.valid_classes = [c for c in range(256) if len(self.class_idx[c]) >= k_shot + 3]
        print(f"  Valid classes: {len(self.valid_classes)}/256")

    def sample(self):
        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        for cls in self.valid_classes:
            idx = self.class_idx[cls]
            sel = np.random.choice(idx, self.k_shot + 3, replace=False)
            sup_x.append(self.X[sel[:self.k_shot]])
            sup_y.append([cls] * self.k_shot)
            qry_x.append(self.X[sel[self.k_shot:]])
            qry_y.append([cls] * 3)
        return (torch.FloatTensor(np.vstack(sup_x)), torch.LongTensor(np.hstack(sup_y)),
                torch.FloatTensor(np.vstack(qry_x)), torch.LongTensor(np.hstack(qry_y)))

def maml_inner_loop(model, sup_x, sup_y, steps=10, lr=0.01):
    adapted = ImprovedCNN().to(device)
    adapted.load_state_dict(model.state_dict())
    optimizer = optim.SGD(adapted.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(adapted(sup_x), sup_y)
        loss.backward()
        optimizer.step()
    return adapted

def maml_train_step(model, meta_opt, sampler):
    meta_opt.zero_grad()
    total_loss = 0
    for _ in range(4):
        sup_x, sup_y, qry_x, qry_y = sampler.sample()
        sup_x, sup_y = sup_x.to(device), sup_y.to(device)
        qry_x, qry_y = qry_x.to(device), qry_y.to(device)
        adapted = maml_inner_loop(model, sup_x, sup_y, steps=INNER_STEPS, lr=INNER_LR)
        qry_loss = F.cross_entropy(adapted(qry_x), qry_y)
        total_loss += qry_loss
    total_loss = total_loss / 4
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    meta_opt.step()
    return total_loss.item()

print("="*70)
print(f"Training MAML ({K_SHOT}-shot, {META_ITERS} iterations)")
print("="*70)

sampler = TaskSampler(X_train, y_train, k_shot=K_SHOT)
model = ImprovedCNN().to(device)
meta_optimizer = optim.Adam(model.parameters(), lr=META_LR)

losses = []
for i in range(META_ITERS):
    loss = maml_train_step(model, meta_optimizer, sampler)
    losses.append(loss)
    if (i + 1) % 20 == 0:
        print(f"  Iter {i+1}/{META_ITERS}: Loss = {loss:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('Meta-iteration')
plt.ylabel('Meta-loss')
plt.title('MAML Training (Raw ASCAD 100k points)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('maml_raw_ascad.png', dpi=100)
print("\n✓ Saved: maml_raw_ascad.png")

# Attack evaluation
print("\n" + "="*70)
print("Attack Evaluation")
print("="*70)

test_sampler = TaskSampler(X_test, y_test, k_shot=K_SHOT)
sup_x, sup_y, _, _ = test_sampler.sample()

print(f"Support set: {len(sup_y)} examples")
adapted_model = maml_inner_loop(model, sup_x.to(device), sup_y.to(device), steps=INNER_STEPS, lr=INNER_LR)
adapted_model.eval()

# Batch processing
all_probs = []
for i in range(0, len(X_attack_sel), ATTACK_BATCH_SIZE):
    batch = torch.FloatTensor(X_attack_sel[i:i+ATTACK_BATCH_SIZE]).to(device)
    with torch.no_grad():
        logits = adapted_model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    all_probs.append(probs)
    del batch, logits
    torch.cuda.empty_cache()

probs = np.vstack(all_probs)
log_probs = np.log(probs + 1e-40)

# Key ranking
traces_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
ranks = []

for n in traces_list:
    if n > len(log_probs):
        break
    scores = np.zeros(256)
    for k in range(256):
        intermediate = sbox[plaintext_attack[:n] ^ k ^ masks_attack[:n]]
        scores[k] = np.sum(log_probs[np.arange(n), intermediate])
    rank = np.argsort(-scores).tolist().index(correct_key)
    ranks.append(rank)
    status = "✓ SUCCESS!" if rank == 0 else ("✓ EXCELLENT!" if rank < 10 else "")
    print(f"  {n:5d} traces → rank {rank:3d}  {status}")

plt.figure(figsize=(8, 5))
plt.semilogx(traces_list[:len(ranks)], ranks, 'o-', linewidth=2, markersize=8, color='darkgreen')
plt.xlabel('Number of attack traces')
plt.ylabel('Key rank')
plt.title(f'MAML on Raw ASCAD ({K_SHOT}-shot, 100k points)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('attack_raw_ascad.png', dpi=100)
print("\n✓ Saved: attack_raw_ascad.png")

print(f"\n{'='*70}")
print(f"Best rank: {min(ranks)}")
print(f"Final rank: {ranks[-1]}")
if min(ranks) == 0:
    print("✓✓✓ PERFECT! Key recovered!")
elif min(ranks) < 10:
    print("✓✓ EXCELLENT! Publication-worthy results!")
elif min(ranks) < 50:
    print("✓ GOOD! Promising results.")
print('='*70)
