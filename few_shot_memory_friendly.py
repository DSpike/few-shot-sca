"""
Memory-Friendly Few-Shot SCA
=============================
Load only selected points, not full 100k traces
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

K_SHOT = 10
META_ITERS = 50
INNER_STEPS = 5
INNER_LR = 0.01
META_LR = 0.001

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

print("Opening file (not loading yet)...")
file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ATMEGA_AES_v1\ATM_AES_v1_fixed_key\ASCAD_data\ASCAD_databases\ATMega8515_raw_traces.h5'

# Use the known good POI range for ASCAD (around 45000-50000)
# These are documented in the ASCAD paper as the leakage points
POI_START = 45000
POI_LENGTH = 3000
selected_points = list(range(POI_START, POI_START + POI_LENGTH))

print(f"Using documented POI range: {POI_START} to {POI_START + POI_LENGTH}")
print("Loading only selected {len(selected_points)} points...")

with h5py.File(file_path, 'r') as f:
    # Load ONLY the selected columns
    X_prof = f['traces'][:30000, selected_points].astype(np.float32)
    X_attack = f['traces'][30000:40000, selected_points].astype(np.float32)

    meta_prof = f['metadata'][:30000]
    meta_attack = f['metadata'][30000:40000]

    plaintext_prof = np.array([m[0][3] for m in meta_prof])
    key_byte = meta_prof[0][2][3]
    masks_prof = np.array([m[3][3] for m in meta_prof])
    y_prof = sbox[plaintext_prof ^ key_byte ^ masks_prof]

    plaintext_attack = np.array([m[0][3] for m in meta_attack])
    masks_attack = np.array([m[3][3] for m in meta_attack])
    correct_key = meta_attack[0][2][3]

print(f"✓ Loaded data shape: {X_prof.shape}")
print(f"✓ Correct key: {hex(correct_key)}")

# Normalize
mean = X_prof.mean(axis=0)
std = X_prof.std(axis=0) + 1e-8
X_prof = (X_prof - mean) / std
X_attack = (X_attack - mean) / std

split = int(0.8 * len(X_prof))
X_train, y_train = X_prof[:split], y_prof[:split]
X_test, y_test = X_prof[split:], y_prof[split:]

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}\n")

# Simple model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 11, padding=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 11, padding=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 256)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TaskSampler:
    def __init__(self, X, y, k_shot=10):
        self.X, self.y, self.k_shot = X, y, k_shot
        self.class_idx = defaultdict(list)
        for i, label in enumerate(y):
            self.class_idx[label].append(i)
        self.valid_classes = [c for c in range(256) if len(self.class_idx[c]) >= k_shot + 3]
        print(f"  Valid classes: {len(self.valid_classes)}/256")

    def sample(self):
        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        for cls in self.valid_classes:
            sel = np.random.choice(self.class_idx[cls], self.k_shot + 3, replace=False)
            sup_x.append(self.X[sel[:self.k_shot]])
            sup_y.extend([cls] * self.k_shot)
            qry_x.append(self.X[sel[self.k_shot:]])
            qry_y.extend([cls] * 3)
        return (torch.FloatTensor(np.vstack(sup_x)), torch.LongTensor(sup_y),
                torch.FloatTensor(np.vstack(qry_x)), torch.LongTensor(qry_y))

def maml_inner_loop(model, sup_x, sup_y):
    adapted = SimpleCNN().to(device)
    adapted.load_state_dict(model.state_dict())
    opt = optim.SGD(adapted.parameters(), lr=INNER_LR)
    for _ in range(INNER_STEPS):
        opt.zero_grad()
        F.cross_entropy(adapted(sup_x), sup_y).backward()
        opt.step()
    return adapted

print("="*70)
print(f"Training MAML ({K_SHOT}-shot, {META_ITERS} iterations)")
print("="*70)

sampler = TaskSampler(X_train, y_train, k_shot=K_SHOT)
model = SimpleCNN().to(device)
meta_opt = optim.Adam(model.parameters(), lr=META_LR)

for i in range(META_ITERS):
    meta_opt.zero_grad()
    sup_x, sup_y, qry_x, qry_y = sampler.sample()
    sup_x, sup_y = sup_x.to(device), sup_y.to(device)
    qry_x, qry_y = qry_x.to(device), qry_y.to(device)

    adapted = maml_inner_loop(model, sup_x, sup_y)
    loss = F.cross_entropy(adapted(qry_x), qry_y)
    loss.backward()
    meta_opt.step()

    if (i + 1) % 10 == 0:
        print(f"  Iter {i+1}/{META_ITERS}: Loss = {loss.item():.4f}")

print("\n✓ Training complete\n")

# Attack
test_sampler = TaskSampler(X_test, y_test, k_shot=K_SHOT)
sup_x, sup_y, _, _ = test_sampler.sample()
adapted_model = maml_inner_loop(model, sup_x.to(device), sup_y.to(device))
adapted_model.eval()

all_probs = []
for i in range(0, len(X_attack), 100):
    batch = torch.FloatTensor(X_attack[i:i+100]).to(device)
    with torch.no_grad():
        probs = F.softmax(adapted_model(batch), dim=1).cpu().numpy()
    all_probs.append(probs)

probs = np.vstack(all_probs)
log_probs = np.log(probs + 1e-40)

print("Key Recovery Results:")
for n in [100, 500, 1000, 5000, 10000]:
    if n > len(log_probs):
        break
    scores = np.zeros(256)
    for k in range(256):
        intermediate = sbox[plaintext_attack[:n] ^ k ^ masks_attack[:n]]
        scores[k] = np.sum(log_probs[np.arange(n), intermediate])
    rank = np.argsort(-scores).tolist().index(correct_key)
    status = "✓ SUCCESS!" if rank == 0 else ("✓ EXCELLENT!" if rank < 10 else ("✓ GOOD!" if rank < 50 else ""))
    print(f"  {n:5d} traces → rank {rank:3d}  {status}")

print("\n✓ Done!")
