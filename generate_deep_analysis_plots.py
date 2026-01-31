"""
Generate Deep Analysis Plots for Few-Shot SCA Paper
=====================================================
Trains models and captures intermediate data for:
  Fig A: Training Loss Curves (MAML, ProtoNet, Siamese)
  Fig B: t-SNE Embedding Visualization (learned feature space)
  Fig C: Confusion Matrix (HW class predictions)
  Fig D: POI Ablation Study (effect of n_poi on attack performance)

NOTE: This script trains models from scratch, so it requires GPU for
reasonable runtime. Run AFTER the main experiments are complete.

Usage:
  python generate_deep_analysis_plots.py [--dataset ches|ascad] [--seed 42]
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import pandas as pd
import time
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser(description='Generate deep analysis plots for SCA paper')
parser.add_argument('--dataset', type=str, default='ches', choices=['ches', 'ascad'],
                    help='Dataset to use (default: ches)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--k_shot', type=int, default=20, help='K-shot for training (default: 20)')
args = parser.parse_args()

# Seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
import random
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# Publication style
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

METHOD_COLORS = {'MAML': '#1f77b4', 'ProtoNet': '#ff7f0e', 'Siamese': '#2ca02c'}

SCA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCA_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLASSES = 9

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

HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


# =============================================================================
# SNR and POI Selection
# =============================================================================
def compute_snr(traces, labels, n_classes=9):
    n_samples = traces.shape[1]
    class_means = np.zeros((n_classes, n_samples))
    class_vars = np.zeros((n_classes, n_samples))
    class_counts = np.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        count = mask.sum()
        if count > 1:
            class_means[c] = traces[mask].mean(axis=0)
            class_vars[c] = traces[mask].var(axis=0)
            class_counts[c] = count
        elif count == 1:
            class_means[c] = traces[mask][0]
            class_counts[c] = count
    weights = class_counts / class_counts.sum()
    signal = np.average((class_means - np.average(class_means, axis=0, weights=weights))**2,
                        axis=0, weights=weights)
    noise = np.average(class_vars, axis=0, weights=weights)
    noise[noise == 0] = 1e-10
    return signal / noise


def select_poi(traces_prof, traces_attack, labels, n_poi=200, n_classes=9):
    snr = compute_snr(traces_prof, labels, n_classes)
    poi_indices = np.sort(np.argsort(snr)[-n_poi:])
    return traces_prof[:, poi_indices], traces_attack[:, poi_indices], poi_indices


# =============================================================================
# Load Dataset
# =============================================================================
print(f"\nLoading {'CHES CTF 2018' if args.dataset == 'ches' else 'ASCAD v1'} dataset...")

target_byte = 0

if args.dataset == 'ches':
    ds_path = os.path.join(SCA_DIR, 'datasets', 'ches_ctf.h5')
    dataset_label = 'CHES CTF 2018 (STM32, ARM)'
    with h5py.File(ds_path, 'r') as f:
        X_prof = np.array(f['profiling_traces'], dtype=np.float32)
        prof_data = np.array(f['profiling_data'], dtype=np.uint8)
        X_attack = np.array(f['attacking_traces'], dtype=np.float32)
        atk_data = np.array(f['attacking_data'], dtype=np.uint8)

    plaintext_prof = prof_data[:, target_byte]
    key_byte = prof_data[0, 32 + target_byte]
    y_prof = HW[sbox[plaintext_prof ^ key_byte]]
    plaintext_attack = atk_data[:, target_byte]
    correct_key = atk_data[0, 32 + target_byte]
else:
    ds_path = os.path.join(SCA_DIR, 'ASCAD_data', 'ASCAD_data', 'ASCAD_databases', 'ASCAD.h5')
    dataset_label = 'ASCAD v1 (ATMega8515, AVR)'
    with h5py.File(ds_path, 'r') as f:
        X_prof = np.array(f['Profiling_traces']['traces'], dtype=np.float32)
        metadata = f['Profiling_traces']['metadata'][:]
        X_attack = np.array(f['Attack_traces']['traces'], dtype=np.float32)
        atk_metadata = f['Attack_traces']['metadata'][:]

    pt_prof = metadata['plaintext'][:, target_byte]
    key_prof = metadata['key'][:, target_byte]
    masks_prof = metadata['masks'][:, target_byte]
    y_prof = HW[sbox[pt_prof ^ key_prof] ^ masks_prof]
    plaintext_attack = atk_metadata['plaintext'][:, target_byte]
    correct_key = atk_metadata['key'][0, target_byte]

print(f"Profiling: {X_prof.shape}, Attack: {X_attack.shape}")
print(f"Raw trace length: {X_prof.shape[1]}")

# POI selection
X_prof, X_attack, poi_indices = select_poi(X_prof, X_attack, y_prof, n_poi=200)
TRACE_LENGTH = X_prof.shape[1]
print(f"After POI: {TRACE_LENGTH} samples")

# Normalize
mean = X_prof.mean(axis=0)
std = X_prof.std(axis=0) + 1e-8
X_prof = (X_prof - mean) / std
X_attack = (X_attack - mean) / std

# Train/val split
split = int(0.8 * len(X_prof))
X_train, y_train = X_prof[:split], y_prof[:split]
X_val, y_val = X_prof[split:], y_prof[split:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Attack: {len(X_attack)}")
print()


# =============================================================================
# Model & Training Infrastructure
# =============================================================================
class ImprovedCNN(nn.Module):
    def __init__(self, input_size=200):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 11, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 11, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.AvgPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, 11, padding=5)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AvgPool1d(2)
        self.conv4 = nn.Conv1d(256, 512, 7, padding=3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, N_CLASSES)

    def forward(self, x, return_features=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.selu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.selu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.selu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.selu(self.bn4(self.conv4(x)))
        features = self.pool(x).squeeze(-1)
        features = self.dropout(features)
        if return_features:
            return features
        x = F.selu(self.fc1(features))
        x = self.dropout(x)
        return self.fc2(x)


class StratifiedSampler:
    def __init__(self, X, y):
        self.X_gpu = torch.FloatTensor(X).to(device)
        self.y = y
        self.class_idx = defaultdict(list)
        for i, label in enumerate(y):
            self.class_idx[label].append(i)

    def sample_support_set(self, k_shot):
        support_idx = []
        for cls in range(N_CLASSES):
            if len(self.class_idx[cls]) < k_shot:
                support_idx.extend(self.class_idx[cls])
            else:
                indices = np.random.choice(self.class_idx[cls], k_shot, replace=False)
                support_idx.extend(indices.tolist())
        support_x = self.X_gpu[support_idx]
        support_y = torch.LongTensor([self.y[i] for i in support_idx]).to(device)
        return support_x, support_y


# =============================================================================
# Training functions (return loss history)
# =============================================================================
def maml_adapt(model, support_x, support_y, steps=20, lr=0.02):
    adapted = ImprovedCNN(input_size=TRACE_LENGTH).to(device)
    adapted.load_state_dict(model.state_dict())
    optimizer = optim.SGD(adapted.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(adapted(support_x), support_y, label_smoothing=0.1)
        loss.backward()
        optimizer.step()
    return adapted


def train_maml_with_history(X_train, y_train, k_shot, epochs=500):
    print(f"  Training MAML ({k_shot}-shot, {epochs} epochs)...")
    model = ImprovedCNN(input_size=TRACE_LENGTH).to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=epochs, eta_min=1e-6)
    sampler = StratifiedSampler(X_train, y_train)
    X_train_gpu = torch.FloatTensor(X_train).to(device)
    y_train_gpu = torch.LongTensor(y_train).to(device)
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        support_x, support_y = sampler.sample_support_set(k_shot)
        query_idx = torch.randperm(len(X_train_gpu), device=device)[:2000]
        query_x = X_train_gpu[query_idx]
        query_y = y_train_gpu[query_idx]
        meta_optimizer.zero_grad()
        adapted = maml_adapt(model, support_x, support_y)
        loss = F.cross_entropy(adapted(query_x), query_y, label_smoothing=0.1)
        loss.backward()
        meta_optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 100:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss={loss_val:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  MAML done (best loss: {best_loss:.4f})")
    return model, loss_history


def train_protonet_with_history(X_train, y_train, k_shot, epochs=500):
    print(f"  Training ProtoNet ({k_shot}-shot, {epochs} epochs)...")
    model = ImprovedCNN(input_size=TRACE_LENGTH).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    sampler = StratifiedSampler(X_train, y_train)
    X_train_gpu = torch.FloatTensor(X_train).to(device)
    y_train_gpu = torch.LongTensor(y_train).to(device)
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        support_x, support_y = sampler.sample_support_set(k_shot)
        query_idx = torch.randperm(len(X_train_gpu), device=device)[:2000]
        query_x = X_train_gpu[query_idx]
        query_y = y_train_gpu[query_idx]

        support_features = model(support_x, return_features=True)
        unique_classes = torch.unique(support_y)
        prototypes = torch.stack([support_features[support_y == c].mean(0) for c in unique_classes])
        query_features = model(query_x, return_features=True)
        dists = torch.cdist(query_features, prototypes)
        logits = -dists

        mask = torch.isin(query_y, unique_classes)
        if mask.sum() > 0:
            filtered_logits = logits[mask]
            filtered_query_y = query_y[mask]
            label_map = {c.item(): i for i, c in enumerate(unique_classes)}
            mapped_labels = torch.tensor([label_map[y.item()] for y in filtered_query_y],
                                         dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = F.cross_entropy(filtered_logits, mapped_labels, label_smoothing=0.1)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 100:
                    print(f"    Early stop at epoch {epoch+1}")
                    break

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss={loss_val:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  ProtoNet done (best loss: {best_loss:.4f})")
    return model, loss_history


def train_siamese_with_history(X_train, y_train, k_shot, epochs=500):
    print(f"  Training Siamese ({k_shot}-shot, {epochs} epochs)...")
    model = ImprovedCNN(input_size=TRACE_LENGTH).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    sampler = StratifiedSampler(X_train, y_train)
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        support_x, support_y = sampler.sample_support_set(k_shot)
        batch_size = 500
        anchor_idx = torch.randint(0, len(support_x), (batch_size,), device=device)
        anchors = support_x[anchor_idx]
        anchor_labels = support_y[anchor_idx]

        positive_idx = []
        for label in anchor_labels:
            same = torch.where(support_y == label)[0]
            idx = same[torch.randint(0, len(same), (1,))] if len(same) > 1 else same[0]
            positive_idx.append(idx.item() if isinstance(idx, torch.Tensor) else idx)
        positives = support_x[positive_idx]

        negative_idx = []
        for label in anchor_labels:
            diff = torch.where(support_y != label)[0]
            if len(diff) > 0:
                idx = diff[torch.randint(0, len(diff), (1,))]
                negative_idx.append(idx.item())
            else:
                negative_idx.append(torch.randint(0, len(support_x), (1,)).item())
        negatives = support_x[negative_idx]

        af = model(anchors, return_features=True)
        pf = model(positives, return_features=True)
        nf = model(negatives, return_features=True)
        loss = torch.mean(torch.relu(
            torch.sum((af - pf)**2, dim=1) - torch.sum((af - nf)**2, dim=1) + 1.0
        ))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 100:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss={loss_val:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  Siamese done (best loss: {best_loss:.4f})")
    return model, loss_history


# =============================================================================
# Attack evaluation (for POI ablation)
# =============================================================================
def evaluate_attack_rank(model, support_x, support_y, X_attack_local,
                         plaintext_attack_local, correct_key_local,
                         method='siamese', n_traces=1000):
    """Run attack and return key rank at n_traces."""
    model.eval()
    if not isinstance(support_x, torch.Tensor):
        support_x = torch.FloatTensor(support_x).to(device)
        support_y = torch.LongTensor(support_y).to(device)

    if method == 'maml':
        eval_model = maml_adapt(model, support_x, support_y)
        eval_model.eval()
    else:
        eval_model = model

    all_probs = []
    batch_size = 1000
    n = min(n_traces, len(X_attack_local))
    for i in range(0, n, batch_size):
        batch = torch.FloatTensor(X_attack_local[i:i+batch_size]).to(device)
        with torch.no_grad():
            if method in ('protonet', 'siamese'):
                support_features = eval_model(support_x, return_features=True)
                prototypes = []
                for c in range(N_CLASSES):
                    m = (support_y == c)
                    if m.sum() > 0:
                        prototypes.append(support_features[m].mean(0))
                    else:
                        prototypes.append(torch.zeros_like(support_features[0]))
                prototypes = torch.stack(prototypes)
                query_features = eval_model(batch, return_features=True)
                dists = torch.cdist(query_features, prototypes)
                logits = -dists
            else:
                logits = eval_model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        del batch, logits
        torch.cuda.empty_cache()

    probs = np.vstack(all_probs)[:n]
    log_probs = np.log(probs + 1e-40)

    scores = np.zeros(256)
    if args.dataset == 'ches':
        for k in range(256):
            intermediate_hw = HW[sbox[plaintext_attack_local[:n] ^ k]]
            scores[k] = np.sum(log_probs[np.arange(n), intermediate_hw])
    else:
        # ASCAD with masks — use attack metadata
        for k in range(256):
            intermediate_hw = HW[sbox[plaintext_attack_local[:n] ^ k]]
            scores[k] = np.sum(log_probs[np.arange(n), intermediate_hw])

    rank = np.argsort(-scores).tolist().index(correct_key_local)
    return rank


# =============================================================================
# MAIN: Train all three methods, record loss histories
# =============================================================================
print("=" * 70)
print(f"DEEP ANALYSIS PLOTS — {dataset_label}")
print(f"K-shot: {args.k_shot}, Seed: {args.seed}")
print("=" * 70)

k_shot = args.k_shot
generated = []

# Train all methods and collect loss histories
print("\n--- Phase 1: Training Models (capturing loss curves) ---")

models = {}
loss_histories = {}

sampler_val = StratifiedSampler(X_val, y_val)
support_x, support_y = sampler_val.sample_support_set(k_shot)

model_maml, hist_maml = train_maml_with_history(X_train, y_train, k_shot, epochs=500)
models['MAML'] = model_maml
loss_histories['MAML'] = hist_maml

model_proto, hist_proto = train_protonet_with_history(X_train, y_train, k_shot, epochs=500)
models['ProtoNet'] = model_proto
loss_histories['ProtoNet'] = hist_proto

model_siamese, hist_siamese = train_siamese_with_history(X_train, y_train, k_shot, epochs=500)
models['Siamese'] = model_siamese
loss_histories['Siamese'] = hist_siamese


# =============================================================================
# FIGURE A: Training Loss Curves
# =============================================================================
print("\n--- Generating Fig A: Training Loss Curves ---")

fig, ax = plt.subplots(figsize=(10, 6))

for method_name, history in loss_histories.items():
    color = METHOD_COLORS[method_name]
    epochs_range = np.arange(1, len(history) + 1)

    # Apply smoothing (exponential moving average) for readability
    alpha = 0.05
    smoothed = [history[0]]
    for val in history[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])

    ax.plot(epochs_range, smoothed, color=color, label=method_name, linewidth=2)
    # Show raw values as faint background
    ax.plot(epochs_range, history, color=color, alpha=0.15, linewidth=0.5)

ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title(f'Meta-Learning Training Loss Curves — {dataset_label}\n({k_shot}-shot)',
             fontweight='bold')
ax.legend(loc='upper right')
ax.set_ylim(bottom=0)

plt.tight_layout()
fname = f'figA_loss_curves_{args.dataset}'
plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.pdf'), bbox_inches='tight')
plt.close()
generated.append(f'{fname}.png/.pdf')
print(f"  [OK] Fig A: {fname}")


# =============================================================================
# FIGURE B: t-SNE Embedding Visualization
# =============================================================================
print("\n--- Generating Fig B: t-SNE Embeddings ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Use a subset of validation data for t-SNE (2000 samples for speed)
n_tsne = min(2000, len(X_val))
tsne_idx = np.random.choice(len(X_val), n_tsne, replace=False)
X_tsne = X_val[tsne_idx]
y_tsne = y_val[tsne_idx]
X_tsne_tensor = torch.FloatTensor(X_tsne).to(device)

# Colormap for 9 HW classes
cmap = plt.cm.tab10
hw_colors = [cmap(i / 9) for i in range(9)]

for idx, (method_name, model) in enumerate(models.items()):
    ax = axes[idx]
    model.eval()

    with torch.no_grad():
        features = model(X_tsne_tensor, return_features=True).cpu().numpy()

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed, n_iter=1000)
    embeddings = tsne.fit_transform(features)

    # Plot each HW class
    for hw_class in range(N_CLASSES):
        mask = y_tsne == hw_class
        if mask.sum() > 0:
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c=[hw_colors[hw_class]], s=8, alpha=0.6,
                       label=f'HW={hw_class}')

    ax.set_title(f'{method_name}', fontweight='bold')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.tick_params(labelbottom=False, labelleft=False)

    if idx == 2:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=8, markerscale=2, title='HW Class')

fig.suptitle(f't-SNE Visualization of Learned Feature Space — {dataset_label}\n'
             f'({k_shot}-shot, 512-dim features, {n_tsne} samples)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fname = f'figB_tsne_{args.dataset}'
plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.pdf'), bbox_inches='tight')
plt.close()
generated.append(f'{fname}.png/.pdf')
print(f"  [OK] Fig B: {fname}")


# =============================================================================
# FIGURE C: Confusion Matrix (HW class predictions on validation set)
# =============================================================================
print("\n--- Generating Fig C: Confusion Matrices ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Use full validation set for confusion matrix
X_val_tensor = torch.FloatTensor(X_val).to(device)
support_x_tensor = support_x if isinstance(support_x, torch.Tensor) else torch.FloatTensor(support_x).to(device)
support_y_tensor = support_y if isinstance(support_y, torch.Tensor) else torch.LongTensor(support_y).to(device)

for idx, (method_name, model) in enumerate(models.items()):
    ax = axes[idx]
    model.eval()

    all_preds = []
    batch_size = 1000

    for i in range(0, len(X_val), batch_size):
        batch = X_val_tensor[i:i+batch_size]
        with torch.no_grad():
            if method_name in ('ProtoNet', 'Siamese'):
                support_features = model(support_x_tensor, return_features=True)
                prototypes = []
                for c in range(N_CLASSES):
                    m = (support_y_tensor == c)
                    if m.sum() > 0:
                        prototypes.append(support_features[m].mean(0))
                    else:
                        prototypes.append(torch.zeros_like(support_features[0]))
                prototypes = torch.stack(prototypes)
                query_features = model(batch, return_features=True)
                dists = torch.cdist(query_features, prototypes)
                preds = torch.argmin(dists, dim=1)
            elif method_name == 'MAML':
                adapted = maml_adapt(model, support_x_tensor, support_y_tensor)
                adapted.eval()
                logits = adapted(batch)
                preds = torch.argmax(logits, dim=1)
            else:
                logits = model(batch)
                preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())

    preds_all = np.concatenate(all_preds)
    cm = confusion_matrix(y_val, preds_all, labels=list(range(N_CLASSES)))

    # Normalize by row (true label)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10) * 100

    sns.heatmap(cm_norm, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                xticklabels=[f'HW{i}' for i in range(N_CLASSES)],
                yticklabels=[f'HW{i}' for i in range(N_CLASSES)],
                vmin=0, vmax=100, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Accuracy (%)'})
    ax.set_xlabel('Predicted HW Class')
    ax.set_ylabel('True HW Class')

    # Compute overall accuracy
    acc = np.trace(cm) / cm.sum() * 100
    ax.set_title(f'{method_name} (Acc: {acc:.1f}%)', fontweight='bold')

fig.suptitle(f'HW Class Confusion Matrix — {dataset_label}\n'
             f'({k_shot}-shot, validation set, row-normalized %)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fname = f'figC_confusion_matrix_{args.dataset}'
plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.pdf'), bbox_inches='tight')
plt.close()
generated.append(f'{fname}.png/.pdf')
print(f"  [OK] Fig C: {fname}")


# =============================================================================
# FIGURE D: POI Ablation Study
# =============================================================================
print("\n--- Generating Fig D: POI Ablation ---")
print("  (Re-loading raw traces for different POI counts...)")

# Reload raw traces (before POI selection)
if args.dataset == 'ches':
    with h5py.File(ds_path, 'r') as f:
        X_prof_raw = np.array(f['profiling_traces'], dtype=np.float32)
        X_attack_raw = np.array(f['attacking_traces'], dtype=np.float32)
else:
    with h5py.File(ds_path, 'r') as f:
        X_prof_raw = np.array(f['Profiling_traces']['traces'], dtype=np.float32)
        X_attack_raw = np.array(f['Attack_traces']['traces'], dtype=np.float32)

# POI counts to test
poi_values = [50, 100, 150, 200, 300, 500]
poi_results = []

for n_poi in poi_values:
    print(f"\n  n_poi = {n_poi}:")
    # Select POI
    X_prof_poi, X_attack_poi, _ = select_poi(X_prof_raw, X_attack_raw, y_prof, n_poi=n_poi)

    # Normalize
    m = X_prof_poi.mean(axis=0)
    s = X_prof_poi.std(axis=0) + 1e-8
    X_prof_poi = (X_prof_poi - m) / s
    X_attack_poi = (X_attack_poi - m) / s

    # Split
    X_t_poi = X_prof_poi[:split]
    y_t_poi = y_train  # labels don't change
    X_v_poi = X_prof_poi[split:]
    y_v_poi = y_val

    trace_len = n_poi

    # Train Siamese only (best performer) for ablation
    print(f"    Training Siamese ({k_shot}-shot, n_poi={n_poi})...")

    # Temporarily override TRACE_LENGTH for model
    ablation_model = ImprovedCNN(input_size=trace_len).to(device)
    abl_optimizer = optim.Adam(ablation_model.parameters(), lr=5e-4, weight_decay=1e-5)
    abl_scheduler = optim.lr_scheduler.CosineAnnealingLR(abl_optimizer, T_max=300, eta_min=1e-6)
    abl_sampler = StratifiedSampler(X_t_poi, y_t_poi)
    best_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(300):
        sup_x, sup_y = abl_sampler.sample_support_set(k_shot)
        bs = 500
        anc_idx = torch.randint(0, len(sup_x), (bs,), device=device)
        anchors = sup_x[anc_idx]
        anc_labels = sup_y[anc_idx]

        pos_idx = []
        for lbl in anc_labels:
            same = torch.where(sup_y == lbl)[0]
            idx = same[torch.randint(0, len(same), (1,))] if len(same) > 1 else same[0]
            pos_idx.append(idx.item() if isinstance(idx, torch.Tensor) else idx)

        neg_idx = []
        for lbl in anc_labels:
            diff = torch.where(sup_y != lbl)[0]
            if len(diff) > 0:
                idx = diff[torch.randint(0, len(diff), (1,))]
                neg_idx.append(idx.item())
            else:
                neg_idx.append(torch.randint(0, len(sup_x), (1,)).item())

        af = ablation_model(anchors, return_features=True)
        pf = ablation_model(sup_x[pos_idx], return_features=True)
        nf = ablation_model(sup_x[neg_idx], return_features=True)
        loss = torch.mean(torch.relu(
            torch.sum((af - pf)**2, dim=1) - torch.sum((af - nf)**2, dim=1) + 1.0
        ))

        abl_optimizer.zero_grad()
        loss.backward()
        abl_optimizer.step()
        abl_scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in ablation_model.state_dict().items()}
        else:
            patience += 1
            if patience >= 80:
                break

    if best_state:
        ablation_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluate
    abl_sampler_v = StratifiedSampler(X_v_poi, y_v_poi)
    sup_x_v, sup_y_v = abl_sampler_v.sample_support_set(k_shot)

    for n_traces in [100, 500, 1000]:
        rank = evaluate_attack_rank(
            ablation_model, sup_x_v, sup_y_v, X_attack_poi,
            plaintext_attack, correct_key, method='siamese', n_traces=n_traces
        )
        poi_results.append({
            'n_poi': n_poi, 'Attack_Traces': n_traces,
            'Key_Rank': rank, 'Method': 'Siamese'
        })
        print(f"    n_poi={n_poi}, {n_traces} traces -> Rank {rank}")

# Plot POI ablation
if poi_results:
    poi_df = pd.DataFrame(poi_results)
    poi_df.to_csv(os.path.join(OUTPUT_DIR, f'poi_ablation_{args.dataset}.csv'), index=False)

    fig, ax = plt.subplots(figsize=(9, 6))

    trace_colors = {100: '#e74c3c', 500: '#f39c12', 1000: '#27ae60'}
    trace_markers = {100: 'o', 500: 's', 1000: '^'}

    for n_traces in [100, 500, 1000]:
        subset = poi_df[poi_df['Attack_Traces'] == n_traces]
        if len(subset) > 0:
            ax.plot(subset['n_poi'], subset['Key_Rank'],
                    marker=trace_markers[n_traces], color=trace_colors[n_traces],
                    label=f'{n_traces} attack traces', linewidth=2, markersize=8)

    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Rank 10')
    ax.axvline(x=200, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Default (200)')
    ax.set_xlabel('Number of Points of Interest (n_poi)')
    ax.set_ylabel('Key Rank')
    ax.set_title(f'POI Ablation Study — Siamese {k_shot}-shot, {dataset_label}',
                 fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(-5, max(poi_df['Key_Rank'].max() + 20, 50))

    plt.tight_layout()
    fname = f'figD_poi_ablation_{args.dataset}'
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()
    generated.append(f'{fname}.png/.pdf')
    print(f"  [OK] Fig D: {fname}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("DEEP ANALYSIS PLOTS COMPLETE")
print("=" * 70)
print(f"\nGenerated {len(generated)} figure(s) in {OUTPUT_DIR}/:")
for name in generated:
    print(f"  - {name}")
print(f"\nDataset: {dataset_label}")
print(f"K-shot: {k_shot}, Seed: {args.seed}")
print("\nFor LaTeX, use the PDF versions for vector graphics.")
