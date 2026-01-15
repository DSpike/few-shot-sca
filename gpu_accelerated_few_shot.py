"""
GPU-Accelerated Few-Shot SCA
=============================
Optimized for maximum GPU utilization
- Uses GPU-based k-means (PyTorch implementation)
- Larger batch sizes
- Reduced CPU bottlenecks
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

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

# =============================================================================
# GPU-Accelerated K-Means
# =============================================================================
def kmeans_gpu(X_tensor, n_clusters, max_iter=50):
    """
    GPU-accelerated k-means clustering
    X_tensor: (n_samples, n_features) on GPU
    """
    n_samples = X_tensor.shape[0]

    # Random initialization
    idx = torch.randperm(n_samples, device=device)[:n_clusters]
    centroids = X_tensor[idx].clone()

    for _ in range(max_iter):
        # Compute distances to all centroids (GPU operation)
        distances = torch.cdist(X_tensor, centroids)
        labels = torch.argmin(distances, dim=1)

        # Update centroids
        new_centroids = torch.stack([
            X_tensor[labels == k].mean(dim=0) if (labels == k).sum() > 0
            else centroids[k]
            for k in range(n_clusters)
        ])

        # Check convergence
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return centroids, labels

# =============================================================================
# GPU-Accelerated Minimal Variance Sampler
# =============================================================================
class GPUMinimalVarianceSampler:
    """
    GPU-accelerated minimal variance sampling
    """
    def __init__(self, X, y):
        # Keep data on GPU
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.LongTensor(y).to(device)

        self.class_idx = defaultdict(list)
        for i in range(len(y)):
            self.class_idx[y[i]].append(i)

        # Pre-compute minimal variance shots for each k
        self.cached_shots = {}

    def sample_minimal_variance(self, k_shot):
        """Sample using GPU k-means"""
        if k_shot in self.cached_shots:
            return self.cached_shots[k_shot]

        support_idx = []

        for cls in range(256):
            if len(self.class_idx[cls]) < k_shot:
                continue

            class_samples = self.X[self.class_idx[cls]]

            if k_shot == 1:
                # Select sample closest to mean
                mean = class_samples.mean(dim=0)
                distances = torch.sum((class_samples - mean)**2, dim=1)
                idx = torch.argmin(distances).item()
                support_idx.append(self.class_idx[cls][idx])
            else:
                # GPU k-means
                centroids, _ = kmeans_gpu(class_samples, k_shot)

                # Find closest samples to centroids
                for center in centroids:
                    distances = torch.sum((class_samples - center)**2, dim=1)
                    idx = torch.argmin(distances).item()
                    support_idx.append(self.class_idx[cls][idx])

        support_x = self.X[support_idx]
        support_y = self.y[support_idx]

        # Cache result
        self.cached_shots[k_shot] = (support_x, support_y)
        return support_x, support_y

# =============================================================================
# Model
# =============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 11, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 11, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 256)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# =============================================================================
# MAML Training
# =============================================================================
def train_maml_gpu(X_train, y_train, k_shot, epochs=100, inner_steps=5):
    """GPU-optimized MAML training"""
    print(f"Training MAML ({k_shot}-shot)...")

    model = SimpleCNN(X_train.shape[1]).to(device)
    meta_opt = optim.Adam(model.parameters(), lr=1e-3)
    sampler = GPUMinimalVarianceSampler(X_train, y_train)

    # Pre-sample support set (cached)
    sup_x, sup_y = sampler.sample_minimal_variance(k_shot)

    # Create query set from remaining data
    all_idx = set(range(len(X_train)))
    sup_idx_set = set([i for i in range(len(X_train)) if y_train[i] in sup_y.cpu().numpy()])
    query_idx = list(all_idx - sup_idx_set)[:1000]  # Use 1000 query samples

    query_x = torch.FloatTensor(X_train[query_idx]).to(device)
    query_y = torch.LongTensor(y_train[query_idx]).to(device)

    start_time = time.time()

    for epoch in range(epochs):
        # Inner loop adaptation
        adapted = SimpleCNN(X_train.shape[1]).to(device)
        adapted.load_state_dict(model.state_dict())
        inner_opt = optim.SGD(adapted.parameters(), lr=0.01)

        for _ in range(inner_steps):
            inner_opt.zero_grad()
            loss = F.cross_entropy(adapted(sup_x), sup_y)
            loss.backward()
            inner_opt.step()

        # Meta-update on query set
        meta_opt.zero_grad()
        query_loss = F.cross_entropy(adapted(query_x), query_y)
        query_loss.backward()
        meta_opt.step()

        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {query_loss.item():.4f} (Time: {elapsed:.1f}s)")

    print(f"✓ Training complete ({time.time() - start_time:.1f}s)\n")
    return model

# =============================================================================
# Prototypical Networks
# =============================================================================
def train_protonet_gpu(X_train, y_train, k_shot, epochs=100):
    """GPU-optimized Prototypical Networks"""
    print(f"Training ProtoNet ({k_shot}-shot)...")

    model = SimpleCNN(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sampler = GPUMinimalVarianceSampler(X_train, y_train)

    sup_x, sup_y = sampler.sample_minimal_variance(k_shot)

    # Query set
    all_idx = set(range(len(X_train)))
    sup_idx_set = set([i for i in range(len(X_train)) if y_train[i] in sup_y.cpu().numpy()])
    query_idx = list(all_idx - sup_idx_set)[:1000]

    query_x = torch.FloatTensor(X_train[query_idx]).to(device)
    query_y = torch.LongTensor(y_train[query_idx]).to(device)

    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Compute prototypes
        sup_features = model(sup_x)
        prototypes = torch.stack([
            sup_features[sup_y == c].mean(dim=0)
            for c in sup_y.unique()
        ])

        # Compute distances for query
        query_features = model(query_x)
        distances = torch.cdist(query_features, prototypes)
        logits = -distances

        # Map query labels to prototype indices
        unique_classes = sup_y.unique().cpu().numpy()
        label_map = {c: i for i, c in enumerate(unique_classes)}
        mapped_query_y = torch.LongTensor([label_map.get(y.item(), 0) for y in query_y]).to(device)

        loss = F.cross_entropy(logits, mapped_query_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f} (Time: {elapsed:.1f}s)")

    print(f"✓ Training complete ({time.time() - start_time:.1f}s)\n")
    return model

# =============================================================================
# Attack Evaluation
# =============================================================================
def evaluate_attack_gpu(model, sup_x, sup_y, X_attack, plaintext_attack, masks_attack, correct_key):
    """GPU-optimized attack evaluation"""
    model.eval()

    # Adapt model on support set
    adapted = SimpleCNN(X_attack.shape[1]).to(device)
    adapted.load_state_dict(model.state_dict())
    optimizer = optim.SGD(adapted.parameters(), lr=0.01)

    for _ in range(10):
        optimizer.zero_grad()
        loss = F.cross_entropy(adapted(sup_x), sup_y)
        loss.backward()
        optimizer.step()

    # Get predictions on attack traces (large batches on GPU)
    X_attack_tensor = torch.FloatTensor(X_attack).to(device)

    with torch.no_grad():
        logits = adapted(X_attack_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    log_probs = np.log(probs + 1e-40)

    # Key recovery
    results = []
    for n in [100, 500, 1000, 5000, 10000]:
        if n > len(log_probs):
            break

        scores = np.zeros(256)
        for k in range(256):
            intermediate = sbox[plaintext_attack[:n] ^ k ^ masks_attack[:n]]
            scores[k] = np.sum(log_probs[np.arange(n), intermediate])

        rank = np.argsort(-scores).tolist().index(correct_key)
        results.append((n, rank))

    return results

# =============================================================================
# Main Experiment
# =============================================================================
print("="*70)
print("GPU-Accelerated Few-Shot SCA Study")
print("="*70)

print("\nLoading ASCAD dataset...")
file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD.h5'

with h5py.File(file_path, 'r') as f:
    X_prof = np.array(f['Profiling_traces/traces'], dtype=np.float32)
    metadata_prof = f['Profiling_traces/metadata']
    X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
    metadata_attack = f['Attack_traces/metadata']

    plaintext_prof = np.array([m['plaintext'][3] for m in metadata_prof])
    masks_prof = np.array([m['masks'][3] for m in metadata_prof])
    key_byte = metadata_prof[0]['key'][3]
    y_prof = sbox[plaintext_prof ^ key_byte ^ masks_prof]

    plaintext_attack = np.array([m['plaintext'][3] for m in metadata_attack])
    masks_attack = np.array([m['masks'][3] for m in metadata_attack])
    correct_key = metadata_attack[0]['key'][3]

print(f"✓ Loaded: {X_prof.shape}, Correct key: {hex(correct_key)}")

# Normalize
mean = X_prof.mean(axis=0)
std = X_prof.std(axis=0) + 1e-8
X_prof = (X_prof - mean) / std
X_attack = (X_attack - mean) / std

split = int(0.8 * len(X_prof))
X_train, y_train = X_prof[:split], y_prof[:split]
X_test, y_test = X_prof[split:], y_prof[split:]

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}\n")

# Run experiments
all_results = []
k_shots = [5, 10, 15, 20]
methods = ['MAML', 'ProtoNet']

for k_shot in k_shots:
    print(f"\n{'='*70}")
    print(f"Running {k_shot}-shot experiments")
    print(f"{'='*70}\n")

    # Prepare test support set
    test_sampler = GPUMinimalVarianceSampler(X_test, y_test)
    sup_x, sup_y = test_sampler.sample_minimal_variance(k_shot)

    # MAML
    print(f"\n[1/2] MAML with {k_shot}-shot")
    model_maml = train_maml_gpu(X_train, y_train, k_shot, epochs=50)
    results_maml = evaluate_attack_gpu(model_maml, sup_x, sup_y, X_attack,
                                       plaintext_attack, masks_attack, correct_key)

    for n_traces, rank in results_maml:
        all_results.append({
            'Method': 'MAML',
            'K-Shot': k_shot,
            'Attack_Traces': n_traces,
            'Key_Rank': rank
        })
        print(f"  {n_traces:5d} traces → rank {rank:3d}")

    # ProtoNet
    print(f"\n[2/2] ProtoNet with {k_shot}-shot")
    model_proto = train_protonet_gpu(X_train, y_train, k_shot, epochs=50)
    results_proto = evaluate_attack_gpu(model_proto, sup_x, sup_y, X_attack,
                                        plaintext_attack, masks_attack, correct_key)

    for n_traces, rank in results_proto:
        all_results.append({
            'Method': 'ProtoNet',
            'K-Shot': k_shot,
            'Attack_Traces': n_traces,
            'Key_Rank': rank
        })
        print(f"  {n_traces:5d} traces → rank {rank:3d}")

# Save results
df = pd.DataFrame(all_results)
df.to_csv('gpu_accelerated_results.csv', index=False)

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print("\nKey Rank at 1000 Attack Traces:")
pivot = df[df['Attack_Traces'] == 1000].pivot(index='K-Shot', columns='Method', values='Key_Rank')
print(pivot)

print("\n✓ Results saved to: gpu_accelerated_results.csv")
print("✓ Complete!\n")
