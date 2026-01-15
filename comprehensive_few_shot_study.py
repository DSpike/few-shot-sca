"""
Comprehensive Few-Shot SCA Study
=================================
Methods: MAML, Prototypical Networks, Siamese Networks
Sampling: Stratified Random Sampling (balanced k-shot per class)
Ablation: 5-shot, 10-shot, 15-shot, 20-shot
GPU-Optimized for maximum utilization
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
import random

# Parse command line arguments for reproducible seeding
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None,
                   help='Random seed for reproducibility (default: None = random)')
args = parser.parse_args()

# Set random seeds if specified
if args.seed is not None:
    print(f"Setting random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # For deterministic behavior (slightly slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

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
# 1. Data Loading with Preprocessed ASCAD (700 points - optimized features)
# =============================================================================
print("Loading ASCAD dataset...")
file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD.h5'

with h5py.File(file_path, 'r') as f:
    X_prof = np.array(f['Profiling_traces/traces'], dtype=np.float32)
    metadata_prof = f['Profiling_traces/metadata']
    X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
    metadata_attack = f['Attack_traces/metadata']

    # Use byte 3
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

# Split
split = int(0.8 * len(X_prof))
X_train, y_train = X_prof[:split], y_prof[:split]
X_test, y_test = X_prof[split:], y_prof[split:]

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}\n")

# =============================================================================
# 2. Stratified Random Sampling for Few-Shot Support Sets
# =============================================================================
class StratifiedSampler:
    """
    Stratified random sampling for few-shot learning
    Ensures balanced representation across all 256 classes
    Each class contributes exactly k_shot samples
    """
    def __init__(self, X, y):
        # Move data to GPU for consistency with training
        self.X_gpu = torch.FloatTensor(X).to(device)
        self.y = y

        # Build class-to-indices mapping
        self.class_idx = defaultdict(list)
        for i, label in enumerate(y):
            self.class_idx[label].append(i)

        # Cache for different k-shot values
        self.cache = {}

    def sample_support_set(self, k_shot):
        """
        Stratified random sampling: k_shot samples per class
        Ensures balanced representation and reproducibility with seeding
        """
        # Return cached result if available
        if k_shot in self.cache:
            return self.cache[k_shot]

        print(f"    Sampling {k_shot}-shot support set (stratified random sampling)...")
        start_time = time.time()
        support_idx = []

        # Sample k_shot examples from each class
        for cls in range(256):
            if len(self.class_idx[cls]) < k_shot:
                # If not enough samples, use all available
                support_idx.extend(self.class_idx[cls])
            else:
                # Stratified random selection
                indices = np.random.choice(self.class_idx[cls], k_shot, replace=False)
                support_idx.extend(indices.tolist())

        # Convert to GPU tensors
        support_x = self.X_gpu[support_idx]
        support_y = torch.LongTensor([self.y[i] for i in support_idx]).to(device)

        # Cache the result
        self.cache[k_shot] = (support_x, support_y)

        elapsed = time.time() - start_time
        print(f"    ✓ Sampled {len(support_idx)} examples ({elapsed:.1f}s)")

        return support_x, support_y

# =============================================================================
# 4. Simple CNN Model
# =============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_size=700):
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

    def forward(self, x, return_features=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        features = self.pool(x).squeeze(-1)

        if return_features:
            return features
        return self.fc(features)

# =============================================================================
# 4. MAML Implementation
# =============================================================================
def maml_adapt(model, support_x, support_y, steps=10, lr=0.01):
    """MAML inner loop adaptation"""
    adapted = SimpleCNN(input_size=X_prof.shape[1]).to(device)
    adapted.load_state_dict(model.state_dict())
    optimizer = optim.SGD(adapted.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(adapted(support_x), support_y)
        loss.backward()
        optimizer.step()

    return adapted

def train_maml(X_train, y_train, k_shot, epochs=100):
    """GPU-optimized MAML training with early stopping"""
    print(f"  Training MAML ({k_shot}-shot)...")
    start_time = time.time()

    model = SimpleCNN(input_size=X_prof.shape[1]).to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sampler = StratifiedSampler(X_train, y_train)

    # Pre-sample support set once (cached on GPU)
    support_x, support_y = sampler.sample_support_set(k_shot)

    # Pre-create query set on GPU
    X_train_gpu = torch.FloatTensor(X_train).to(device)
    y_train_gpu = torch.LongTensor(y_train).to(device)

    # Early stopping
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Random query set (on GPU)
        query_idx = torch.randperm(len(X_train_gpu), device=device)[:1000]
        query_x = X_train_gpu[query_idx]
        query_y = y_train_gpu[query_idx]

        # Meta-update (all on GPU)
        meta_optimizer.zero_grad()
        adapted = maml_adapt(model, support_x, support_y, steps=10)
        loss = F.cross_entropy(adapted(query_x), query_y)
        loss.backward()
        meta_optimizer.step()

        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1} (best loss: {best_loss:.4f})")
                break

        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f} (Time: {elapsed:.1f}s)")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    total_time = time.time() - start_time
    print(f"  ✓ Training complete ({total_time:.1f}s, best loss: {best_loss:.4f})")
    return model

# =============================================================================
# 5. Prototypical Networks
# =============================================================================
def train_protonet(X_train, y_train, k_shot, epochs=100):
    """GPU-optimized Prototypical Network training with early stopping"""
    print(f"  Training ProtoNet ({k_shot}-shot)...")
    start_time = time.time()

    model = SimpleCNN(input_size=X_prof.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sampler = StratifiedSampler(X_train, y_train)

    # Pre-sample support set (cached on GPU)
    support_x, support_y = sampler.sample_support_set(k_shot)

    # Pre-create query set on GPU
    X_train_gpu = torch.FloatTensor(X_train).to(device)
    y_train_gpu = torch.LongTensor(y_train).to(device)

    # Early stopping
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Random query set (on GPU)
        query_idx = torch.randperm(len(X_train_gpu), device=device)[:1000]
        query_x = X_train_gpu[query_idx]
        query_y = y_train_gpu[query_idx]

        # Compute prototypes (all on GPU)
        support_features = model(support_x, return_features=True)
        # Only compute prototypes for classes present in support set
        unique_classes = torch.unique(support_y)
        prototypes = torch.stack([support_features[support_y == c].mean(0)
                                 for c in unique_classes])

        # Query distances (GPU operation)
        query_features = model(query_x, return_features=True)
        dists = torch.cdist(query_features, prototypes)
        logits = -dists

        # Map query labels to prototype indices
        # Only keep queries whose labels exist in support set
        mask = torch.isin(query_y, unique_classes)
        if mask.sum() > 0:
            filtered_logits = logits[mask]
            filtered_query_y = query_y[mask]

            # Create label mapping
            label_map = {c.item(): i for i, c in enumerate(unique_classes)}
            mapped_labels = torch.tensor([label_map[y.item()] for y in filtered_query_y],
                                        dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss = F.cross_entropy(filtered_logits, mapped_labels)
            loss.backward()
            optimizer.step()

            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1} (best loss: {best_loss:.4f})")
                    break
        else:
            # Skip update if no valid queries
            loss = torch.tensor(0.0, device=device)

        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f} (Time: {elapsed:.1f}s)")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    total_time = time.time() - start_time
    print(f"  ✓ Training complete ({total_time:.1f}s, best loss: {best_loss:.4f})")
    return model

# =============================================================================
# 6. Siamese Network
# =============================================================================
def train_siamese(X_train, y_train, k_shot, epochs=100):
    """GPU-optimized Siamese Network training with early stopping"""
    print(f"  Training Siamese Network ({k_shot}-shot)...")
    start_time = time.time()

    model = SimpleCNN(input_size=X_prof.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sampler = StratifiedSampler(X_train, y_train)

    # Pre-sample support set (cached on GPU)
    support_x, support_y = sampler.sample_support_set(k_shot)

    # Pre-create training set on GPU
    X_train_gpu = torch.FloatTensor(X_train).to(device)
    y_train_gpu = torch.LongTensor(y_train).to(device)

    # Early stopping
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Sample pairs: positive (same class) and negative (different class)
        batch_size = 500
        anchor_idx = torch.randint(0, len(support_x), (batch_size,), device=device)
        anchors = support_x[anchor_idx]
        anchor_labels = support_y[anchor_idx]

        # Positive pairs: same class
        positive_idx = []
        for label in anchor_labels:
            same_class_mask = (support_y == label)
            same_class_indices = torch.where(same_class_mask)[0]
            if len(same_class_indices) > 1:
                # Randomly pick another sample from same class
                idx = same_class_indices[torch.randint(0, len(same_class_indices), (1,))]
                positive_idx.append(idx.item())
            else:
                # If only one sample, use the same one
                positive_idx.append(torch.where(same_class_mask)[0][0].item())

        positives = support_x[positive_idx]

        # Negative pairs: different class
        negative_idx = []
        for label in anchor_labels:
            diff_class_mask = (support_y != label)
            diff_class_indices = torch.where(diff_class_mask)[0]
            if len(diff_class_indices) > 0:
                idx = diff_class_indices[torch.randint(0, len(diff_class_indices), (1,))]
                negative_idx.append(idx.item())
            else:
                # Fallback: use random sample
                negative_idx.append(torch.randint(0, len(support_x), (1,)).item())

        negatives = support_x[negative_idx]

        # Compute embeddings
        anchor_features = model(anchors, return_features=True)
        positive_features = model(positives, return_features=True)
        negative_features = model(negatives, return_features=True)

        # Triplet loss with margin
        margin = 1.0
        pos_dist = torch.sum((anchor_features - positive_features)**2, dim=1)
        neg_dist = torch.sum((anchor_features - negative_features)**2, dim=1)
        loss = torch.mean(torch.relu(pos_dist - neg_dist + margin))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1} (best loss: {best_loss:.4f})")
                break

        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f} (Time: {elapsed:.1f}s)")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    total_time = time.time() - start_time
    print(f"  ✓ Training complete ({total_time:.1f}s, best loss: {best_loss:.4f})")
    return model

# =============================================================================
# 7. Attack Evaluation
# =============================================================================
def evaluate_attack(model, support_x, support_y, X_attack, plaintext_attack,
                   masks_attack, correct_key, method='maml', k_shot=10):
    """Evaluate key recovery attack"""

    # Convert to tensors only if not already tensors
    if not isinstance(support_x, torch.Tensor):
        support_x = torch.FloatTensor(support_x).to(device)
        support_y = torch.LongTensor(support_y).to(device)
    # If already tensors, ensure they're on the correct device
    else:
        support_x = support_x.to(device)
        support_y = support_y.to(device)

    if method == 'maml':
        # Adapt model
        adapted_model = maml_adapt(model, support_x, support_y, steps=10)
        adapted_model.eval()
    elif method == 'protonet':
        # Use prototypes
        adapted_model = model
        adapted_model.eval()
    elif method == 'siamese':
        # Siamese network inference
        adapted_model = model
        adapted_model.eval()

    # Get predictions (large batch processing for GPU utilization)
    all_probs = []
    batch_size = 1000  # Increased from 100 for better GPU utilization
    for i in range(0, len(X_attack), batch_size):
        batch = torch.FloatTensor(X_attack[i:i+batch_size]).to(device)
        with torch.no_grad():
            if method == 'protonet' or method == 'siamese':
                # Use prototype/embedding distances
                support_features = adapted_model(support_x, return_features=True)

                # Create prototypes for all 256 classes, using mean of support features
                # For missing classes, use zero vector
                unique_classes = torch.unique(support_y)
                all_prototypes = []

                for c in range(256):
                    mask = (support_y == c)
                    if mask.sum() > 0:
                        all_prototypes.append(support_features[mask].mean(0))
                    else:
                        # Use zero vector for classes not in support set
                        all_prototypes.append(torch.zeros_like(support_features[0]))

                prototypes = torch.stack(all_prototypes)
                query_features = adapted_model(batch, return_features=True)
                dists = torch.cdist(query_features, prototypes)
                logits = -dists
            else:
                logits = adapted_model(batch)

            probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        del batch, logits
        torch.cuda.empty_cache()

    probs = np.vstack(all_probs)
    log_probs = np.log(probs + 1e-40)

    # Key ranking
    traces_list = [100, 500, 1000, 2000, 5000, 10000]
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

    return traces_list[:len(ranks)], ranks

# =============================================================================
# 7. Main Experiment: Ablation Study
# =============================================================================
print("="*70)
print("COMPREHENSIVE FEW-SHOT SCA STUDY")
print("="*70)

k_shots = [5, 10, 15, 20]
methods = ['MAML', 'ProtoNet', 'Siamese']
results = []

for k_shot in k_shots:
    print(f"\n{'='*70}")
    print(f"K-SHOT = {k_shot}")
    print(f"{'='*70}")

    # Prepare support set with stratified random sampling
    sampler = StratifiedSampler(X_test, y_test)
    support_x, support_y = sampler.sample_support_set(k_shot)

    for method_name in methods:
        print(f"\n[{method_name}]")

        # Train model (increased epochs for stability)
        if method_name == 'MAML':
            model = train_maml(X_train, y_train, k_shot, epochs=500)  # MAML needs more epochs
            method_type = 'maml'
        elif method_name == 'ProtoNet':
            model = train_protonet(X_train, y_train, k_shot, epochs=300)
            method_type = 'protonet'
        elif method_name == 'Siamese':
            model = train_siamese(X_train, y_train, k_shot, epochs=300)
            method_type = 'siamese'

        # Evaluate attack
        print(f"  Evaluating attack...")
        traces_list, ranks = evaluate_attack(
            model, support_x, support_y, X_attack,
            plaintext_attack, masks_attack, correct_key,
            method=method_type, k_shot=k_shot
        )

        # Store results
        for n_traces, rank in zip(traces_list, ranks):
            results.append({
                'Method': method_name,
                'K-Shot': k_shot,
                'Attack Traces': n_traces,
                'Key Rank': rank
            })

        print(f"  Results: {dict(zip(traces_list, ranks))}")

# =============================================================================
# 8. Generate Comparison Table
# =============================================================================
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

df = pd.DataFrame(results)

# Pivot table for easy comparison
print("\nKey Rank at 1000 Attack Traces:")
pivot = df[df['Attack Traces'] == 1000].pivot(
    index='K-Shot', columns='Method', values='Key Rank'
)
print(pivot)

# Save results
df.to_csv('few_shot_sca_results.csv', index=False)
print("\n✓ Full results saved to: few_shot_sca_results.csv")

# Summary statistics
print("\n" + "="*70)
print("BEST RESULTS")
print("="*70)
for method in methods:
    method_data = df[df['Method'] == method]
    best_rank = method_data['Key Rank'].min()
    best_config = method_data[method_data['Key Rank'] == best_rank].iloc[0]
    print(f"{method:10s}: Rank {int(best_rank):3d} ({int(best_config['K-Shot'])}-shot, {int(best_config['Attack Traces'])} traces)")

print("\n✓ Comprehensive study complete!")
print("\nNext steps for publication:")
print("  1. Run multiple times (5-10 runs) for confidence intervals")
print("  2. Add baseline comparisons (standard training)")
print("  3. Generate plots from CSV data")
print("  4. Write paper following RESEARCH_GUIDE.md")
