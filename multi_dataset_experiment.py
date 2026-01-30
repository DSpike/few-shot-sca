"""
Multi-Dataset Few-Shot SCA Experiment
======================================
Runs MAML, ProtoNet, Siamese on ASCAD, AES_HD, and CHES_CTF datasets.
Also runs Template Attack and Standard CNN baselines on each dataset.

Usage:
    python multi_dataset_experiment.py --seed 42
    python multi_dataset_experiment.py --seed 42 --dataset ascad
    python multi_dataset_experiment.py --seed 42 --dataset aes_hd
    python multi_dataset_experiment.py --seed 42 --dataset ches_ctf
"""

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
import os

from dataset_loader import load_dataset, key_rank_attack, AES_SBOX, AES_SBOX_INV

# =============================================================================
# Parse Arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='all',
                   choices=['all', 'ascad', 'aes_hd', 'ches_ctf'])
parser.add_argument('--leakage', type=str, default='ID',
                   choices=['ID', 'HW', 'HD'],
                   help='Leakage model: ID (256 classes), HW/HD (9 classes)')
args = parser.parse_args()

# Set seeds
print(f"Setting random seed: {args.seed}")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# =============================================================================
# Adaptive CNN Model (adjusts to trace length and number of classes)
# =============================================================================
class AdaptiveCNN(nn.Module):
    def __init__(self, input_size, n_classes=256):
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
        self.fc = nn.Linear(256, n_classes)

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
# Stratified Sampler
# =============================================================================
class StratifiedSampler:
    def __init__(self, X, y, n_classes):
        self.X_gpu = torch.FloatTensor(X).to(device)
        self.y = y
        self.n_classes = n_classes
        self.class_idx = defaultdict(list)
        for i, label in enumerate(y):
            self.class_idx[label].append(i)
        self.cache = {}

    def sample_support_set(self, k_shot):
        if k_shot in self.cache:
            return self.cache[k_shot]

        support_idx = []
        for cls in range(self.n_classes):
            if len(self.class_idx[cls]) < k_shot:
                support_idx.extend(self.class_idx[cls])
            else:
                indices = np.random.choice(self.class_idx[cls], k_shot, replace=False)
                support_idx.extend(indices.tolist())

        support_x = self.X_gpu[support_idx]
        support_y = torch.LongTensor([self.y[i] for i in support_idx]).to(device)
        self.cache[k_shot] = (support_x, support_y)
        return support_x, support_y

# =============================================================================
# MAML
# =============================================================================
def maml_adapt(model, support_x, support_y, input_size, n_classes, steps=10, lr=0.01):
    adapted = AdaptiveCNN(input_size=input_size, n_classes=n_classes).to(device)
    adapted.load_state_dict(model.state_dict())
    optimizer = optim.SGD(adapted.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(adapted(support_x), support_y)
        loss.backward()
        optimizer.step()
    return adapted

def train_maml(X_train, y_train, k_shot, input_size, n_classes, epochs=500):
    print(f"  Training MAML ({k_shot}-shot)...")
    start_time = time.time()
    model = AdaptiveCNN(input_size=input_size, n_classes=n_classes).to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sampler = StratifiedSampler(X_train, y_train, n_classes)
    support_x, support_y = sampler.sample_support_set(k_shot)
    X_train_gpu = torch.FloatTensor(X_train).to(device)
    y_train_gpu = torch.LongTensor(y_train).to(device)

    best_loss = float('inf')
    patience, patience_counter = 50, 0
    best_state = None

    for epoch in range(epochs):
        query_idx = torch.randperm(len(X_train_gpu), device=device)[:1000]
        query_x, query_y = X_train_gpu[query_idx], y_train_gpu[query_idx]
        meta_optimizer.zero_grad()
        adapted = maml_adapt(model, support_x, support_y, input_size, n_classes, steps=10)
        loss = F.cross_entropy(adapted(query_x), query_y)
        loss.backward()
        meta_optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  MAML done ({time.time()-start_time:.1f}s)")
    return model

# =============================================================================
# Prototypical Networks
# =============================================================================
def train_protonet(X_train, y_train, k_shot, input_size, n_classes, epochs=300):
    print(f"  Training ProtoNet ({k_shot}-shot)...")
    start_time = time.time()
    model = AdaptiveCNN(input_size=input_size, n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sampler = StratifiedSampler(X_train, y_train, n_classes)
    support_x, support_y = sampler.sample_support_set(k_shot)
    X_train_gpu = torch.FloatTensor(X_train).to(device)
    y_train_gpu = torch.LongTensor(y_train).to(device)

    best_loss = float('inf')
    patience, patience_counter = 50, 0
    best_state = None

    for epoch in range(epochs):
        query_idx = torch.randperm(len(X_train_gpu), device=device)[:1000]
        query_x, query_y = X_train_gpu[query_idx], y_train_gpu[query_idx]
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
            loss = F.cross_entropy(filtered_logits, mapped_labels)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  ProtoNet done ({time.time()-start_time:.1f}s)")
    return model

# =============================================================================
# Siamese Network
# =============================================================================
def train_siamese(X_train, y_train, k_shot, input_size, n_classes, epochs=300):
    print(f"  Training Siamese ({k_shot}-shot)...")
    start_time = time.time()
    model = AdaptiveCNN(input_size=input_size, n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sampler = StratifiedSampler(X_train, y_train, n_classes)
    support_x, support_y = sampler.sample_support_set(k_shot)

    best_loss = float('inf')
    patience, patience_counter = 50, 0
    best_state = None

    for epoch in range(epochs):
        batch_size = 500
        anchor_idx = torch.randint(0, len(support_x), (batch_size,), device=device)
        anchors = support_x[anchor_idx]
        anchor_labels = support_y[anchor_idx]

        positive_idx = []
        for label in anchor_labels:
            same_mask = (support_y == label)
            same_indices = torch.where(same_mask)[0]
            idx = same_indices[torch.randint(0, len(same_indices), (1,))]
            positive_idx.append(idx.item())
        positives = support_x[positive_idx]

        negative_idx = []
        for label in anchor_labels:
            diff_mask = (support_y != label)
            diff_indices = torch.where(diff_mask)[0]
            if len(diff_indices) > 0:
                idx = diff_indices[torch.randint(0, len(diff_indices), (1,))]
                negative_idx.append(idx.item())
            else:
                negative_idx.append(torch.randint(0, len(support_x), (1,)).item())
        negatives = support_x[negative_idx]

        anchor_feat = model(anchors, return_features=True)
        pos_feat = model(positives, return_features=True)
        neg_feat = model(negatives, return_features=True)

        pos_dist = torch.sum((anchor_feat - pos_feat)**2, dim=1)
        neg_dist = torch.sum((anchor_feat - neg_feat)**2, dim=1)
        loss = torch.mean(torch.relu(pos_dist - neg_dist + 1.0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  Siamese done ({time.time()-start_time:.1f}s)")
    return model

# =============================================================================
# Standard CNN Baseline
# =============================================================================
def train_standard_cnn(X_train, y_train, input_size, n_classes, epochs=100, batch_size=128):
    print(f"  Training Standard CNN ({len(X_train)} traces)...")
    start_time = time.time()
    model = AdaptiveCNN(input_size=input_size, n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    patience, patience_counter = 20, 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  Standard CNN done ({time.time()-start_time:.1f}s)")
    return model

# =============================================================================
# Template Attack Baseline
# =============================================================================
def run_template_attack(X_train, y_train, X_attack, dataset_info, n_components=20):
    from sklearn.decomposition import PCA
    from scipy.stats import multivariate_normal

    n_classes = dataset_info['n_classes']
    print(f"  Running Template Attack ({len(X_train)} traces, {n_components} PCA dims)...")
    start_time = time.time()

    # Adjust components
    n_comp = min(n_components, len(X_train) // max(n_classes, 1) - 1, X_train.shape[1])
    n_comp = max(n_comp, 5)

    # PCA
    pca = PCA(n_components=n_comp)
    X_reduced = pca.fit_transform(X_train)

    # Build templates
    class_data = {}
    templates = {}
    for cls in range(n_classes):
        mask = (y_train == cls)
        n_cls = mask.sum()
        if n_cls >= 2:
            X_cls = X_reduced[mask]
            class_data[cls] = X_cls
            templates[cls] = {'mean': X_cls.mean(axis=0)}
        elif n_cls == 1:
            X_cls = X_reduced[mask]
            class_data[cls] = X_cls
            templates[cls] = {'mean': X_cls[0]}
        else:
            templates[cls] = {'mean': X_reduced.mean(axis=0)}

    # Pooled covariance
    residuals = []
    for cls, X_cls in class_data.items():
        if len(X_cls) >= 2:
            residuals.append(X_cls - templates[cls]['mean'])
    if residuals:
        all_res = np.vstack(residuals)
        pooled_cov = np.cov(all_res, rowvar=False) + np.eye(n_comp) * 1e-6
    else:
        pooled_cov = np.eye(n_comp)

    # Compute log-probs on attack traces
    X_attack_reduced = pca.transform(X_attack)
    log_probs = np.zeros((len(X_attack_reduced), n_classes))
    for cls in range(n_classes):
        try:
            rv = multivariate_normal(mean=templates[cls]['mean'], cov=pooled_cov,
                                    allow_singular=True)
            log_probs[:, cls] = rv.logpdf(X_attack_reduced)
        except Exception:
            diff = X_attack_reduced - templates[cls]['mean']
            log_probs[:, cls] = -0.5 * np.sum(diff ** 2, axis=1)

    print(f"  Template Attack done ({time.time()-start_time:.1f}s)")
    return log_probs

# =============================================================================
# Attack Evaluation (unified across all methods)
# =============================================================================
def evaluate_model_attack(model, support_x, support_y, X_attack, dataset_info,
                          method='maml', input_size=700, n_classes=256):
    """Get log-probabilities from a trained meta-learning model."""
    if not isinstance(support_x, torch.Tensor):
        support_x = torch.FloatTensor(support_x).to(device)
        support_y = torch.LongTensor(support_y).to(device)

    if method == 'maml':
        adapted = maml_adapt(model, support_x, support_y, input_size, n_classes, steps=10)
        adapted.eval()
    else:
        adapted = model
        adapted.eval()

    all_probs = []
    batch_size = 1000
    for i in range(0, len(X_attack), batch_size):
        batch = torch.FloatTensor(X_attack[i:i+batch_size]).to(device)
        with torch.no_grad():
            if method in ('protonet', 'siamese'):
                support_features = adapted(support_x, return_features=True)
                all_prototypes = []
                for c in range(n_classes):
                    mask = (support_y == c)
                    if mask.sum() > 0:
                        all_prototypes.append(support_features[mask].mean(0))
                    else:
                        all_prototypes.append(torch.zeros_like(support_features[0]))
                prototypes = torch.stack(all_prototypes)
                query_features = adapted(batch, return_features=True)
                logits = -torch.cdist(query_features, prototypes)
            else:
                logits = adapted(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        del batch, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probs = np.vstack(all_probs)
    log_probs = np.log(probs + 1e-40)
    return log_probs


def compute_key_ranks(log_probs, dataset_info, traces_list=None):
    """Compute key ranks at various trace counts."""
    if traces_list is None:
        traces_list = [100, 500, 1000, 2000, 5000, 10000]

    ranks = []
    for n in traces_list:
        if n > len(log_probs):
            break
        rank = key_rank_attack(log_probs, dataset_info, n)
        ranks.append(rank)

    return traces_list[:len(ranks)], ranks

# =============================================================================
# Main Experiment
# =============================================================================
def run_experiment_on_dataset(ds_name, leakage_model='ID'):
    """Run all methods on a single dataset."""
    print(f"\n{'='*70}")
    print(f"DATASET: {ds_name.upper()} (leakage={leakage_model})")
    print(f"{'='*70}")

    data = load_dataset(ds_name, leakage_model=leakage_model)
    X_prof = data['X_prof']
    y_prof = data['y_prof']
    X_attack = data['X_attack']
    n_classes = data['n_classes']
    input_size = data['trace_length']

    # Train/test split
    split = int(0.8 * len(X_prof))
    X_train, y_train = X_prof[:split], y_prof[:split]
    X_test, y_test = X_prof[split:], y_prof[split:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Classes: {n_classes}")

    k_shots = [5, 10, 15, 20]
    meta_methods = ['MAML', 'ProtoNet', 'Siamese']
    results = []

    # --- Few-Shot Meta-Learning Methods ---
    for k_shot in k_shots:
        print(f"\n--- K-SHOT = {k_shot} ---")
        sampler = StratifiedSampler(X_test, y_test, n_classes)
        support_x, support_y = sampler.sample_support_set(k_shot)

        for method_name in meta_methods:
            print(f"\n[{method_name}]")
            if method_name == 'MAML':
                model = train_maml(X_train, y_train, k_shot, input_size, n_classes)
                method_type = 'maml'
            elif method_name == 'ProtoNet':
                model = train_protonet(X_train, y_train, k_shot, input_size, n_classes)
                method_type = 'protonet'
            elif method_name == 'Siamese':
                model = train_siamese(X_train, y_train, k_shot, input_size, n_classes)
                method_type = 'siamese'

            log_probs = evaluate_model_attack(model, support_x, support_y, X_attack,
                                             data, method_type, input_size, n_classes)
            traces_list, ranks = compute_key_ranks(log_probs, data)

            for n_traces, rank in zip(traces_list, ranks):
                results.append({
                    'Dataset': ds_name.upper(),
                    'Method': method_name,
                    'K-Shot': k_shot,
                    'Training_Size': k_shot * n_classes,
                    'Attack Traces': n_traces,
                    'Key Rank': rank,
                    'Leakage_Model': leakage_model,
                    'N_Classes': n_classes
                })
            print(f"  Results: {dict(zip(traces_list, ranks))}")

    # --- Template Attack Baseline ---
    print(f"\n--- TEMPLATE ATTACK BASELINE ---")
    for k_shot in k_shots:
        n_samples = k_shot * n_classes
        indices = []
        samples_per_class = k_shot
        for cls in range(n_classes):
            class_indices = np.where(y_prof == cls)[0]
            if len(class_indices) >= samples_per_class:
                selected = np.random.choice(class_indices, samples_per_class, replace=False)
                indices.extend(selected)
        indices = np.array(indices)
        X_sub, y_sub = X_prof[indices], y_prof[indices]

        log_probs = run_template_attack(X_sub, y_sub, X_attack, data)
        traces_list, ranks = compute_key_ranks(log_probs, data)

        for n_traces, rank in zip(traces_list, ranks):
            results.append({
                'Dataset': ds_name.upper(),
                'Method': 'Template Attack',
                'K-Shot': k_shot,
                'Training_Size': n_samples,
                'Attack Traces': n_traces,
                'Key Rank': rank,
                'Leakage_Model': leakage_model,
                'N_Classes': n_classes
            })
        print(f"  TA {k_shot}-shot: {dict(zip(traces_list, ranks))}")

    # --- Standard CNN Baseline ---
    print(f"\n--- STANDARD CNN BASELINE ---")
    for k_shot in k_shots:
        n_samples = k_shot * n_classes
        indices = []
        samples_per_class = k_shot
        for cls in range(n_classes):
            class_indices = np.where(y_prof == cls)[0]
            if len(class_indices) >= samples_per_class:
                selected = np.random.choice(class_indices, samples_per_class, replace=False)
                indices.extend(selected)
        indices = np.array(indices)
        X_sub, y_sub = X_prof[indices], y_prof[indices]

        model = train_standard_cnn(X_sub, y_sub, input_size, n_classes)
        model.eval()

        all_probs = []
        for i in range(0, len(X_attack), 1000):
            batch = torch.FloatTensor(X_attack[i:i+1000]).to(device)
            with torch.no_grad():
                probs = F.softmax(model(batch), dim=1).cpu().numpy()
            all_probs.append(probs)
        log_probs = np.log(np.vstack(all_probs) + 1e-40)
        traces_list, ranks = compute_key_ranks(log_probs, data)

        for n_traces, rank in zip(traces_list, ranks):
            results.append({
                'Dataset': ds_name.upper(),
                'Method': 'Standard CNN',
                'K-Shot': k_shot,
                'Training_Size': n_samples,
                'Attack Traces': n_traces,
                'Key Rank': rank,
                'Leakage_Model': leakage_model,
                'N_Classes': n_classes
            })
        print(f"  CNN {k_shot}-shot: {dict(zip(traces_list, ranks))}")

    return results

# =============================================================================
# Run All Datasets
# =============================================================================
print("=" * 70)
print("MULTI-DATASET FEW-SHOT SCA EXPERIMENT")
print("=" * 70)

all_results = []
output_dir = 'multi_dataset_results'
os.makedirs(output_dir, exist_ok=True)

if args.dataset == 'all':
    datasets_to_run = ['ascad', 'aes_hd', 'ches_ctf']
else:
    datasets_to_run = [args.dataset]

for ds_name in datasets_to_run:
    try:
        results = run_experiment_on_dataset(ds_name, leakage_model=args.leakage)
        all_results.extend(results)

        # Save per-dataset results
        ds_df = pd.DataFrame(results)
        ds_df.to_csv(os.path.join(output_dir, f'{ds_name}_results.csv'), index=False)
        print(f"\nSaved: {output_dir}/{ds_name}_results.csv")

    except FileNotFoundError as e:
        print(f"\nWARNING: Dataset {ds_name} not found: {e}")
        print(f"  Run: python download_datasets.py --dataset {ds_name}")
        continue

# Save combined results
if all_results:
    combined_df = pd.DataFrame(all_results)
    combined_df.to_csv(os.path.join(output_dir, 'all_datasets_combined.csv'), index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-DATASET RESULTS SUMMARY (Key Rank @ 1000 traces)")
    print("=" * 70)

    summary = combined_df[combined_df['Attack Traces'] == 1000]
    for ds in summary['Dataset'].unique():
        print(f"\n  {ds}:")
        ds_data = summary[summary['Dataset'] == ds]
        for method in ds_data['Method'].unique():
            method_data = ds_data[ds_data['Method'] == method]
            for _, row in method_data.iterrows():
                print(f"    {method:15s} {int(row['K-Shot']):2d}-shot: Key Rank = {int(row['Key Rank'])}")

    print(f"\nAll results saved to: {output_dir}/all_datasets_combined.csv")

print("\nMulti-dataset experiment complete!")
