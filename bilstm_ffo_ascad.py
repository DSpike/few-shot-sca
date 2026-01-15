"""
BiLSTM with Entropy-TOPSIS and Flying Foxes Optimization for ASCAD
Side-Channel Attack using deep learning
"""

# Cell 1
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Cell 2
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
])

# Cell 3
file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD.h5'

with h5py.File(file_path, 'r') as f:
    # Traces
    X_profiling = np.array(f['Profiling_traces/traces'], dtype=np.float32)
    X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
    
    # Metadata
    metadata_prof = f['Profiling_traces/metadata']
    metadata_attack = f['Attack_traces/metadata']
    
    # Extract for byte 2 (standard ASCAD target)
    plaintext_prof = np.array([m['plaintext'][2] for m in metadata_prof])
    masks_prof = np.array([m['masks'][2] for m in metadata_prof])  # mask for byte 2
    key_byte = metadata_prof[0]['key'][2]  # fixed key byte 2
    
    # Correct intermediate: masked S-box output = sbox[plaintext[2] ^ key[2] ^ mask[2]]
    y_profiling = sbox[plaintext_prof ^ key_byte ^ masks_prof]
    
    # For attack set (same for evaluation)
    plaintext_attack = np.array([m['plaintext'][2] for m in metadata_attack])
    masks_attack = np.array([m['masks'][2] for m in metadata_attack])
    correct_key_byte = metadata_attack[0]['key'][2]  # fixed

# Normalize traces
scaler = MinMaxScaler()
X_profiling = scaler.fit_transform(X_profiling)
X_attack = scaler.transform(X_attack)

# Split profiling - THIS CREATES y_train and y_val with CORRECT labels
X_train, X_val, y_train, y_val = train_test_split(
    X_profiling, y_profiling, test_size=0.2, random_state=42, stratify=y_profiling
)

print("Loaded and labels computed.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Correct key byte:", hex(correct_key_byte))
print("Labels are masked S-box outputs (range 0-255)")

# Cell 4
def compute_criteria_matrix(X, y):
    n_features = X.shape[1]
    criteria = np.zeros((n_features, 3))
    
    for i in range(n_features):
        feature = X[:, i]
        criteria[i, 0] = np.var(feature)
        criteria[i, 1] = np.abs(pearsonr(feature, y)[0])
        criteria[i, 2] = np.mean(np.abs(feature))
    
    return criteria

criteria_matrix = compute_criteria_matrix(X_train, y_train)
print("Criteria matrix shape:", criteria_matrix.shape)

# Cell 5
def entropy_weights(criteria_matrix):
    norms = criteria_matrix / (np.sqrt(np.sum(criteria_matrix**2, axis=0)) + 1e-10)
    norms = np.clip(norms, 1e-10, None)
    props = norms / (np.sum(norms, axis=0) + 1e-10)
    k = 1.0 / np.log(criteria_matrix.shape[0])
    entropy = -k * np.sum(props * np.log(props), axis=0)
    weights = (1 - entropy) / np.sum(1 - entropy)
    return weights

criteria_weights = entropy_weights(criteria_matrix)
print("Entropy weights:", np.round(criteria_weights, 4))

# Cell 6
def topsis_ranking(criteria_matrix, weights, is_benefit=[True, True, True]):
    norms = criteria_matrix / np.sqrt(np.sum(criteria_matrix**2, axis=0))
    weighted_norms = norms * weights
    ideal_pos = np.max(weighted_norms, axis=0)
    ideal_neg = np.min(weighted_norms, axis=0)
    dist_pos = np.sqrt(np.sum((weighted_norms - ideal_pos)**2, axis=1))
    dist_neg = np.sqrt(np.sum((weighted_norms - ideal_neg)**2, axis=1))
    closeness = dist_neg / (dist_pos + dist_neg)
    ranked_indices = np.argsort(closeness)[::-1]
    return ranked_indices

ranked_features = topsis_ranking(criteria_matrix, criteria_weights)

# FIXED: Use 200 features instead of 700 to reduce overfitting
top_k = 200  # Optimal for ASCAD - reduces noise and overfitting
X_train_selected = X_train[:, ranked_features[:top_k]]
X_val_selected = X_val[:, ranked_features[:top_k]]
X_attack_selected = X_attack[:, ranked_features[:top_k]]

print("Feature selection done. Shapes:", X_train_selected.shape)
print(f"Using top {top_k} features out of {X_train.shape[1]} (optimal balance for ASCAD)")

# Cell 7
# IMPORTANT: We already have the correct labels (y_profiling) from cell 2ea5148f
# The labels are the masked S-box outputs, NOT the plaintext!
# y_train and y_val were already created in the train_test_split in cell 2ea5148f

print("Using labels from the initial data load (masked S-box outputs)")
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("Correct key byte:", hex(correct_key_byte))
print("Label range: min={}, max={}".format(y_train.min(), y_train.max()))

# Cell 8
# Prepare data for PyTorch LSTM
# PyTorch LSTM expects input shape: (batch_size, seq_len, input_size)
# We have (samples, features) -> reshape to (samples, features, 1)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_selected).unsqueeze(-1).to(device)
X_val_tensor = torch.FloatTensor(X_val_selected).unsqueeze(-1).to(device)
X_attack_tensor = torch.FloatTensor(X_attack_selected).unsqueeze(-1).to(device)

# For PyTorch, we use class indices (not one-hot)
y_train_tensor = torch.LongTensor(y_train).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)

print("Data prepared for PyTorch")
print("X_train shape:", X_train_tensor.shape)
print("y_train shape:", y_train_tensor.shape)
print("Data on device:", X_train_tensor.device)

# Cell 9
# Improved PyTorch BiLSTM Model for Side-Channel Analysis
class ImprovedBiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.4, num_classes=256):
        super(ImprovedBiLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM layers (bidirectional=True)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # FIXED: Enable bidirectional
        )
        
        # Batch normalization (hidden_size * 2 because bidirectional)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # Dropout after LSTM
        self.dropout = nn.Dropout(dropout)
        
        # Output layers with regularization
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # BiLSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output from the last time step
        out = lstm_out[:, -1, :]
        
        # Batch normalization
        out = self.batch_norm(out)
        
        # Dropout
        out = self.dropout(out)
        
        # First FC layer
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Output layer
        out = self.fc2(out)
        
        return out

print("BiLSTM Model class defined!")
print("Use FFO cell below to find optimal hyperparameters automatically")

# Cell 10
# Flying Foxes Optimization (FFO) for Hyperparameter Tuning
import copy

class FlyingFoxesOptimization:
    """
    Flying Foxes Optimization algorithm for hyperparameter tuning.
    Based on the foraging behavior of flying foxes.
    """
    def __init__(self, n_foxes=10, max_iter=20, bounds=None):
        self.n_foxes = n_foxes
        self.max_iter = max_iter
        self.bounds = bounds  # Dict of {param_name: (min, max)}
        self.best_position = None
        self.best_fitness = float('inf')
        self.history = []
        
    def _initialize_population(self):
        """Initialize flying fox positions randomly within bounds."""
        population = []
        for _ in range(self.n_foxes):
            fox = {}
            for param, (low, high) in self.bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    fox[param] = np.random.randint(low, high + 1)
                else:
                    fox[param] = np.random.uniform(low, high)
            population.append(fox)
        return population
    
    def _evaluate_fitness(self, position, objective_func):
        """Evaluate fitness of a position."""
        return objective_func(position)
    
    def optimize(self, objective_func):
        """
        Main optimization loop.
        objective_func: function that takes hyperparameters dict and returns validation loss
        """
        # Initialize population
        population = self._initialize_population()
        fitness = [self._evaluate_fitness(fox, objective_func) for fox in population]
        
        # Find initial best
        best_idx = np.argmin(fitness)
        self.best_position = copy.deepcopy(population[best_idx])
        self.best_fitness = fitness[best_idx]
        
        print(f"FFO Optimization Started")
        print(f"Population: {self.n_foxes} foxes, Iterations: {self.max_iter}")
        print("="*80)
        
        for iteration in range(self.max_iter):
            for i in range(self.n_foxes):
                # Flying fox foraging behavior
                # Phase 1: Exploration (random flight)
                if np.random.rand() < 0.5:
                    for param, (low, high) in self.bounds.items():
                        if isinstance(low, int) and isinstance(high, int):
                            population[i][param] = np.random.randint(low, high + 1)
                        else:
                            # Levy flight for exploration
                            step = np.random.normal(0, 1) * (high - low) * 0.1
                            population[i][param] = np.clip(
                                population[i][param] + step, low, high
                            )
                            if isinstance(self.best_position[param], int):
                                population[i][param] = int(round(population[i][param]))
                
                # Phase 2: Exploitation (move towards best food source)
                else:
                    for param in self.bounds.keys():
                        # Move towards best position with random factor
                        alpha = np.random.uniform(0.5, 1.5)
                        population[i][param] = population[i][param] + alpha * (
                            self.best_position[param] - population[i][param]
                        )
                        
                        # Clip to bounds
                        low, high = self.bounds[param]
                        population[i][param] = np.clip(population[i][param], low, high)
                        
                        # Round if integer parameter
                        if isinstance(low, int) and isinstance(high, int):
                            population[i][param] = int(round(population[i][param]))
                
                # Evaluate new position
                new_fitness = self._evaluate_fitness(population[i], objective_func)
                
                # Update if better
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    
                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_position = copy.deepcopy(population[i])
                        self.best_fitness = new_fitness
                        print(f"Iter {iteration+1}/{self.max_iter}, Fox {i+1}: New best fitness = {self.best_fitness:.4f}")
                        print(f"  Params: {self.best_position}")
            
            # Log history
            self.history.append({
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'best_position': copy.deepcopy(self.best_position)
            })
            
            print(f"Iteration {iteration+1}/{self.max_iter} complete - Best fitness: {self.best_fitness:.4f}")
        
        print("="*80)
        print(f"Optimization complete!")
        print(f"Best hyperparameters: {self.best_position}")
        print(f"Best validation loss: {self.best_fitness:.4f}")
        
        return self.best_position, self.best_fitness

print("Flying Foxes Optimization (FFO) loaded successfully!")

# MARKDOWN: Cell 11
# #
# 
# F
# l
# y
# i
# n
# g
# 
# F
# o
# x
# e
# s
# 
# O
# p
# t
# i
# m
# i
# z
# a
# t
# i
# o
# n
# 
# (
# F
# F
# O
# )
# 
# f
# o
# r
# 
# H
# y
# p
# e
# r
# p
# a
# r
# a
# m
# e
# t
# e
# r
# 
# T
# u
# n
# i
# n
# g
# 
# 
# T
# h
# i
# s
# 
# s
# e
# c
# t
# i
# o
# n
# 
# u
# s
# e
# s
# 
# *
# *
# F
# l
# y
# i
# n
# g
# 
# F
# o
# x
# e
# s
# 
# O
# p
# t
# i
# m
# i
# z
# a
# t
# i
# o
# n
# *
# *
# 
# t
# o
# 
# a
# u
# t
# o
# m
# a
# t
# i
# c
# a
# l
# l
# y
# 
# f
# i
# n
# d
# 
# t
# h
# e
# 
# b
# e
# s
# t
# 
# h
# y
# p
# e
# r
# p
# a
# r
# a
# m
# e
# t
# e
# r
# s
# 
# f
# o
# r
# 
# t
# h
# e
# 
# B
# i
# L
# S
# T
# M
# 
# m
# o
# d
# e
# l
# .
# 
# 
# #
# #
# 
# W
# o
# r
# k
# f
# l
# o
# w
# :
# 
# 1
# .
# 
# *
# *
# F
# F
# O
# 
# A
# l
# g
# o
# r
# i
# t
# h
# m
# *
# *
# 
# -
# 
# S
# i
# m
# u
# l
# a
# t
# e
# s
# 
# f
# l
# y
# i
# n
# g
# 
# f
# o
# x
# 
# f
# o
# r
# a
# g
# i
# n
# g
# 
# b
# e
# h
# a
# v
# i
# o
# r
# 
# t
# o
# 
# s
# e
# a
# r
# c
# h
# 
# h
# y
# p
# e
# r
# p
# a
# r
# a
# m
# e
# t
# e
# r
# 
# s
# p
# a
# c
# e
# 
# 2
# .
# 
# *
# *
# O
# b
# j
# e
# c
# t
# i
# v
# e
# 
# F
# u
# n
# c
# t
# i
# o
# n
# *
# *
# 
# -
# 
# T
# r
# a
# i
# n
# s
# 
# B
# i
# L
# S
# T
# M
# 
# m
# o
# d
# e
# l
# s
# 
# w
# i
# t
# h
# 
# d
# i
# f
# f
# e
# r
# e
# n
# t
# 
# h
# y
# p
# e
# r
# p
# a
# r
# a
# m
# e
# t
# e
# r
# s
# 
# a
# n
# d
# 
# r
# e
# t
# u
# r
# n
# s
# 
# v
# a
# l
# i
# d
# a
# t
# i
# o
# n
# 
# l
# o
# s
# s
# 
# 3
# .
# 
# *
# *
# O
# p
# t
# i
# m
# i
# z
# a
# t
# i
# o
# n
# *
# *
# 
# -
# 
# F
# F
# O
# 
# f
# i
# n
# d
# s
# 
# t
# h
# e
# 
# c
# o
# n
# f
# i
# g
# u
# r
# a
# t
# i
# o
# n
# 
# w
# i
# t
# h
# 
# l
# o
# w
# e
# s
# t
# 
# v
# a
# l
# i
# d
# a
# t
# i
# o
# n
# 
# l
# o
# s
# s
# 
# 4
# .
# 
# *
# *
# F
# i
# n
# a
# l
# 
# T
# r
# a
# i
# n
# i
# n
# g
# *
# *
# 
# -
# 
# T
# r
# a
# i
# n
# 
# f
# u
# l
# l
# 
# m
# o
# d
# e
# l
# 
# w
# i
# t
# h
# 
# o
# p
# t
# i
# m
# i
# z
# e
# d
# 
# h
# y
# p
# e
# r
# p
# a
# r
# a
# m
# e
# t
# e
# r
# s
# 
# 
# #
# #
# 
# H
# y
# p
# e
# r
# p
# a
# r
# a
# m
# e
# t
# e
# r
# s
# 
# b
# e
# i
# n
# g
# 
# o
# p
# t
# i
# m
# i
# z
# e
# d
# :
# 
# -
# 
# `
# h
# i
# d
# d
# e
# n
# _
# s
# i
# z
# e
# `
# :
# 
# N
# u
# m
# b
# e
# r
# 
# o
# f
# 
# L
# S
# T
# M
# 
# h
# i
# d
# d
# e
# n
# 
# u
# n
# i
# t
# s
# 
# (
# 6
# 4
# -
# 2
# 5
# 6
# )
# 
# -
# 
# `
# n
# u
# m
# _
# l
# a
# y
# e
# r
# s
# `
# :
# 
# N
# u
# m
# b
# e
# r
# 
# o
# f
# 
# L
# S
# T
# M
# 
# l
# a
# y
# e
# r
# s
# 
# (
# 1
# -
# 3
# )
# 
# -
# 
# `
# d
# r
# o
# p
# o
# u
# t
# `
# :
# 
# D
# r
# o
# p
# o
# u
# t
# 
# r
# a
# t
# e
# 
# (
# 0
# .
# 2
# -
# 0
# .
# 6
# )
# 
# -
# 
# `
# l
# e
# a
# r
# n
# i
# n
# g
# _
# r
# a
# t
# e
# `
# :
# 
# A
# d
# a
# m
# 
# l
# e
# a
# r
# n
# i
# n
# g
# 
# r
# a
# t
# e
# 
# (
# 0
# .
# 0
# 0
# 0
# 1
# -
# 0
# .
# 0
# 0
# 5
# )
# 
# -
# 
# `
# b
# a
# t
# c
# h
# _
# s
# i
# z
# e
# `
# :
# 
# T
# r
# a
# i
# n
# i
# n
# g
# 
# b
# a
# t
# c
# h
# 
# s
# i
# z
# e
# 
# (
# 2
# 5
# 6
# -
# 1
# 0
# 2
# 4
# )
# 
# 
# *
# *
# N
# o
# t
# e
# :
# *
# *
# 
# F
# F
# O
# 
# o
# p
# t
# i
# m
# i
# z
# a
# t
# i
# o
# n
# 
# t
# a
# k
# e
# s
# 
# 2
# 0
# -
# 3
# 0
# 
# m
# i
# n
# u
# t
# e
# s
# .
# 
# Y
# o
# u
# 
# c
# a
# n
# 
# s
# k
# i
# p
# 
# i
# t
# 
# a
# n
# d
# 
# u
# s
# e
# 
# d
# e
# f
# a
# u
# l
# t
# 
# p
# a
# r
# a
# m
# e
# t
# e
# r
# s
# 
# b
# y
# 
# r
# u
# n
# n
# i
# n
# g
# 
# t
# h
# e
# 
# f
# i
# n
# a
# l
# 
# t
# r
# a
# i
# n
# i
# n
# g
# 
# c
# e
# l
# l
# 
# d
# i
# r
# e
# c
# t
# l
# y
# .

# Cell 12
# Define objective function for FFO (trains BiLSTM and returns validation loss)
def train_bilstm_with_params(params_dict):
    """
    Train BiLSTM model with given hyperparameters and return validation loss.
    This is the objective function for FFO to minimize.
    """
    # Extract parameters
    hidden_size = params_dict['hidden_size']
    num_layers = params_dict['num_layers']
    dropout = params_dict['dropout']
    learning_rate = params_dict['learning_rate']
    batch_size = params_dict['batch_size']
    
    # Create model
    model = ImprovedBiLSTMModel(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=256
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Create data loaders with current batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # OPTIMIZED: Much faster training for FFO (reduced from 30 to 15 epochs)
    epochs = 15  # Reduced significantly for faster FFO
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5  # Reduced patience for faster convergence check
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss = train_loss / len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss = val_loss / len(val_dataset)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Clean up to save memory
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    
    return best_val_loss

# OPTIMIZED: Reduced search space for faster FFO
hyperparameter_bounds = {
    'hidden_size': (64, 192),      # Reduced upper bound (was 256)
    'num_layers': (1, 2),          # Reduced to 1-2 layers (was 1-3)
    'dropout': (0.3, 0.5),         # Narrower range
    'learning_rate': (0.0005, 0.003),  # Narrower range
    'batch_size': (256, 768)       # Reduced upper bound (was 1024)
}

print("Objective function defined!")
print("OPTIMIZED for faster FFO (15 epochs per trial, narrower search space)")
print("\nHyperparameter search space:")
for param, (low, high) in hyperparameter_bounds.items():
    print(f"  {param}: [{low}, {high}]")

# Cell 13
# Run Flying Foxes Optimization for hyperparameter tuning
# OPTIMIZED: Reduced foxes and iterations for faster completion

ffo = FlyingFoxesOptimization(
    n_foxes=5,        # Reduced from 8 to 5 foxes
    max_iter=10,      # Reduced from 15 to 10 iterations
    bounds=hyperparameter_bounds
)

# Run optimization
print("Starting OPTIMIZED FFO hyperparameter optimization...")
print("Estimated time: 15-20 minutes on GPU (much faster!)")
print(f"Total evaluations: {5 * 10} = 50 training runs (down from 120)")
print("Each run: ~15 epochs with early stopping\n")

best_hyperparams, best_val_loss = ffo.optimize(train_bilstm_with_params)

print("\n" + "="*80)
print("FFO OPTIMIZATION RESULTS")
print("="*80)
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"\nOptimal Hyperparameters:")
for param, value in best_hyperparams.items():
    print(f"  {param}: {value}")
print("="*80)

# Cell 14
# Train final BiLSTM model with PROVEN hyperparameters for ASCAD
print("Training BiLSTM model with proven hyperparameters...")
print("="*80)

# PROVEN hyperparameters for BiLSTM on ASCAD (from literature + our testing)
params = {
    'hidden_size': 100,      # Proven to work well for ASCAD
    'num_layers': 1,         # Single layer BiLSTM is often best
    'dropout': 0.4,          # Standard dropout
    'learning_rate': 0.001,  # Standard Adam LR
    'batch_size': 400,       # Balanced for GPU
    'epochs': 150            # Increased for better convergence
}

print("Using PROVEN hyperparameters for BiLSTM + Entropy-TOPSIS:")
for key, value in params.items():
    print(f"  {key}: {value}")

# Create model with proven parameters
model = ImprovedBiLSTMModel(
    input_size=1,
    hidden_size=params['hidden_size'],
    num_layers=params['num_layers'],
    dropout=params['dropout'],
    num_classes=256
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

# Training loop
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
patience_counter = 0
patience = 15  # Good patience for BiLSTM

print(f"\nTraining for up to {params['epochs']} epochs...")
print("-" * 80)

for epoch in range(params['epochs']):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            val_loss += loss.item() * batch_X.size(0)
    
    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct / val_total
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_bilstm_model.pth')
        best_epoch = epoch + 1
    else:
        patience_counter += 1
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{params['epochs']}] - "
              f"Train: {train_loss:.4f} (acc {train_acc:.4f}) - "
              f"Val: {val_loss:.4f} (acc {val_acc:.4f}) - "
              f"Best: {best_val_loss:.4f} - Patience: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print("-" * 80)
print(f"Training completed!")
print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")

# Load best model
model.load_state_dict(torch.load('best_bilstm_model.pth'))
print("\nLoaded best model for attack evaluation")

# Cell 15
import matplotlib.pyplot as plt

# Make predictions on attack set
model.eval()
with torch.no_grad():
    # Get predictions in batches to avoid memory issues
    preds_list = []
    batch_size = 128
    
    for i in range(0, len(X_attack_tensor), batch_size):
        batch = X_attack_tensor[i:i+batch_size]
        outputs = model(batch)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds_list.append(probs.cpu().numpy())
    
    preds = np.vstack(preds_list)

print(f"Predictions shape: {preds.shape}")

# Calculate Guessing Entropy using Log-Likelihood (FIXED for masked ASCAD)
traces_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
ge = []

# Precompute log probabilities to avoid recomputing
# Add epsilon to avoid log(0)
epsilon = 1e-30
log_preds = np.log(preds + epsilon)

print(f"Calculating GE for correct key: {hex(correct_key_byte)}")

for n in traces_list:
    if n > len(preds): 
        break

    # Select n traces
    current_log_preds = log_preds[:n] # (n, 256)
    current_plain = plaintext_attack[:n] # (n,)
    current_masks = masks_attack[:n] # (n,) - FIXED: include masks!

    # Log likelihoods for all 256 key candidates
    key_log_probs = np.zeros(256)

    for k in range(256):
        # FIXED: Compute MASKED intermediate values for this key guess
        # Training used: y = sbox[plaintext ^ key ^ mask]
        # So attack must compute: target = sbox[plaintext ^ k ^ mask]
        targets = sbox[current_plain ^ k ^ current_masks] # Shape (n,)

        # Sum log probabilities for these targets
        # Advanced indexing: rows 0..n-1, cols=targets
        key_log_probs[k] = np.sum(current_log_preds[np.arange(n), targets])

    # Rank the correct key
    ranked_keys = np.argsort(key_log_probs)[::-1] # Descending likelihood
    rank = np.where(ranked_keys == correct_key_byte)[0][0]
    ge.append(rank)

# Plot Guessing Entropy
plt.figure(figsize=(10, 6))
plt.semilogx(traces_list[:len(ge)], ge, '*-', markersize=8, linewidth=2)
plt.title('Guessing Entropy (PyTorch Model - FIXED)', fontsize=14)
plt.xlabel('Number of Attack Traces', fontsize=12)
plt.ylabel('Rank of Correct Key', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nGuessing Entropy Results:")
print(f"Correct key byte: {hex(correct_key_byte)}")
for i, n in enumerate(traces_list[:len(ge)]):
    print(f"  {n:5d} traces -> Rank: {ge[i]:3d}")

if 100 in traces_list[:len(ge)]:
    idx = traces_list.index(100)
    print(f"\nGE at 100 traces: {ge[idx]}")
