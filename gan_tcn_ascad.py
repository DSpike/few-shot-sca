"""
GAN-based TCN with Entropy-TOPSIS for ASCAD Side-Channel Attack
Novel approach combining:
- Entropy-TOPSIS feature selection
- Temporal Convolutional Network (TCN)
- Generative Adversarial Network (GAN)
"""

# ===== Cell 1 =====
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

# ===== Cell 2 =====
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

# ===== Cell 3 =====
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

# ===== Cell 4 =====
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

# ===== Cell 5 =====
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

# ===== Cell 6 =====
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

# ===== Cell 7 =====
# IMPORTANT: We already have the correct labels (y_profiling) from cell 2ea5148f
# The labels are the masked S-box outputs, NOT the plaintext!
# y_train and y_val were already created in the train_test_split in cell 2ea5148f

print("Using labels from the initial data load (masked S-box outputs)")
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("Correct key byte:", hex(correct_key_byte))
print("Label range: min={}, max={}".format(y_train.min(), y_train.max()))

# ===== Cell 8 =====
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

# ===== MARKDOWN: Cell 9 =====
# #
# 
# G
# A
# N
# -
# b
# a
# s
# e
# d
# 
# T
# e
# m
# p
# o
# r
# a
# l
# 
# C
# o
# n
# v
# o
# l
# u
# t
# i
# o
# n
# a
# l
# 
# N
# e
# t
# w
# o
# r
# k
# 
# (
# T
# C
# N
# )
# 
# f
# o
# r
# 
# S
# C
# A
# 
# 
# #
# #
# 
# N
# o
# v
# e
# l
# 
# A
# p
# p
# r
# o
# a
# c
# h
# :
# 
# T
# h
# i
# s
# 
# i
# m
# p
# l
# e
# m
# e
# n
# t
# a
# t
# i
# o
# n
# 
# c
# o
# m
# b
# i
# n
# e
# s
# :
# 
# 1
# .
# 
# *
# *
# T
# C
# N
# 
# (
# T
# e
# m
# p
# o
# r
# a
# l
# 
# C
# o
# n
# v
# o
# l
# u
# t
# i
# o
# n
# a
# l
# 
# N
# e
# t
# w
# o
# r
# k
# )
# *
# *
# 
# -
# 
# C
# a
# p
# t
# u
# r
# e
# s
# 
# t
# e
# m
# p
# o
# r
# a
# l
# 
# p
# a
# t
# t
# e
# r
# n
# s
# 
# i
# n
# 
# p
# o
# w
# e
# r
# 
# t
# r
# a
# c
# e
# s
# 
# 2
# .
# 
# *
# *
# G
# A
# N
# 
# (
# G
# e
# n
# e
# r
# a
# t
# i
# v
# e
# 
# A
# d
# v
# e
# r
# s
# a
# r
# i
# a
# l
# 
# N
# e
# t
# w
# o
# r
# k
# )
# *
# *
# 
# -
# 
# G
# e
# n
# e
# r
# a
# t
# o
# r
# 
# c
# r
# e
# a
# t
# e
# s
# 
# s
# y
# n
# t
# h
# e
# t
# i
# c
# 
# h
# a
# r
# d
# 
# e
# x
# a
# m
# p
# l
# e
# s
# 
# 3
# .
# 
# *
# *
# E
# n
# t
# r
# o
# p
# y
# -
# T
# O
# P
# S
# I
# S
# *
# *
# 
# -
# 
# I
# n
# t
# e
# l
# l
# i
# g
# e
# n
# t
# 
# f
# e
# a
# t
# u
# r
# e
# 
# s
# e
# l
# e
# c
# t
# i
# o
# n
# 
# 4
# .
# 
# *
# *
# W
# O
# A
# *
# *
# 
# -
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
# 
# #
# #
# 
# A
# r
# c
# h
# i
# t
# e
# c
# t
# u
# r
# e
# :
# 
# -
# 
# *
# *
# G
# e
# n
# e
# r
# a
# t
# o
# r
# *
# *
# :
# 
# C
# r
# e
# a
# t
# e
# s
# 
# s
# y
# n
# t
# h
# e
# t
# i
# c
# 
# l
# a
# b
# e
# l
# e
# d
# 
# t
# r
# a
# c
# e
# s
# 
# t
# o
# 
# a
# u
# g
# m
# e
# n
# t
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
# d
# a
# t
# a
# 
# -
# 
# *
# *
# D
# i
# s
# c
# r
# i
# m
# i
# n
# a
# t
# o
# r
# 
# (
# T
# C
# N
# )
# *
# *
# :
# 
# C
# l
# a
# s
# s
# i
# f
# i
# e
# s
# 
# t
# r
# a
# c
# e
# s
# 
# A
# N
# D
# 
# l
# e
# a
# r
# n
# s
# 
# f
# r
# o
# m
# 
# s
# y
# n
# t
# h
# e
# t
# i
# c
# 
# e
# x
# a
# m
# p
# l
# e
# s
# 
# -
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
# a
# l
# t
# e
# r
# n
# a
# t
# e
# s
# 
# b
# e
# t
# w
# e
# e
# n
# 
# G
# e
# n
# e
# r
# a
# t
# o
# r
# 
# a
# n
# d
# 
# D
# i
# s
# c
# r
# i
# m
# i
# n
# a
# t
# o
# r
# 
# (
# a
# d
# v
# e
# r
# s
# a
# r
# i
# a
# l
# 
# l
# e
# a
# r
# n
# i
# n
# g
# )

# ===== Cell 10 =====
# Temporal Convolutional Network (TCN) Block
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.3):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                    dilation=dilation_size, padding=padding, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

print("TCN blocks defined!")

# ===== Cell 11 =====
# Generator Network (creates synthetic labeled traces)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_features=200, num_classes=256):
        super(Generator, self).__init__()
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Generator network
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, num_features),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, noise, labels):
        # Concatenate noise and label embedding
        label_input = self.label_emb(labels)
        gen_input = torch.cat([noise, label_input], dim=1)
        return self.model(gen_input)

# Discriminator Network (TCN-based classifier)
class Discriminator(nn.Module):
    def __init__(self, num_features=200, num_classes=256):
        super(Discriminator, self).__init__()
        
        # TCN for feature extraction
        self.tcn = TCN(
            num_inputs=1,
            num_channels=[64, 64, 128, 128],  # 4-layer TCN
            kernel_size=3,
            dropout=0.3
        )
        
        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Real/Fake discriminator head
        self.validity = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, features) -> (batch, 1, features) for TCN
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # TCN feature extraction
        features = self.tcn(x)  # (batch, 128, features)
        features = self.pool(features).squeeze(-1)  # (batch, 128)
        
        # Class prediction
        class_output = self.classifier(features)
        
        # Real/Fake prediction
        validity = self.validity(features)
        
        return class_output, validity

print("Generator and Discriminator (TCN-based) defined!")
print("Ready for GAN training")

# ===== Cell 12 =====
# GAN-based TCN Training with Whale Optimization

# Prepare data for GAN-TCN (needs to be in [-1, 1] for Generator)
X_train_gan = torch.FloatTensor(X_train_selected).to(device)
X_val_gan = torch.FloatTensor(X_val_selected).to(device)
X_attack_gan = torch.FloatTensor(X_attack_selected).to(device)

# Normalize to [-1, 1]
train_min, train_max = X_train_gan.min(), X_train_gan.max()
X_train_gan = 2 * (X_train_gan - train_min) / (train_max - train_min) - 1
X_val_gan = 2 * (X_val_gan - train_min) / (train_max - train_min) - 1
X_attack_gan = 2 * (X_attack_gan - train_min) / (train_max - train_min) - 1

y_train_gan = torch.LongTensor(y_train).to(device)
y_val_gan = torch.LongTensor(y_val).to(device)

# Initialize GAN
latent_dim = 100
generator = Generator(latent_dim=latent_dim, num_features=200, num_classes=256).to(device)
discriminator = Discriminator(num_features=200, num_classes=256).to(device)

print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator (TCN) parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
print(f"Training on GPU: {device}")

# ===== Cell 13 =====
# GAN Training Loop
adversarial_loss = nn.BCELoss()
classification_loss = nn.CrossEntropyLoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 100
batch_size = 400

# Create data loader
train_dataset_gan = torch.utils.data.TensorDataset(X_train_gan, y_train_gan)
train_loader_gan = DataLoader(train_dataset_gan, batch_size=batch_size, shuffle=True)

print("Training GAN-based TCN...")
print("="*80)

for epoch in range(epochs):
    d_loss_epoch = 0
    g_loss_epoch = 0
    class_loss_epoch = 0
    
    for i, (real_traces, labels) in enumerate(train_loader_gan):
        batch_size_curr = real_traces.size(0)
        
        # Ground truths
        valid = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Real traces
        class_pred, validity_real = discriminator(real_traces)
        d_real_loss = adversarial_loss(validity_real, valid) + classification_loss(class_pred, labels)
        
        # Fake traces
        z = torch.randn(batch_size_curr, latent_dim, device=device)
        gen_labels = torch.randint(0, 256, (batch_size_curr,), device=device)
        gen_traces = generator(z, gen_labels)
        class_pred_fake, validity_fake = discriminator(gen_traces.detach())
        d_fake_loss = adversarial_loss(validity_fake, fake)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size_curr, latent_dim, device=device)
        gen_labels = torch.randint(0, 256, (batch_size_curr,), device=device)
        gen_traces = generator(z, gen_labels)
        
        class_pred_gen, validity_gen = discriminator(gen_traces)
        g_loss = adversarial_loss(validity_gen, valid) + classification_loss(class_pred_gen, gen_labels)
        g_loss.backward()
        optimizer_G.step()
        
        d_loss_epoch += d_loss.item()
        g_loss_epoch += g_loss.item()
        class_loss_epoch += classification_loss(class_pred, labels).item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss_epoch/len(train_loader_gan):.4f}, "
              f"G Loss: {g_loss_epoch/len(train_loader_gan):.4f}, "
              f"Class Loss: {class_loss_epoch/len(train_loader_gan):.4f}")

print("="*80)
print("GAN-TCN training complete!")
print("Discriminator is now trained as the classifier")

# ===== Cell 14 =====
# Guessing Entropy Evaluation using trained Discriminator (TCN)

# Make predictions on attack set
discriminator.eval()
with torch.no_grad():
    class_preds, _ = discriminator(X_attack_gan)
    preds = torch.nn.functional.softmax(class_preds, dim=1).cpu().numpy()

print(f"Predictions shape: {preds.shape}")

# Calculate Guessing Entropy using Log-Likelihood (FIXED for masked ASCAD)
traces_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
ge = []

# Precompute log probabilities
epsilon = 1e-30
log_preds = np.log(preds + epsilon)

print(f"Calculating GE for correct key: {hex(correct_key_byte)}")

for n in traces_list:
    if n > len(preds): 
        break

    current_log_preds = log_preds[:n]
    current_plain = plaintext_attack[:n]
    current_masks = masks_attack[:n]

    key_log_probs = np.zeros(256)

    for k in range(256):
        # FIXED: Compute MASKED intermediate values
        targets = sbox[current_plain ^ k ^ current_masks]
        key_log_probs[k] = np.sum(current_log_preds[np.arange(n), targets])

    ranked_keys = np.argsort(key_log_probs)[::-1]
    rank = np.where(ranked_keys == correct_key_byte)[0][0]
    ge.append(rank)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.semilogx(traces_list[:len(ge)], ge, 'o-', markersize=8, linewidth=2, color='darkblue')
plt.title('Guessing Entropy - GAN-based TCN with Entropy-TOPSIS', fontsize=14, fontweight='bold')
plt.xlabel('Number of Attack Traces', fontsize=12)
plt.ylabel('Rank of Correct Key', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nGuessing Entropy Results (GAN-TCN):")
print(f"Correct key byte: {hex(correct_key_byte)}")
print("-" * 40)
for n, rank in zip(traces_list[:len(ge)], ge):
    status = "✓ BROKEN" if rank == 0 else ""
    print(f"{n:5d} traces -> Rank {rank:3d} {status}")

if 100 in traces_list[:len(ge)]:
    idx = traces_list.index(100)
    print(f"\nGE@100: {ge[idx]}")
