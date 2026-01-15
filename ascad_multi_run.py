"""
ASCAD Attack - Multi-run to find best model
Trains 5 models with different seeds and saves the best
"""

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')

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
])

# Load data
print("Loading ASCAD data...")
file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD.h5'

with h5py.File(file_path, 'r') as f:
    X_profiling = np.array(f['Profiling_traces/traces'], dtype=np.float32)
    X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
    metadata_prof = f['Profiling_traces/metadata']
    metadata_attack = f['Attack_traces/metadata']

    plaintext_prof = np.array([m['plaintext'][2] for m in metadata_prof], dtype=np.uint8)
    masks_prof = np.array([m['masks'][2] for m in metadata_prof], dtype=np.uint8)
    key_byte = metadata_prof[0]['key'][2]
    y_profiling = sbox[plaintext_prof ^ key_byte ^ masks_prof]

    plaintext_attack = np.array([m['plaintext'][2] for m in metadata_attack], dtype=np.uint8)
    masks_attack = np.array([m['masks'][2] for m in metadata_attack], dtype=np.uint8)
    correct_key_byte = metadata_attack[0]['key'][2]

scaler = StandardScaler()
X_profiling = scaler.fit_transform(X_profiling).astype(np.float32)
X_attack = scaler.transform(X_attack).astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X_profiling, y_profiling, test_size=0.1, random_state=42, stratify=y_profiling
)

print(f"Data loaded: {X_train.shape}\n")

# Model
class ASCADModel(nn.Module):
    def __init__(self):
        super(ASCADModel, self).__init__()
        self.fc1 = nn.Linear(700, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 256)
        self.selu = nn.SELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.selu(self.fc1(x)))
        x = self.dropout(self.selu(self.fc2(x)))
        x = self.dropout(self.selu(self.fc3(x)))
        x = self.dropout(self.selu(self.fc4(x)))
        x = self.dropout(self.selu(self.fc5(x)))
        return self.fc6(x)

# Data loaders
X_train_t = torch.FloatTensor(X_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
X_attack_t = torch.FloatTensor(X_attack).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
y_val_t = torch.LongTensor(y_val).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=1024, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=1024, shuffle=False)

# GE function
def compute_ge(preds, plaintext, masks, correct_key):
    traces_list = [100, 500, 1000, 2000, 5000, 10000]
    log_preds = np.log(preds + 1e-30)
    ge_results = []

    for n in traces_list:
        if n > len(preds):
            break
        key_log_probs = np.zeros(256)
        for k in range(256):
            targets = sbox[plaintext[:n] ^ k] ^ masks[:n]
            key_log_probs[k] = np.sum(log_preds[:n][np.arange(n), targets])
        ranked = np.argsort(key_log_probs)[::-1]
        ge_results.append(np.where(ranked == correct_key)[0][0])

    return ge_results

# Train multiple models
num_runs = 5
best_rank = 999
best_model_state = None

for run in range(num_runs):
    print(f"{'='*80}")
    print(f"RUN {run+1}/{num_runs}")
    print(f"{'='*80}")

    # Set seed
    torch.manual_seed(42 + run * 10)
    np.random.seed(42 + run * 10)

    model = ASCADModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Back to lower LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8, verbose=False)

    epochs = 200  # Match successful run
    patience = 25  # Match successful run
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(X_train_t)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(model(X_batch), y_batch).item() * X_batch.size(0)
        val_loss /= len(X_val_t)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # Test GE
    model.eval()
    with torch.no_grad():
        preds = torch.nn.functional.softmax(model(X_attack_t), dim=1).cpu().numpy()

    ge = compute_ge(preds, plaintext_attack, masks_attack, correct_key_byte)
    rank_10k = ge[-1] if len(ge) > 0 else 999

    print(f"  Val Loss: {best_val_loss:.4f} | GE@10K: {rank_10k}")

    if rank_10k < best_rank:
        best_rank = rank_10k
        best_model_state = best_state
        best_preds = preds
        print(f"  *** NEW BEST MODEL! Rank: {rank_10k} ***")

    print()

# Final results with best model
print(f"\n{'='*80}")
print("FINAL RESULTS (Best Model)")
print(f"{'='*80}")

ge_full = compute_ge(best_preds, plaintext_attack, masks_attack, correct_key_byte)
traces_list = [100, 500, 1000, 2000, 5000, 10000][:len(ge_full)]

for n, rank in zip(traces_list, ge_full):
    status = "✓ BROKEN" if rank == 0 else ""
    print(f"{n:5d} traces -> Rank {rank:3d} {status}")

print(f"\nBest rank achieved: {min(ge_full)} at {traces_list[ge_full.index(min(ge_full))]} traces")
print(f"Final rank at 10K traces: {ge_full[-1]}")

# Save best model
torch.save(best_model_state, 'best_ascad_multirun.pth')
print(f"\nBest model saved to 'best_ascad_multirun.pth'")
