"""
Unified Dataset Loader for SCA Experiments
============================================
Supports: ASCAD, AES_HD, CHES_CTF
Provides consistent interface across all datasets.

Download datasets:
    python download_datasets.py
"""

import h5py
import numpy as np
import os

# AES Forward S-box
AES_SBOX = np.array([
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

# AES Inverse S-box (for AES_HD last-round attacks)
AES_SBOX_INV = np.array([
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
], dtype=np.uint8)

# ShiftRows inverse mapping for AES_HD (maps output byte to input byte)
AES_SHIFT_MAP = [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3]


def hamming_weight(x):
    """Compute Hamming weight (number of 1-bits) of byte values."""
    return np.array([bin(v).count('1') for v in np.atleast_1d(x)], dtype=np.uint8)


class DatasetConfig:
    """Configuration for each dataset."""
    def __init__(self, name, file_path, target_byte, leakage_model, n_classes):
        self.name = name
        self.file_path = file_path
        self.target_byte = target_byte
        self.leakage_model = leakage_model  # 'ID' (identity/256 classes) or 'HW'/'HD' (9 classes)
        self.n_classes = n_classes


def load_ascad(file_path, target_byte=2, leakage_model='ID'):
    """
    Load ASCAD dataset (ATMega8515, masked AES, first-round S-box).

    Leakage model:
        - 'ID': S-box output identity (256 classes)
        - 'HW': Hamming weight of S-box output (9 classes)

    Uses masks in label computation (masked implementation).
    """
    print(f"Loading ASCAD dataset (byte={target_byte}, model={leakage_model})...")

    with h5py.File(file_path, 'r') as f:
        X_prof = np.array(f['Profiling_traces/traces'], dtype=np.float32)
        metadata_prof = f['Profiling_traces/metadata']
        X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
        metadata_attack = f['Attack_traces/metadata']

        # Extract metadata
        plaintext_prof = np.array([m['plaintext'][target_byte] for m in metadata_prof])
        masks_prof = np.array([m['masks'][target_byte] for m in metadata_prof])
        key_byte_prof = metadata_prof[0]['key'][target_byte]

        plaintext_attack = np.array([m['plaintext'][target_byte] for m in metadata_attack])
        masks_attack = np.array([m['masks'][target_byte] for m in metadata_attack])
        correct_key = metadata_attack[0]['key'][target_byte]

    # Compute labels: S-box(plaintext XOR key XOR mask) for masked implementation
    y_prof_id = AES_SBOX[plaintext_prof ^ key_byte_prof ^ masks_prof]

    if leakage_model == 'ID':
        y_prof = y_prof_id
        n_classes = 256
    else:  # HW
        y_prof = hamming_weight(y_prof_id)
        n_classes = 9

    # Normalize
    mean = X_prof.mean(axis=0)
    std = X_prof.std(axis=0) + 1e-8
    X_prof = (X_prof - mean) / std
    X_attack = (X_attack - mean) / std

    print(f"  Profiling: {X_prof.shape}, Attack: {X_attack.shape}")
    print(f"  Classes: {n_classes}, Correct key byte: {hex(correct_key)}")

    return {
        'X_prof': X_prof,
        'y_prof': y_prof,
        'X_attack': X_attack,
        'plaintext_attack': plaintext_attack,
        'masks_attack': masks_attack,
        'correct_key': correct_key,
        'n_classes': n_classes,
        'leakage_model': leakage_model,
        'target_byte': target_byte,
        'dataset_name': 'ASCAD',
        'sbox': AES_SBOX,
        'uses_masks': True,
        'trace_length': X_prof.shape[1],
        'attack_function': 'ascad'
    }


def load_aes_hd(file_path, target_byte=0, leakage_model='ID'):
    """
    Load AES_HD dataset (Xilinx FPGA, unprotected AES, last-round HD).

    Leakage model:
        - 'ID': Intermediate value identity (256 classes)
        - 'HD': Hamming distance between register values (9 classes)

    Last-round attack: uses ciphertext and inverse S-box.
    """
    print(f"Loading AES_HD dataset (byte={target_byte}, model={leakage_model})...")

    with h5py.File(file_path, 'r') as f:
        X_prof = np.array(f['Profiling_traces/traces'], dtype=np.float32)
        metadata_prof = f['Profiling_traces/metadata']
        X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
        metadata_attack = f['Attack_traces/metadata']

        # AES_HD uses ciphertext for last-round attack
        ciphertext_prof = np.array([m['ciphertext'][target_byte] for m in metadata_prof])
        key_byte_prof = metadata_prof[0]['key'][target_byte]

        ciphertext_attack = np.array([m['ciphertext'][target_byte] for m in metadata_attack])
        ciphertext_shift_attack = np.array([m['ciphertext'][AES_SHIFT_MAP[target_byte]] for m in metadata_attack])
        correct_key = metadata_attack[0]['key'][target_byte]

        # Also get profiling shift bytes for label computation
        ciphertext_shift_prof = np.array([m['ciphertext'][AES_SHIFT_MAP[target_byte]] for m in metadata_prof])

    # Compute labels: Sbox_inv(ct[j] XOR key[j]) XOR ct[shift(j)]
    intermediate_prof = AES_SBOX_INV[ciphertext_prof ^ key_byte_prof] ^ ciphertext_shift_prof

    if leakage_model == 'ID':
        y_prof = intermediate_prof
        n_classes = 256
    else:  # HD
        y_prof = hamming_weight(intermediate_prof)
        n_classes = 9

    # Normalize
    mean = X_prof.mean(axis=0)
    std = X_prof.std(axis=0) + 1e-8
    X_prof = (X_prof - mean) / std
    X_attack = (X_attack - mean) / std

    print(f"  Profiling: {X_prof.shape}, Attack: {X_attack.shape}")
    print(f"  Classes: {n_classes}, Correct key byte: {hex(correct_key)}")

    return {
        'X_prof': X_prof,
        'y_prof': y_prof,
        'X_attack': X_attack,
        'ciphertext_attack': ciphertext_attack,
        'ciphertext_shift_attack': ciphertext_shift_attack,
        'correct_key': correct_key,
        'n_classes': n_classes,
        'leakage_model': leakage_model,
        'target_byte': target_byte,
        'dataset_name': 'AES_HD',
        'sbox_inv': AES_SBOX_INV,
        'uses_masks': False,
        'trace_length': X_prof.shape[1],
        'attack_function': 'aes_hd'
    }


def load_ches_ctf(file_path, target_byte=0, leakage_model='ID'):
    """
    Load CHES CTF 2018 dataset (STM32, masked AES, first-round S-box).

    Leakage model:
        - 'ID': S-box output identity (256 classes)
        - 'HW': Hamming weight of S-box output (9 classes)

    First-round attack: uses plaintext and forward S-box.
    Masked implementation but masks may not be in metadata.
    """
    print(f"Loading CHES_CTF dataset (byte={target_byte}, model={leakage_model})...")

    with h5py.File(file_path, 'r') as f:
        X_prof = np.array(f['Profiling_traces/traces'], dtype=np.float32)
        metadata_prof = f['Profiling_traces/metadata']
        X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
        metadata_attack = f['Attack_traces/metadata']

        plaintext_prof = np.array([m['plaintext'][target_byte] for m in metadata_prof])
        key_byte_prof = metadata_prof[0]['key'][target_byte]

        plaintext_attack = np.array([m['plaintext'][target_byte] for m in metadata_attack])
        correct_key = metadata_attack[0]['key'][target_byte]

    # Compute labels: S-box(plaintext XOR key) - no masks in profiling labels
    y_prof_id = AES_SBOX[plaintext_prof ^ key_byte_prof]

    if leakage_model == 'ID':
        y_prof = y_prof_id
        n_classes = 256
    else:  # HW
        y_prof = hamming_weight(y_prof_id)
        n_classes = 9

    # Normalize (CHES_CTF may already be normalized, but we do it anyway for consistency)
    mean = X_prof.mean(axis=0)
    std = X_prof.std(axis=0) + 1e-8
    X_prof = (X_prof - mean) / std
    X_attack = (X_attack - mean) / std

    print(f"  Profiling: {X_prof.shape}, Attack: {X_attack.shape}")
    print(f"  Classes: {n_classes}, Correct key byte: {hex(correct_key)}")

    # CHES_CTF doesn't use masks in label computation
    dummy_masks = np.zeros_like(plaintext_attack, dtype=np.uint8)

    return {
        'X_prof': X_prof,
        'y_prof': y_prof,
        'X_attack': X_attack,
        'plaintext_attack': plaintext_attack,
        'masks_attack': dummy_masks,
        'correct_key': correct_key,
        'n_classes': n_classes,
        'leakage_model': leakage_model,
        'target_byte': target_byte,
        'dataset_name': 'CHES_CTF',
        'sbox': AES_SBOX,
        'uses_masks': False,
        'trace_length': X_prof.shape[1],
        'attack_function': 'ches_ctf'
    }


def key_rank_attack(log_probs, dataset_info, n_traces):
    """
    Compute key rank for a given number of attack traces.
    Handles different datasets (ASCAD uses masks, AES_HD uses ciphertext, etc.)

    Args:
        log_probs: (n_attack_traces, n_classes) log-probabilities from model
        dataset_info: dict from load_* functions
        n_traces: number of attack traces to use

    Returns:
        rank: key rank (0 = correct key is top-ranked)
    """
    correct_key = dataset_info['correct_key']
    attack_fn = dataset_info['attack_function']
    n = min(n_traces, len(log_probs))

    if attack_fn == 'ascad':
        # ASCAD: first-round with masks
        pt = dataset_info['plaintext_attack'][:n]
        masks = dataset_info['masks_attack'][:n]
        scores = np.zeros(256)
        for k in range(256):
            intermediate = AES_SBOX[pt ^ k ^ masks]
            if dataset_info['leakage_model'] == 'HW':
                intermediate = hamming_weight(intermediate)
            scores[k] = np.sum(log_probs[np.arange(n), intermediate])

    elif attack_fn == 'aes_hd':
        # AES_HD: last-round, uses ciphertext
        ct = dataset_info['ciphertext_attack'][:n]
        ct_shift = dataset_info['ciphertext_shift_attack'][:n]
        scores = np.zeros(256)
        for k in range(256):
            intermediate = AES_SBOX_INV[ct ^ k] ^ ct_shift
            if dataset_info['leakage_model'] == 'HD':
                intermediate = hamming_weight(intermediate)
            scores[k] = np.sum(log_probs[np.arange(n), intermediate])

    elif attack_fn == 'ches_ctf':
        # CHES_CTF: first-round, no masks in attack
        pt = dataset_info['plaintext_attack'][:n]
        scores = np.zeros(256)
        for k in range(256):
            intermediate = AES_SBOX[pt ^ k]
            if dataset_info['leakage_model'] == 'HW':
                intermediate = hamming_weight(intermediate)
            scores[k] = np.sum(log_probs[np.arange(n), intermediate])

    rank = np.argsort(-scores).tolist().index(correct_key)
    return rank


def load_dataset(dataset_name, target_byte=None, leakage_model='ID'):
    """
    Load any supported dataset by name.

    Args:
        dataset_name: 'ascad', 'aes_hd', or 'ches_ctf'
        target_byte: target key byte index (default varies by dataset)
        leakage_model: 'ID' (256 classes) or 'HW'/'HD' (9 classes)

    Returns:
        dict with all dataset information
    """
    base_dir = r'C:\Users\Dspike\Documents\sca'

    if dataset_name.lower() == 'ascad':
        file_path = os.path.join(base_dir, 'ASCAD_data', 'ASCAD_data', 'ASCAD_databases', 'ASCAD.h5')
        byte = target_byte if target_byte is not None else 2
        return load_ascad(file_path, byte, leakage_model)

    elif dataset_name.lower() == 'aes_hd':
        file_path = os.path.join(base_dir, 'datasets', 'aes_hd.h5')
        byte = target_byte if target_byte is not None else 0
        model = leakage_model if leakage_model != 'HW' else 'HD'  # AES_HD uses HD, not HW
        return load_aes_hd(file_path, byte, model)

    elif dataset_name.lower() == 'ches_ctf':
        file_path = os.path.join(base_dir, 'datasets', 'ches_ctf.h5')
        byte = target_byte if target_byte is not None else 0
        return load_ches_ctf(file_path, byte, leakage_model)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Supported: ascad, aes_hd, ches_ctf")


# Quick test
if __name__ == '__main__':
    import sys
    ds_name = sys.argv[1] if len(sys.argv) > 1 else 'ascad'
    print(f"\nTesting dataset loader for: {ds_name}")
    try:
        data = load_dataset(ds_name)
        print(f"\nDataset: {data['dataset_name']}")
        print(f"  Traces: {data['X_prof'].shape}")
        print(f"  Attack traces: {data['X_attack'].shape}")
        print(f"  Classes: {data['n_classes']}")
        print(f"  Leakage model: {data['leakage_model']}")
        print(f"  Correct key: {hex(data['correct_key'])}")
        print(f"  Trace length: {data['trace_length']}")
        print("\nDataset loaded successfully!")
    except FileNotFoundError as e:
        print(f"\nDataset file not found: {e}")
        print("Run: python download_datasets.py")
