"""
Diagnose ASCAD Dataset Issues
==============================
Check SNR, leakage, and data quality
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

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

file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD.h5'

print("="*70)
print("ASCAD Dataset Diagnosis")
print("="*70)

with h5py.File(file_path, 'r') as f:
    print("\n1. Dataset Structure:")
    print(f"   Keys in file: {list(f.keys())}")

    X_prof = np.array(f['Profiling_traces/traces'][:5000], dtype=np.float32)
    metadata_prof = f['Profiling_traces/metadata'][:5000]

    print(f"\n2. Trace Information:")
    print(f"   Trace shape: {X_prof.shape}")
    print(f"   Trace length: {X_prof.shape[1]} points")
    print(f"   Trace min/max: {X_prof.min():.2f} / {X_prof.max():.2f}")
    print(f"   Trace mean/std: {X_prof.mean():.2f} / {X_prof.std():.2f}")

    print(f"\n3. Metadata Check:")
    sample_meta = metadata_prof[0]
    print(f"   Metadata fields: {sample_meta.dtype.names}")
    print(f"   Sample plaintext: {[hex(x) for x in sample_meta['plaintext'][:4]]}")
    print(f"   Sample key: {[hex(x) for x in sample_meta['key'][:4]]}")
    print(f"   Sample masks: {[hex(x) for x in sample_meta['masks'][:4]]}")

    # Try different bytes
    print(f"\n4. Testing Different Key Bytes:")
    for byte_idx in [0, 2, 3]:
        plaintext = np.array([m['plaintext'][byte_idx] for m in metadata_prof])
        masks = np.array([m['masks'][byte_idx] for m in metadata_prof])
        key_byte = metadata_prof[0]['key'][byte_idx]

        # Label: S-box output of (plaintext XOR key XOR mask)
        labels = sbox[plaintext ^ key_byte ^ masks]

        # Compute SNR for this byte
        n_features = min(100, X_prof.shape[1])  # Check first 100 points
        snr = np.zeros(n_features)

        for i in range(n_features):
            class_means = []
            class_vars = []

            for c in range(256):
                mask = (labels == c)
                if mask.sum() > 2:
                    class_means.append(np.mean(X_prof[mask, i]))
                    class_vars.append(np.var(X_prof[mask, i]))

            if len(class_means) > 1:
                snr[i] = np.var(class_means) / (np.mean(class_vars) + 1e-12)

        max_snr = snr.max()
        print(f"   Byte {byte_idx}: Max SNR = {max_snr:.4f} (key={hex(key_byte)})")

        if max_snr > 0.1:
            print(f"      ✓ GOOD SNR! Byte {byte_idx} has clear leakage")
        else:
            print(f"      ✗ LOW SNR - Byte {byte_idx} might be problematic")

    # Plot SNR for best byte
    print(f"\n5. Computing Full SNR for Byte 3 (often has best leakage)...")
    plaintext = np.array([m['plaintext'][3] for m in metadata_prof])
    masks = np.array([m['masks'][3] for m in metadata_prof])
    key_byte = metadata_prof[0]['key'][3]
    labels = sbox[plaintext ^ key_byte ^ masks]

    print(f"   Computing SNR for all {X_prof.shape[1]} points...")
    full_snr = np.zeros(X_prof.shape[1])

    for i in range(X_prof.shape[1]):
        if i % 100 == 0:
            print(f"      Progress: {i}/{X_prof.shape[1]}")

        class_means = []
        class_vars = []

        for c in range(256):
            mask = (labels == c)
            if mask.sum() > 2:
                class_means.append(np.mean(X_prof[mask, i]))
                class_vars.append(np.var(X_prof[mask, i]))

        if len(class_means) > 1:
            full_snr[i] = np.var(class_means) / (np.mean(class_vars) + 1e-12)

    print(f"\n6. SNR Analysis (Byte 3):")
    print(f"   Min SNR: {full_snr.min():.6f}")
    print(f"   Max SNR: {full_snr.max():.6f}")
    print(f"   Mean SNR: {full_snr.mean():.6f}")
    print(f"   Points with SNR > 0.1: {(full_snr > 0.1).sum()}")
    print(f"   Points with SNR > 1.0: {(full_snr > 1.0).sum()}")

    # Find peak
    peak_idx = np.argmax(full_snr)
    print(f"   Peak SNR at point {peak_idx}: {full_snr[peak_idx]:.6f}")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(full_snr)
    plt.xlabel('Time Point')
    plt.ylabel('SNR')
    plt.title('SNR vs Time (Byte 3)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('snr_diagnosis.png', dpi=100)
    print(f"\n✓ Saved SNR plot to: snr_diagnosis.png")

    print(f"\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)

    if full_snr.max() > 1.0:
        print("✓ Dataset looks GOOD - high SNR detected")
        print(f"  Recommendation: Use byte 3, select top 500-700 points")
    elif full_snr.max() > 0.1:
        print("⚠ Dataset has MODERATE SNR")
        print(f"  Recommendation: Use byte 3, increase training iterations")
    else:
        print("✗ Dataset has VERY LOW SNR")
        print(f"  Possible issues:")
        print(f"    - Wrong dataset variant (desynchronized?)")
        print(f"    - Incorrect label computation")
        print(f"    - This is a very challenging dataset")
