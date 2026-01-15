"""
Check the raw traces file - this might be the full-length data!
"""
import h5py
import os

file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ATMEGA_AES_v1\ATM_AES_v1_fixed_key\ASCAD_data\ASCAD_databases\ATMega8515_raw_traces.h5'

print("="*70)
print("Checking RAW TRACES File (5.6 GB)")
print("="*70)

if not os.path.exists(file_path):
    print(f"\n✗ File not found")
else:
    file_size = os.path.getsize(file_path) / (1024**3)
    print(f"\n✓ File exists!")
    print(f"  File size: {file_size:.2f} GB")

    with h5py.File(file_path, 'r') as f:
        print(f"\n  Keys in file: {list(f.keys())}")

        # Try to find traces
        for key in f.keys():
            print(f"\n  Exploring '{key}':")
            if isinstance(f[key], h5py.Group):
                print(f"    Subkeys: {list(f[key].keys())}")

                # Check for traces
                if 'traces' in f[key].keys():
                    shape = f[key]['traces'].shape
                    print(f"    Traces shape: {shape}")
                    print(f"    Trace length: {shape[1]:,} points")

                    if shape[1] > 50000:
                        print(f"    ✓✓✓ FULL-LENGTH TRACES FOUND!")

print("\n" + "="*70)
