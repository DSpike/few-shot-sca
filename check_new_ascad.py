"""
Check the ASCAD file in ATMEGA directory
"""
import h5py
import os

file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ATMEGA_AES_v1\ATM_AES_v1_fixed_key\ASCAD_data\ASCAD_databases\ASCAD.h5'

print("="*70)
print("Checking ASCAD in ATMEGA Directory")
print("="*70)

if not os.path.exists(file_path):
    print(f"\n✗ File not found at: {file_path}")
else:
    file_size = os.path.getsize(file_path) / (1024**3)  # GB
    print(f"\n✓ File exists: {file_path}")
    print(f"  File size: {file_size:.2f} GB")

    with h5py.File(file_path, 'r') as f:
        print(f"\n  Keys in file: {list(f.keys())}")

        prof_shape = f['Profiling_traces/traces'].shape
        attack_shape = f['Attack_traces/traces'].shape

        print(f"\n  Profiling traces: {prof_shape}")
        print(f"  Attack traces: {attack_shape}")

        trace_length = prof_shape[1]
        print(f"  Trace length: {trace_length:,} points")

        if trace_length < 1000:
            print(f"\n  ✗ PREPROCESSED FILE ({trace_length} points)")
            print(f"     Still the 700-point version.")
        elif trace_length > 50000:
            print(f"\n  ✓✓✓ ORIGINAL FULL FILE! ({trace_length:,} points)")
            print(f"     This is the full-length ASCAD database!")
            print(f"     Perfect for few-shot SCA experiments!")
        else:
            print(f"\n  ⚠️ MODERATE LENGTH ({trace_length:,} points)")
            print(f"     This might be partially preprocessed.")

        # Check metadata
        metadata = f['Profiling_traces/metadata'][0]
        print(f"\n  Sample metadata fields: {metadata.dtype.names}")

print("\n" + "="*70)
