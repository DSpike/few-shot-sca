"""
Quick check: What ASCAD file do you have?
"""
import h5py
import os

file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD.h5'

print("="*70)
print("ASCAD File Check")
print("="*70)

if not os.path.exists(file_path):
    print(f"\n✗ File not found at: {file_path}")
    print("\nYou need to download the original ASCAD database!")
    print("Download from: https://www.data.gouv.fr/fr/datasets/ascad/")
else:
    file_size = os.path.getsize(file_path) / (1024**3)  # GB
    print(f"\n✓ File exists: {file_path}")
    print(f"  File size: {file_size:.2f} GB")

    with h5py.File(file_path, 'r') as f:
        prof_shape = f['Profiling_traces/traces'].shape
        attack_shape = f['Attack_traces/traces'].shape

        print(f"\n  Profiling traces: {prof_shape}")
        print(f"  Attack traces: {attack_shape}")

        trace_length = prof_shape[1]

        print(f"\n  Trace length: {trace_length:,} points")

        if trace_length < 1000:
            print(f"\n  ✗ PREPROCESSED FILE ({trace_length} points)")
            print(f"     This is a filtered/reduced version.")
            print(f"     Original ASCAD has ~100,000 points per trace.")
            print(f"\n  ⚠️ You need the ORIGINAL ASCAD.h5 file!")
            print(f"     Download from: https://www.data.gouv.fr/fr/datasets/ascad/")
            print(f"     Look for ASCAD_databases.zip (~4.6 GB)")
        elif trace_length > 50000:
            print(f"\n  ✓ ORIGINAL FILE ({trace_length:,} points)")
            print(f"     This is the full-length original ASCAD database.")
            print(f"     Perfect for few-shot SCA experiments!")
        else:
            print(f"\n  ⚠️ UNKNOWN VERSION ({trace_length:,} points)")
            print(f"     This might be a custom preprocessed version.")

print("\n" + "="*70)
