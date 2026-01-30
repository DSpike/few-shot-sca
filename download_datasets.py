"""
Download SCA Datasets
======================
Downloads ASCADv2 dataset (CHES CTF already downloaded).
ASCAD v1 must be downloaded separately (see README).

Usage:
    python download_datasets.py
"""

import os
import sys
import ssl

# ASCADv2: STM32F303 ARM Cortex-M4, hardened masked AES (shuffling + affine masking)
# Download from French government data portal (data.gouv.fr)
DATASETS = {
    'ascadv2': {
        'urls': [
            'https://object.files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5',
        ],
        'filename': 'ascadv2-extracted.h5',
        'description': 'ASCADv2 (STM32F303 ARM Cortex-M4, hardened masked AES, 800K traces)',
        'reference': 'ANSSI, "ASCADv2: Side Channel Analysis and Deep Learning v2", ePrint 2021/592'
    },
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')


def download_file(urls, filepath, description=''):
    """Download a file, trying multiple URLs with fallback."""
    import urllib.request

    print(f"\nDownloading: {description}")
    print(f"  Destination: {filepath}")

    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  File already exists ({size_mb:.1f} MB). Skipping.")
        return True

    # Create SSL context that doesn't verify (some servers have cert issues)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for i, url in enumerate(urls):
        print(f"  Trying URL {i+1}/{len(urls)}: {url}")

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

            with urllib.request.urlopen(req, context=ctx, timeout=120) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                block_size = 1024 * 1024  # 1 MB chunks

                with open(filepath, 'wb') as f:
                    while True:
                        block = response.read(block_size)
                        if not block:
                            break
                        f.write(block)
                        downloaded += len(block)

                        if total_size > 0:
                            percent = min(100, downloaded * 100 / total_size)
                            mb_down = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                        else:
                            mb_down = downloaded / (1024 * 1024)
                            sys.stdout.write(f"\r  Downloaded: {mb_down:.1f} MB")
                        sys.stdout.flush()

            print()

            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb < 1:
                print(f"  WARNING: File too small ({size_mb:.2f} MB), likely not valid. Removing.")
                os.remove(filepath)
                continue

            print(f"  Downloaded successfully ({size_mb:.1f} MB)")
            return True

        except Exception as e:
            print(f"  ERROR: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            continue

    print(f"  All URLs failed for {description}")
    return False


def verify_dataset(filepath, dataset_name):
    """Verify downloaded dataset by checking HDF5 structure."""
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            # Print all top-level groups to understand structure
            print(f"  Verifying {dataset_name}...")
            print(f"    Top-level groups: {list(f.keys())}")

            if 'Profiling_traces' in f:
                prof_shape = f['Profiling_traces/traces'].shape
                atk_shape = f['Attack_traces/traces'].shape
                print(f"    Profiling traces: {prof_shape}")
                print(f"    Attack traces:    {atk_shape}")
            else:
                # ASCADv2 might have different structure
                for key in f.keys():
                    if hasattr(f[key], 'shape'):
                        print(f"    {key}: {f[key].shape}")
                    else:
                        print(f"    {key}: {list(f[key].keys())}")
            return True
    except Exception as e:
        print(f"  Verification FAILED for {dataset_name}: {e}")
        return False


def main():
    print("=" * 70)
    print("SCA Dataset Downloader")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Check existing datasets
    ascad_path = os.path.join(os.path.dirname(OUTPUT_DIR),
                              'ASCAD_data', 'ASCAD_data', 'ASCAD_databases', 'ASCAD.h5')
    ches_path = os.path.join(OUTPUT_DIR, 'ches_ctf.h5')

    print(f"\nExisting datasets:")
    print(f"  ASCAD v1:  {'FOUND' if os.path.exists(ascad_path) else 'NOT FOUND'} ({ascad_path})")
    print(f"  CHES CTF:  {'FOUND' if os.path.exists(ches_path) else 'NOT FOUND'} ({ches_path})")

    # Download ASCADv2
    success_count = 0
    for ds_name, ds in DATASETS.items():
        filepath = os.path.join(OUTPUT_DIR, ds['filename'])

        ok = download_file(ds['urls'], filepath, ds['description'])
        if ok:
            verified = verify_dataset(filepath, ds_name)
            if verified:
                success_count += 1
                print(f"  Reference: {ds['reference']}")

    print(f"\n{'=' * 70}")
    print(f"Downloaded {success_count}/{len(DATASETS)} datasets successfully")
    print(f"{'=' * 70}")

    if success_count < len(DATASETS):
        print("\nManual download:")
        print("  ASCADv2:")
        print("    wget https://object.files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5")
        print("    Place in datasets/ folder")
        print("    Source: https://www.data.gouv.fr/datasets/ascadv2")

    print("\nDataset locations for experiments:")
    print(f"  ASCAD v1:  {ascad_path}")
    print(f"  CHES CTF:  {ches_path}")
    print(f"  ASCADv2:   {os.path.join(OUTPUT_DIR, 'ascadv2-extracted.h5')}")


if __name__ == '__main__':
    main()
