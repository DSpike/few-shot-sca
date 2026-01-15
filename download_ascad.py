"""
Download Original ASCAD Database
"""
import urllib.request
import ssl
import os

url = "https://www.data.gouv.fr/fr/datasets/r/27b1f3ad-e641-41d9-9285-f6c3ea00e982"
output_file = "ASCAD_databases.zip"

print("="*70)
print("Downloading Original ASCAD Database")
print("="*70)
print(f"\nURL: {url}")
print(f"Output: {output_file}")
print(f"Size: ~4.6 GB (this will take a while...)\n")

# Create SSL context that doesn't verify certificates
ssl_context = ssl._create_unverified_context()

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100.0 / total_size)
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end='')

try:
    print("Starting download...")
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, output_file, download_progress)
    print(f"\n\n✓ Download complete!")
    print(f"  File saved to: {os.path.abspath(output_file)}")
    print(f"\nNext steps:")
    print(f"  1. Unzip ASCAD_databases.zip")
    print(f"  2. Find ASCAD.h5 inside")
    print(f"  3. Replace your current 700-point version")
except Exception as e:
    print(f"\n\n✗ Download failed: {e}")
    print(f"\nTry downloading manually from:")
    print(f"  https://www.data.gouv.fr/fr/datasets/ascad/")
