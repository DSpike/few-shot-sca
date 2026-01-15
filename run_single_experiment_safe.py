"""
Safe single experiment runner - ensures no other Python processes are using GPU
"""
import subprocess
import sys
import time

def check_gpu_python_processes():
    """Check if any Python processes are using GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                               '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split('\n')
        python_procs = [l for l in lines if 'python' in l.lower() and l.strip()]
        return len(python_procs)
    except:
        return 0

print("="*70)
print("SAFE EXPERIMENT RUNNER")
print("="*70)

# Check for competing processes
print("\nChecking for competing GPU processes...")
num_procs = check_gpu_python_processes()

if num_procs > 0:
    print(f"⚠ WARNING: {num_procs} Python process(es) already using GPU!")
    print("Please kill them first to avoid slowdown.")
    print("Run: taskkill //F //IM python.exe")
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(1)

print("\n✓ No competing processes detected")

# Get optional seed from command line
seed = sys.argv[1] if len(sys.argv) > 1 else None
if seed:
    print(f"\nStarting experiment with seed={seed}...")
else:
    print("\nStarting experiment (random seed)...")
print("="*70 + "\n")

# Run the main script with unbuffered output and optional seed
cmd = [sys.executable, "-u", "comprehensive_few_shot_study.py"]
if seed:
    cmd.extend(["--seed", seed])

result = subprocess.run(cmd, cwd=".")

print("\n" + "="*70)
if result.returncode == 0:
    print("✓ Experiment completed successfully!")
else:
    print("✗ Experiment failed!")
print("="*70)

sys.exit(result.returncode)
