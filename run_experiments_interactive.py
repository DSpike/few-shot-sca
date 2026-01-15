"""
Run multiple experiments with visible progress
"""
import subprocess
import os
import time
import shutil

NUM_RUNS = 5
os.makedirs("experiment_results", exist_ok=True)

print("="*70)
print(f"RUNNING {NUM_RUNS} EXPERIMENTS WITH REAL-TIME OUTPUT")
print("="*70)

for run_id in range(1, NUM_RUNS + 1):
    print(f"\n{'='*70}")
    print(f"RUN {run_id}/{NUM_RUNS}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Run with real-time output (no buffering)
    result = subprocess.run(
        ["python", "-u", "comprehensive_few_shot_study.py"],  # -u for unbuffered
        cwd=os.getcwd()
    )

    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✓ Run {run_id} completed ({elapsed/60:.1f} minutes)")

        # Copy results
        if os.path.exists("few_shot_sca_results.csv"):
            dest = f"experiment_results/run_{run_id:02d}_results.csv"
            shutil.copy("few_shot_sca_results.csv", dest)
            print(f"  Saved to: {dest}")
    else:
        print(f"\n✗ Run {run_id} failed!")
        break

print("\n✓ All runs complete!")
print("Now run: python generate_plots.py")
