"""
Quick progress monitor - shows GPU usage and checks for new results
Run this in a separate terminal while your main script runs
"""
import time
import os
import subprocess
from datetime import datetime

print("Monitoring experiment progress...")
print("Press Ctrl+C to stop\n")

csv_file = 'few_shot_sca_results.csv'
last_size = 0
last_mtime = 0

try:
    while True:
        now = datetime.now().strftime('%H:%M:%S')

        # Check GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                   '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=5)
            gpu_info = result.stdout.strip()
            print(f"[{now}] GPU: {gpu_info}", end="")
        except:
            print(f"[{now}] GPU: [unavailable]", end="")

        # Check results file
        if os.path.exists(csv_file):
            size = os.path.getsize(csv_file)
            mtime = os.path.getmtime(csv_file)

            if size != last_size or mtime != last_mtime:
                print(f" | CSV: {size} bytes (UPDATED!)")

                # Show last few lines
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        print(f"    Latest results:")
                        for line in lines[-3:]:
                            print(f"      {line.strip()}")

                last_size = size
                last_mtime = mtime
            else:
                print(f" | CSV: {size} bytes (no change)")
        else:
            print(" | CSV: Not created yet")

        time.sleep(10)

except KeyboardInterrupt:
    print("\n\nMonitoring stopped")
