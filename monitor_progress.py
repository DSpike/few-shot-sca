"""
Monitor experiment progress
"""
import time
import os
from datetime import datetime

csv_file = 'few_shot_sca_results.csv'

print("Monitoring experiment progress...")
print("Press Ctrl+C to stop\n")

last_size = 0
last_time = None

try:
    while True:
        if os.path.exists(csv_file):
            current_size = os.path.getsize(csv_file)
            current_time = os.path.getmtime(csv_file)

            if current_size != last_size or current_time != last_time:
                timestamp = datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
                print(f"[{datetime.now().strftime('%H:%M:%S')}] File updated: {current_size} bytes (modified at {timestamp})")
                last_size = current_size
                last_time = current_time

                # Show last few lines
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 5:
                        print("  Last 3 results:")
                        for line in lines[-3:]:
                            print(f"    {line.strip()}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for {csv_file}...")

        time.sleep(10)  # Check every 10 seconds

except KeyboardInterrupt:
    print("\n\nMonitoring stopped")
