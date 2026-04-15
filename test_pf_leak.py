import subprocess
import time
import os

print("Starting independent process...")

proc = subprocess.Popen(
    ["python", "-m", "http.server", "8000"],
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # IMPORTANT (Windows)
)

print("PID:", proc.pid)

time.sleep(3)

print("Exiting WITHOUT cleanup...")
