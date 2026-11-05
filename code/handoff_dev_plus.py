"""Waits until the 4-class control stage has finished, stops the queued dev-seed control stage, then runs dev_plus.py.
Keeps Windows awake while it runs."""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
LOG = CODE.parent / "results" / "logs" / "controls.log"
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)
marker = "=== rc 0 for ['--tasks', '4class']"
while marker not in LOG.read_text(encoding="utf-8", errors="ignore"):
    time.sleep(20)
time.sleep(5)
# stop the queued dev-seed control stage (redundant with dev_plus)
ps = "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*run_controls.py*--seeds*7*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
subprocess.run(["powershell", "-NoProfile", "-Command", ps])
with open(CODE.parent / "results" / "logs" / "dev_plus.log", "w") as log:
    r = subprocess.run([sys.executable, "-W", "ignore", "dev_plus.py"], cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"=== rc {r.returncode}\n")
