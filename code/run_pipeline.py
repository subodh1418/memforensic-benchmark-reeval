"""Runs all remaining experiment stages sequentially (one GPU user at a time) and keeps Windows awake
only while it runs (SetThreadExecutionState; released automatically when the process exits).
Every stage is resumable: finished seeds / families are skipped.
"""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
LOG = CODE.parent / "results" / "logs" / "pipeline.log"
NJ = "14"
STAGES = [
    ["run_experiments.py", "--task", "4class", "--n_jobs", NJ],
    ["run_experiments.py", "--task", "16class", "--n_jobs", NJ],
    ["run_experiments.py", "--task", "binary", "--n_jobs", NJ, "--skip_sensitivity"],
    ["sota_reimpl.py", "--task", "4class"],
    ["sota_reimpl.py", "--task", "16class"],
    ["sota_reimpl.py", "--task", "binary"],
    ["run_lofo.py", "--n_jobs", NJ],
    ["explain.py", "--task", "4class"],
    ["explain.py", "--task", "16class"],
    ["explain.py", "--task", "binary"],
    ["aggregate_and_plot.py"],
    ["dev_push_study.py", "4class"],
    ["dev_push_study.py", "16class"],
]

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    with open(LOG, "a", encoding="utf-8") as log:
        for st in STAGES:
            t = time.time()
            log.write(f"\n=== START {' '.join(st)} @ {time.ctime()}\n")
            log.flush()
            r = subprocess.run([sys.executable, "-W", "ignore", *st], cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
            log.write(f"=== END {' '.join(st)} rc={r.returncode} ({time.time() - t:.0f}s)\n")
            log.flush()
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
