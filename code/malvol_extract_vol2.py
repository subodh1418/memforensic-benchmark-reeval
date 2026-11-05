"""Offline extraction of CIC-MalMem-2022 features from MalVol-25 dumps with Volatility 2.6 (as CIC-MalMem-2022 itself).

Volatility 2.6 (standalone build shipped with Autopsy) has profiles up to Windows 10 build 14393, so only the Windows 7
and 8.1 pairs of MalVol-25 can be processed. Feature definitions follow VolMemLyzer, the extractor of CIC-MalMem-2022;
unlike Volatility 3, psxview provides all seven cross-view sources, so no indicator is missing.
Output: results/malvol/malvol25_features_vol2.csv and a copy next to the dumps.
Usage: python malvol_extract_vol2.py [--jobs 3]   (resumable)
"""
from __future__ import annotations

import argparse
import os
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DUMPS = ROOT / "Dataset All Files_0" / "Dataset All Files"
OUT = ROOT / "results" / "malvol"
RAW = OUT / "plugin_json_vol2"
VOL2 = os.environ.get("MEMHSE_VOL2", r"C:\Program Files\Autopsy-4.21.0\autopsy\Volatility\volatility_2.6_win64_standalone.exe")
PROFILE = {  # from windows.info (Volatility 3): architecture and build of each machine
    "Cerber": "Win7SP1x86", "PowerLoader": "Win7SP0x86", "WannaCry": "Win7SP1x64", "Win32.BlackWorm": "Win7SP1x64",
    "W32.MyDoom.A.": "Win7SP1x64", "LuckyLcoker": "Win81U1x64", "Dharma": "Win81U1x86",
}
PLUGINS = ["pslist", "dlllist", "handles", "ldrmodules", "malfind", "psxview", "modules", "svcscan", "callbacks"]


def pair_of(dump: Path) -> str:
    return dump.stem.rsplit("-", 1)[0].lstrip("$")


def run_plugin(dump: Path, plugin: str):
    out = RAW / f"{dump.stem}.{plugin}.json"
    if out.exists() and out.stat().st_size > 2:
        return "cached", 0.0
    t0 = time.time()
    p = subprocess.run([VOL2, "-f", str(dump), f"--profile={PROFILE[pair_of(dump)]}", "--output=json", plugin],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = p.stdout
    start = txt.find("{")
    if start < 0:
        (RAW / f"{dump.stem}.{plugin}.err").write_text(p.stderr[-4000:] + "\n" + txt[-2000:], encoding="utf-8")
        return f"FAILED rc={p.returncode}", time.time() - t0
    out.write_text(txt[start:], encoding="utf-8")
    (RAW / f"{dump.stem}.{plugin}.err").unlink(missing_ok=True)
    return "ok", time.time() - t0


def rows(dump: Path, plugin: str):
    f = RAW / f"{dump.stem}.{plugin}.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text(encoding="utf-8"))
    return pd.DataFrame(j.get("rows", []), columns=j.get("columns"))


def col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    raise KeyError(f"none of {names} in {list(df.columns)}")


def is_false(s):
    return s.astype(str).str.lower().isin(["false", "0"])


def features(dump: Path) -> dict:
    f = {"dump": dump.name, "label": int("WithVirus" in dump.name), "pair": pair_of(dump), "profile": PROFILE[pair_of(dump)]}
    d = rows(dump, "pslist")
    if d is not None and len(d):
        f["pslist.nproc"] = len(d)
        f["pslist.nppid"] = col(d, "PPID").nunique()
        f["pslist.avg_threads"] = pd.to_numeric(col(d, "Thds"), errors="coerce").mean()
        f["pslist.nprocs64bit"] = int(pd.to_numeric(col(d, "Wow64"), errors="coerce").fillna(0).astype(bool).sum())
        f["pslist.avg_handlers"] = pd.to_numeric(col(d, "Hnds"), errors="coerce").replace(-1, np.nan).mean()
    d = rows(dump, "dlllist")
    if d is not None and len(d):
        f["dlllist.ndlls"] = len(d)
        f["dlllist.avg_dlls_per_proc"] = len(d) / max(col(d, "Pid", "PID").nunique(), 1)
    d = rows(dump, "handles")
    if d is not None and len(d):
        t = col(d, "Type").astype(str)
        f["handles.nhandles"] = len(d)
        f["handles.avg_handles_per_proc"] = len(d) / max(col(d, "Pid", "PID").nunique(), 1)
        f["handles.nport"] = int(t.isin(["Port", "ALPC Port"]).sum())
        for k in ("File", "Event", "Desktop", "Key", "Thread", "Directory", "Semaphore", "Timer", "Section", "Mutant"):
            f[f"handles.n{k.lower()}"] = int((t == k).sum())
    d = rows(dump, "ldrmodules")
    if d is not None and len(d):
        for c, name in (("InLoad", "load"), ("InInit", "init"), ("InMem", "mem")):
            miss = is_false(col(d, c))
            f[f"ldrmodules.not_in_{name}"] = int(miss.sum())
            f[f"ldrmodules.not_in_{name}_avg"] = float(miss.mean())
    d = rows(dump, "malfind")
    if d is not None:  # an empty result is a genuine zero
        flags = col(d, "Flags").astype(str) if len(d) else pd.Series(dtype=str)
        cc = flags.str.extract(r"CommitCharge:\s*(\d+)")[0].astype(float) if len(d) else pd.Series(dtype=float)
        f["malfind.ninjections"] = len(d)
        f["malfind.commitCharge"] = float(cc.sum())
        f["malfind.protection"] = int(col(d, "Protection").astype(str).eq("PAGE_EXECUTE_READWRITE").sum()) if len(d) else 0
        f["malfind.uniqueInjections"] = col(d, "Pid", "PID").nunique() if len(d) else 0
    d = rows(dump, "psxview")
    if d is not None and len(d):
        for v2col, name in (("pslist", "pslist"), ("psscan", "eprocess_pool"), ("thrdproc", "ethread_pool"), ("pspcid", "pspcid_list"),
                            ("csrss", "csrss_handles"), ("session", "session"), ("deskthrd", "deskthrd")):
            miss = is_false(col(d, v2col))
            f[f"psxview.not_in_{name}"] = int(miss.sum())
            f[f"psxview.not_in_{name}_false_avg"] = float(miss.mean())
    d = rows(dump, "modules")
    if d is not None and len(d):
        f["modules.nmodules"] = len(d)
    d = rows(dump, "svcscan")
    if d is not None and len(d):
        t, s = col(d, "ServiceType", "Type").astype(str), col(d, "State").astype(str)
        f["svcscan.nservices"] = len(d)
        f["svcscan.kernel_drivers"] = int(t.str.contains("KERNEL_DRIVER").sum())
        f["svcscan.fs_drivers"] = int(t.str.contains("FILE_SYSTEM_DRIVER").sum())
        f["svcscan.process_services"] = int(t.str.contains("WIN32_OWN_PROCESS").sum())
        f["svcscan.shared_process_services"] = int(t.str.contains("WIN32_SHARE_PROCESS").sum())
        f["svcscan.interactive_process_services"] = int(t.str.contains("INTERACTIVE").sum())
        f["svcscan.nactive"] = int(s.str.contains("RUNNING").sum())
    d = rows(dump, "callbacks")
    if d is not None:
        f["callbacks.ncallbacks"] = len(d)
        f["callbacks.nanonymous"] = int(col(d, "Module").astype(str).str.upper().eq("UNKNOWN").sum()) if len(d) else 0
        f["callbacks.ngeneric"] = int(col(d, "Type").astype(str).eq("GenericKernelCallback").sum()) if len(d) else 0
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    dumps = sorted(p for p in DUMPS.iterdir() if p.suffix.lower() in (".mem", ".raw", ".vmem") and pair_of(p) in PROFILE)
    tasks = [(d, k) for d in dumps for k in PLUGINS]
    print(f"{len(dumps)} dumps x {len(PLUGINS)} plugins = {len(tasks)} runs (Volatility 2.6)", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for (d, k), (status, sec) in zip(tasks, ex.map(lambda t: run_plugin(*t), tasks)):
            print(f"{d.stem:28s} {k:11s} {status} ({sec:.0f}s)", flush=True)
    feats = pd.DataFrame([features(d) for d in dumps])
    for target in (OUT / "malvol25_features_vol2.csv", DUMPS / "malvol25_features_vol2.csv"):
        feats.to_csv(target, index=False)
        print(f"wrote {target}: {feats.shape}")
    n = feats.isna().sum()
    print(n[n > 0].to_string())


if __name__ == "__main__":
    main()
