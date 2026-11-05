"""Offline extraction of CIC-MalMem-2022-style features from raw MalVol-25 memory dumps with Volatility 3.

CIC-MalMem-2022 summarises each dump by 55 indicators from nine Volatility 2 plugins (VolMemLyzer).
This script runs the corresponding Volatility 3 plugins on every dump, caches each plugin's JSON output,
and computes the same feature names. Kernel symbols must already be cached (run windows.info once per
dump while online); after that everything runs offline.

Differences from Volatility 2 (documented in the paper):
  psxview  Volatility 3 reports pslist, psscan, thrdscan and csrss only; the pspcid, session and deskthrd
           views do not exist, so those six features are left missing (median-imputed by the model).
  handles  'Port' handles are ALPC ports ('ALPC Port') on modern Windows; both names are counted.
Usage: python malvol_extract.py [--jobs 3]   (resumable: finished plugin outputs are reused)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DUMPS = ROOT / "Dataset All Files_0" / "Dataset All Files"
OUT = ROOT / "results" / "malvol"
RAW = OUT / "plugin_json_v2828"  # outputs of the Workbench vol.exe 2.28.2
PLUGINS = {  # feature prefix -> Volatility 3 plugin
    "pslist": "windows.pslist.PsList", "dlllist": "windows.dlllist.DllList", "handles": "windows.handles.Handles",
    "ldrmodules": "windows.malware.ldrmodules.LdrModules", "malfind": "windows.malware.malfind.Malfind",
    "psxview": "windows.malware.psxview.PsXView", "modules": "windows.modules.Modules",
    "svcscan": "windows.svcscan.SvcScan", "callbacks": "windows.callbacks.Callbacks",
}
# Volatility 3.2.28.2 as bundled with Volatility Workbench, with its offline Windows symbol pack (1,700+ kernel
# builds); it also opens the 32-bit PAE dumps that the pip build 2.28.0 cannot
VOL_EXE = os.environ.get("MEMHSE_VOL3", r"C:\Program Files\OSForensics\VolatilityWorkbench\vol.exe")
# offline symbol folders: the pip installation (full 800 MB Windows pack plus the kernel symbols fetched for these
# dumps while online) and Workbench's bundled pack
SYMBOL_DIRS = os.environ.get("MEMHSE_VOL3_SYMBOLS", ";".join([str(Path.home() / r"AppData\Roaming\Python\Python314\site-packages\volatility3\symbols"),
                                                             r"C:\Program Files\OSForensics\VolatilityWorkbench\symbols"]))


def run_plugin(dump: Path, prefix: str):
    out = RAW / f"{dump.stem}.{prefix}.json"
    if out.exists() and out.stat().st_size > 2:
        return out, "cached", 0.0
    t0 = time.time()
    p = subprocess.run([VOL_EXE, "-q", "-s", SYMBOL_DIRS, "-r", "json", "-f", str(dump), PLUGINS[prefix]], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    txt = p.stdout.strip()
    start = txt.find("[")
    if p.returncode != 0 or start < 0:
        (RAW / f"{dump.stem}.{prefix}.err").write_text(p.stderr[-4000:] + "\n" + txt[-2000:], encoding="utf-8")
        return out, f"FAILED rc={p.returncode}", time.time() - t0
    out.write_text(txt[start:], encoding="utf-8")
    (RAW / f"{dump.stem}.{prefix}.err").unlink(missing_ok=True)  # clear an error left by an earlier failed attempt
    return out, "ok", time.time() - t0


def rows(dump: Path, prefix: str) -> pd.DataFrame | None:
    f = RAW / f"{dump.stem}.{prefix}.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text(encoding="utf-8"))

    def flat(rs):  # tree renderers nest children under "__children"
        for r in rs:
            kids = r.pop("__children", [])
            yield r
            yield from flat(kids)
    df = pd.DataFrame(list(flat(data)))
    if len(df.columns) == 0 and prefix not in ("malfind", "callbacks"):
        return None  # e.g. svcscan finds no service records on some builds: the features are missing, not zero
    return df  # an empty malfind / callbacks result is a genuine zero count


def col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    raise KeyError(f"none of {names} in {list(df.columns)}")


def is_false(s):
    return s.astype(str).str.lower().isin(["false", "0"])


def features(dump: Path) -> dict:
    f = {"dump": dump.name, "label": int("WithVirus" in dump.name), "pair": dump.stem.split("-")[0].lstrip("$")}
    d = rows(dump, "pslist")
    if d is not None:
        f["pslist.nproc"] = len(d)
        f["pslist.nppid"] = col(d, "PPID").nunique()
        f["pslist.avg_threads"] = pd.to_numeric(col(d, "Threads"), errors="coerce").mean()
        f["pslist.nprocs64bit"] = int(d["Wow64"].astype(str).str.lower().eq("true").sum()) if "Wow64" in d else np.nan  # constant in CIC (dropped)
        f["pslist.avg_handlers"] = pd.to_numeric(col(d, "Handles"), errors="coerce").mean()
    d = rows(dump, "dlllist")
    if d is not None:
        f["dlllist.ndlls"] = len(d)
        f["dlllist.avg_dlls_per_proc"] = len(d) / max(col(d, "PID").nunique(), 1)
    d = rows(dump, "handles")
    if d is not None:
        t = col(d, "Type").astype(str)
        f["handles.nhandles"] = len(d)
        f["handles.avg_handles_per_proc"] = len(d) / max(col(d, "PID").nunique(), 1)
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
    if d is not None:
        f["malfind.ninjections"] = len(d)
        f["malfind.commitCharge"] = float(pd.to_numeric(col(d, "CommitCharge"), errors="coerce").sum()) if len(d) else 0.0
        f["malfind.protection"] = int(col(d, "Protection").astype(str).eq("PAGE_EXECUTE_READWRITE").sum()) if len(d) else 0
        f["malfind.uniqueInjections"] = col(d, "PID").nunique() if len(d) else 0
    d = rows(dump, "psxview")
    if d is not None:
        for v3, v2 in (("pslist", "pslist"), ("psscan", "eprocess_pool"), ("thrdscan", "ethread_pool"), ("csrss", "csrss_handles")):
            if v3 in d.columns:
                miss = is_false(d[v3])
                f[f"psxview.not_in_{v2}"] = int(miss.sum())
                f[f"psxview.not_in_{v2}_false_avg"] = float(miss.mean())
    d = rows(dump, "modules")
    if d is not None:
        f["modules.nmodules"] = len(d)
    d = rows(dump, "svcscan")
    if d is not None:
        t, s = col(d, "Type").astype(str), col(d, "State").astype(str)
        f["svcscan.nservices"] = len(d)
        f["svcscan.kernel_drivers"] = int(t.str.contains("KERNEL_DRIVER").sum())
        f["svcscan.fs_drivers"] = int(t.str.contains("FILE_SYSTEM_DRIVER").sum())
        f["svcscan.process_services"] = int(t.str.contains("WIN32_OWN_PROCESS").sum())
        f["svcscan.shared_process_services"] = int(t.str.contains("WIN32_SHARE_PROCESS").sum())
        f["svcscan.interactive_process_services"] = int(t.str.contains("INTERACTIVE").sum())
        f["svcscan.nactive"] = int(s.str.contains("RUNNING").sum())
    d = rows(dump, "callbacks")
    if d is not None:
        m = col(d, "Module").astype(str) if len(d) else pd.Series(dtype=str)
        f["callbacks.ncallbacks"] = len(d)
        f["callbacks.nanonymous"] = int(m.str.upper().isin(["UNKNOWN", "NONE", "N/A", ""]).sum())
        f["callbacks.ngeneric"] = int(col(d, "Type").astype(str).str.contains("Generic").sum()) if len(d) else 0
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    dumps = sorted(p for p in DUMPS.iterdir() if p.suffix.lower() in (".mem", ".raw", ".vmem"))
    tasks = [(d, k) for d in dumps for k in PLUGINS]
    print(f"{len(dumps)} dumps x {len(PLUGINS)} plugins = {len(tasks)} runs", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for (d, k), (out, status, sec) in zip(tasks, ex.map(lambda t: run_plugin(*t), tasks)):
            print(f"{d.stem:28s} {k:11s} {status} ({sec:.0f}s)", flush=True)
    feats = pd.DataFrame([features(d) for d in dumps])
    for target in (OUT / "malvol25_features.csv", DUMPS / "malvol25_features.csv"):  # results copy + copy next to the dumps
        feats.to_csv(target, index=False)
        print(f"wrote {target}: {feats.shape}")
    print(feats.isna().sum()[feats.isna().sum() > 0].to_string())


if __name__ == "__main__":
    main()
