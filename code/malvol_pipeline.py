"""Unattended, offline MalVol-25 pipeline: wait for (or run) feature extraction, then the external test, then a report.

  python malvol_pipeline.py --wait   wait until the running extraction has written malvol25_features.csv, then evaluate
  python malvol_pipeline.py          run extraction (resumable) and evaluation now
Writes results/malvol/EXTERNAL_RESULTS.md (human-readable) and logs to results/logs/malvol_pipeline.log.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT = ROOT / "results" / "malvol"
LOG = ROOT / "results" / "logs" / "malvol_pipeline.log"
FEATS = OUT / "malvol25_features.csv"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(script, *args):
    log(f"start {script} {' '.join(args)}")
    with LOG.open("a", encoding="utf-8") as f:
        rc = subprocess.run([sys.executable, "-u", "-W", "ignore", str(CODE / script), *args], cwd=CODE, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"end {script} rc={rc}")
    return rc


def md(df):  # Markdown table without the optional 'tabulate' package (works offline)
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join("" if v != v else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def report():
    import pandas as pd
    pd.DataFrame.to_markdown = lambda self, index=False: md(self)
    lines = [f"# MalVol-25 external test ({datetime.now():%Y-%m-%d %H:%M})", ""]
    f = pd.read_csv(FEATS)
    lines += [f"Dumps with features: {len(f)} ({int(f.label.sum())} infected, {int((1 - f.label).sum())} clean), "
              f"pairs: {f.pair.nunique()}", ""]
    s = OUT / "external_summary.csv"
    if s.exists():
        lines += ["## Train on CIC-MalMem-2022, test on MalVol-25", "", pd.read_csv(s).round(3).to_markdown(index=False), "",
                  "- detection / fpr: threshold 0.5; auc: threshold-free; pair_acc: share of same-machine pairs where the infected dump scores higher.", ""]
    sc = OUT / "external_scores.csv"
    if sc.exists():
        lines += ["## Per-dump malware probability", "", pd.read_csv(sc).round(3).to_markdown(index=False), ""]
    sh = OUT / "shift_report.csv"
    if sh.exists():
        lines += ["## Most shifted features (share of MalVol dumps outside the CIC 1st-99th percentile)", "",
                  pd.read_csv(sh).sort_values("share_outside_cic_p1_p99", ascending=False).head(15).round(3).to_markdown(index=False), ""]
    errs = sorted((OUT / "plugin_json_v2828").glob("*.err"))
    if errs:
        lines += ["## Plugin runs that failed", ""] + [f"- {e.stem}" for e in errs] + [""]
    (OUT / "EXTERNAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"report written: {OUT / 'EXTERNAL_RESULTS.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="wait for the extraction that is already running")
    a = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.wait:
        log("waiting for the running extraction to write malvol25_features.csv")
        xlog = ROOT / "results" / "logs" / "malvol_extract.log"
        while not (FEATS.exists() and FEATS.stat().st_mtime > t0):
            idle = time.time() - max(xlog.stat().st_mtime if xlog.exists() else t0, t0)
            if idle > 20 * 60:  # the session that started the extraction may have closed: take over (resumable)
                log(f"no extraction progress for {idle / 60:.0f} min; running extraction myself")
                break
            time.sleep(60)
    if not (FEATS.exists() and FEATS.stat().st_mtime > t0):
        run("malvol_extract.py", "--jobs", "3")
    if FEATS.exists():
        run("malvol_external.py")
        report()
    log("pipeline finished")


if __name__ == "__main__":
    main()
