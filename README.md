# Reproducibility package: What Limits Memory-Forensic Malware Classification?

Code, split information, per-seed predictions and derived results for the paper
"What Limits Memory-Forensic Malware Classification? A Leakage-Controlled Re-evaluation of CIC-MalMem-2022".

## Contents
| Folder | Content |
|---|---|
| `code/` | All experiment, analysis, table and figure scripts (Python) |
| `results/<task>/seed_<s>/` | Per-seed metrics (`*.csv`) and test-set predictions with test indices (`*.npz`) for binary, 4-class and 16-class |
| `results/<task>/` | Summaries, ablations, controls, sensitivity, robustness |
| `results/lofo`, `results/loco` | Leave-one-family-out and leave-one-category-out results |
| `results/review/` | Shortcut protocols, Cover-Hart bound, timing, Hossain & Islam re-implementation |
| `results/malvol/` | MalVol-25 indicators extracted with Volatility 3 and 2.6, external, category and within-corpus tests |
| `RUN_MALVOL_*.bat` | One-click offline re-extraction of MalVol-25 indicators (Windows) |

## Data (not included; obtain from the original sources)
* **CIC-MalMem-2022**, Canadian Institute for Cybersecurity: https://www.unb.ca/cic/datasets/malmem-2022.html
  Place the three CSV files (`binarycic.csv`, `4clascic.csv`, `16classcicmalmem.csv`) in one folder and set
  `MEMHSE_DATA` to it. De-duplication is deterministic and done by the code (`memdgf_core.load_corpus`).
* **MalVol-25**, IEEE DataPort, doi:10.21227/kg5b-nf37. Only needed to re-extract the MalVol-25 indicators; the
  extracted indicators are included in `results/malvol/`.

## Environment
Python 3.14; see `requirements.txt`. A CUDA GPU is optional (GraphSAGE, FeatureGCN, XGBoost and the neural
re-implementations use it if present). Environment variables (all optional):
`MEMHSE_DATA` (CIC-MalMem-2022 folder), `MEMHSE_VOL3` (Volatility 3 executable), `MEMHSE_VOL3_SYMBOLS` (symbol folders,
`;`-separated), `MEMHSE_VOL2` (Volatility 2.6 standalone executable).

## Reproducing
```
cd code
python run_experiments.py --task 4class          # main comparison, ablations, sensitivity, robustness (also binary, 16class)
python sota_reimpl.py --task 4class              # seven re-implemented journal methods
python run_controls.py                           # grouping and structure controls
python leaky_protocols.py                        # shortcut protocols
python hossain_reimpl.py                         # highest published claim
python malvol_external.py --features malvol25_features.csv --tag _14pairs
python malvol_category.py ; python malvol_within.py
python aggregate_and_plot.py                     # tables and figures (set MEMTGF_PAPER to an output folder)
```
Seeds 42-46 are reported; seeds 7-9 were used for all design decisions. Test indices of every seed are stored in the
`test_idx` array of each `*.npz` file. Internal class names use the earlier working name MemTGF for MemHSE.

## Licence and citation
Code: MIT. Derived results: CC BY 4.0. The paper is under review; until it is published, please cite this repository (see `CITATION.cff`) together with the two datasets.
