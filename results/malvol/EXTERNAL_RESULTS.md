# MalVol-25 external test (2026-11-04 10:18)

Dumps with features: 30 (15 infected, 15 clean), pairs: 15

## Train on CIC-MalMem-2022, test on MalVol-25

| model | n_dumps | detection | fpr | auc | pair_acc | n_pairs |
|---|---|---|---|---|---|---|
| LightGBM | 30 | 0.0 | 0.0 | 0.5 | 0.333 | 15 |
| Random forest | 30 | 0.133 | 0.0 | 0.558 | 0.467 | 15 |
| Extra trees | 30 | 0.0 | 0.067 | 0.502 | 0.4 | 15 |
| MemHSE | 30 | 0.0 | 0.0 | 0.502 | 0.4 | 15 |

- detection / fpr: threshold 0.5; auc: threshold-free; pair_acc: share of same-machine pairs where the infected dump scores higher.

## Per-dump malware probability

| dump | pair | label | LightGBM | Random forest | Extra trees | MemHSE |
|---|---|---|---|---|---|---|
| $LuckyLcoker-NoVirus.mem | LuckyLcoker | 0 | 0.008 | 0.464 | 0.26 | 0.003 |
| $LuckyLcoker-WithVirus.mem | LuckyLcoker | 1 | 0.009 | 0.474 | 0.264 | 0.003 |
| Cerber-NoVirus.mem | Cerber | 0 | 0.0 | 0.162 | 0.126 | 0.0 |
| Cerber-WithVirus.mem | Cerber | 1 | 0.0 | 0.164 | 0.126 | 0.0 |
| DeriaLock-NoVirus.raw | DeriaLock | 0 | 0.009 | 0.428 | 0.228 | 0.003 |
| DeriaLock-WithVirus.mem | DeriaLock | 1 | 0.007 | 0.402 | 0.21 | 0.003 |
| Dharma-NoVirus.mem | Dharma | 0 | 0.018 | 0.486 | 0.35 | 0.006 |
| Dharma-WithVirus.mem | Dharma | 1 | 0.018 | 0.506 | 0.342 | 0.006 |
| DLLHijacking-NoVirus.mem | DLLHijacking | 0 | 0.015 | 0.41 | 0.378 | 0.005 |
| DLLHijacking-WithVirus.mem | DLLHijacking | 1 | 0.008 | 0.402 | 0.238 | 0.003 |
| GandCrab-NoVirus.raw | GandCrab | 0 | 0.008 | 0.402 | 0.226 | 0.003 |
| GandCrab-WithVirus.raw | GandCrab | 1 | 0.008 | 0.402 | 0.23 | 0.003 |
| GoldenEye-NoVirus.raw | GoldenEye | 0 | 0.004 | 0.374 | 0.202 | 0.001 |
| GoldenEye-WithVirus.raw | GoldenEye | 1 | 0.017 | 0.558 | 0.276 | 0.006 |
| InfinityCrypt-NoVirus.raw | InfinityCrypt | 0 | 0.01 | 0.432 | 0.234 | 0.004 |
| InfinityCrypt-WithVirus.raw | InfinityCrypt | 1 | 0.007 | 0.404 | 0.214 | 0.002 |
| Locky.AZ-NoVirus.mem | Locky.AZ | 0 | 0.008 | 0.388 | 0.188 | 0.003 |
| Locky.AZ-WithVirus.mem | Locky.AZ | 1 | 0.016 | 0.422 | 0.24 | 0.006 |
| PowerLoader-NoVirus.mem | PowerLoader | 0 | 0.0 | 0.2 | 0.132 | 0.0 |
| PowerLoader-WithVirus.mem | PowerLoader | 1 | 0.0 | 0.022 | 0.11 | 0.0 |
| RedTail-NoVirus.mem | RedTail | 0 | 0.015 | 0.41 | 0.378 | 0.005 |
| RedTail-WithVirus.mem | RedTail | 1 | 0.02 | 0.456 | 0.228 | 0.007 |
| SporaRansomware-NoVirus.mem | SporaRansomware | 0 | 0.008 | 0.398 | 0.206 | 0.003 |
| SporaRansomware-WithVirus.mem | SporaRansomware | 1 | 0.006 | 0.374 | 0.208 | 0.002 |
| W32.MyDoom.A.-NoVirus.mem | W32.MyDoom.A. | 0 | 0.13 | 0.264 | 0.524 | 0.262 |
| W32.MyDoom.A.-WithVirus.mem | W32.MyDoom.A. | 1 | 0.13 | 0.264 | 0.484 | 0.049 |
| WannaCry-NoVirus.raw | WannaCry | 0 | 0.006 | 0.358 | 0.194 | 0.002 |
| WannaCry-WithVirus.raw | WannaCry | 1 | 0.005 | 0.412 | 0.2 | 0.002 |
| Win32.BlackWorm-NoVirus.mem | Win32.BlackWorm | 0 | 0.004 | 0.476 | 0.19 | 0.002 |
| Win32.BlackWorm-WithVirus.mem | Win32.BlackWorm | 1 | 0.004 | 0.45 | 0.174 | 0.001 |

## Most shifted features (share of MalVol dumps outside the CIC 1st-99th percentile)

| feature | available | share_outside_cic_p1_p99 | cic_median | malvol_median |
|---|---|---|---|---|
| callbacks.ncallbacks | True | 1.0 | 87.0 | 183.0 |
| callbacks.ngeneric | True | 1.0 | 8.0 | 17.5 |
| svcscan.process_services | True | 0.867 | 24.0 | 121.0 |
| svcscan.shared_process_services | True | 0.867 | 118.0 | 458.5 |
| handles.avg_handles_per_proc | True | 0.867 | 248.298 | 403.481 |
| svcscan.nservices | True | 0.867 | 392.0 | 1323.0 |
| svcscan.fs_drivers | True | 0.867 | 26.0 | 70.5 |
| svcscan.kernel_drivers | True | 0.867 | 222.0 | 676.5 |
| modules.nmodules | True | 0.867 | 138.0 | 151.5 |
| svcscan.nactive | True | 0.867 | 122.0 | 403.0 |
| handles.nsection | True | 0.833 | 232.0 | 1161.0 |
| psxview.not_in_csrss_handles | True | 0.833 | 5.0 | 52.0 |
| handles.nevent | True | 0.8 | 3183.0 | 10918.5 |
| handles.nfile | True | 0.767 | 840.0 | 1987.5 |
| psxview.not_in_eprocess_pool | True | 0.767 | 0.0 | 1.5 |
