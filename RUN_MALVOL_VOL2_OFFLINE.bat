@echo off
REM Offline Volatility 2.6 extraction for the Windows 7 / 8.1 pairs of MalVol-25, then the external tests:
REM   _vol2            Volatility 2 features, all pairs Volatility 2 can read
REM   _vol3_samepairs  Volatility 3 features on the six pairs both tools read
REM   _vol2_samepairs  Volatility 2 features on the same six pairs
REM Results in results\malvol\external_summary_*.csv ; CSV also next to the dumps (malvol25_features_vol2.csv)
cd /d "%~dp0code"
set PAIRS=Cerber PowerLoader WannaCry Win32.BlackWorm LuckyLcoker Dharma
python -u -W ignore malvol_extract_vol2.py --jobs 3 > ..\results\logs\malvol_extract_vol2.log 2>&1
python -u -W ignore malvol_external.py --features malvol25_features_vol2.csv --tag _vol2 --exclude > ..\results\logs\malvol_external_vol2.log 2>&1
python -u -W ignore malvol_external.py --features malvol25_features.csv --tag _vol3_samepairs --pairs %PAIRS% > ..\results\logs\malvol_external_vol3_samepairs.log 2>&1
python -u -W ignore malvol_external.py --features malvol25_features_vol2.csv --tag _vol2_samepairs --pairs %PAIRS% > ..\results\logs\malvol_external_vol2_samepairs.log 2>&1
echo VOL2 PIPELINE DONE > ..\results\logs\malvol_vol2_done.txt
