@echo off
REM Offline MalVol-25 pipeline: feature extraction (resumable) + external test + report.
REM No internet needed: uses Volatility Workbench's vol.exe and its bundled symbol tables.
REM Results: results\malvol\EXTERNAL_RESULTS.md   CSV: Dataset All Files_0\Dataset All Files\malvol25_features.csv
cd /d "%~dp0code"
python -u -W ignore malvol_pipeline.py
echo.
echo Done. Open results\malvol\EXTERNAL_RESULTS.md
pause
