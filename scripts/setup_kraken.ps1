# Kraken lives in its own venv (Python 3.11, kraken 7.x) because it pins an older torch stack.
# Run from the repo root in PowerShell:  .\scripts\setup_kraken.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

uv venv --python 3.11 .venv-kraken
uv pip install --python .venv-kraken/Scripts/python.exe kraken
.venv-kraken/Scripts/python.exe -c "import importlib.metadata as m; print('kraken', m.version('kraken'))"

# the recognition model of the kraken-kurrent-xix system (configs/systems.yaml); the segmentation model is built in
New-Item -ItemType Directory -Force models/kraken | Out-Null
curl.exe -sSL -o models/kraken/kraken_german_finetuned.mlmodel "https://raw.githubusercontent.com/MGJamJam/htr_german_kurrent_model/main/Models/kraken_german_finetuned.mlmodel"

# smoke test, writing outside the verified data release (kraken 7.1 needs about 17 s a page on a CPU):
# .venv-kraken/Scripts/kraken.exe -x -i data/pages/abp/ABP_B5_1736_00517.jpg cache/smoke.xml segment -bl
