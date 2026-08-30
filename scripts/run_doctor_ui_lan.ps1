# Run Doctor UI on LAN with persistent local folders.
# Usage (from repo root):
#   .\scripts\run_doctor_ui_lan.ps1
#   .\scripts\run_doctor_ui_lan.ps1 -Port 8502

param(
    [int]$Port = 8501,
    [string]$Address = "0.0.0.0"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$dataDir = Join-Path $Root "doctor_ui\data"
$runsDir = Join-Path $Root "doctor_ui\runs"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
New-Item -ItemType Directory -Force -Path $runsDir | Out-Null

$env:VETAI_DATA_DIR = $dataDir
$env:VETAI_RUNS_DIR = $runsDir

if (Test-Path (Join-Path $Root ".env")) {
    Write-Host "Using .env from $Root"
} else {
    Write-Warning "No .env found - copy .env.deploy.example to .env and fill secrets."
}

Write-Host "Data dir: $dataDir"
Write-Host "Runs dir: $runsDir"
Write-Host ("Starting Streamlit on http://{0}:{1} (LAN: use this PC IP)" -f $Address, $Port)
python -m streamlit run doctor_ui/app.py --server.address $Address --server.port $Port
