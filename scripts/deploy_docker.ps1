# Build + start VetAI Doctor UI via Docker Compose with persistent volumes.
# Usage (from repo root):
#   .\scripts\deploy_docker.ps1
#   .\scripts\deploy_docker.ps1 -Down

param(
    [switch]$Down,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker not found. Install Docker Desktop, or use .\scripts\run_doctor_ui_lan.ps1 instead."
}

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.deploy.example") {
        Copy-Item ".env.deploy.example" ".env"
        Write-Warning "Created .env from .env.deploy.example — fill in API keys and PGPASSWORD, then re-run."
        exit 1
    }
    Write-Error "Missing .env"
}

if ($Down) {
    docker compose down
    Write-Host "Stopped. Named volumes vetai_data / vetai_runs were kept (results preserved)."
    exit 0
}

$composeArgs = @("compose", "up", "-d", "--build")
& docker @composeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Doctor UI:  http://localhost:8501"
Write-Host "Persisted:  docker volumes vetai_data (SQLite) + vetai_runs (artifacts)"
Write-Host "Logs:       docker compose logs -f doctor-ui"
Write-Host "Stop:       .\scripts\deploy_docker.ps1 -Down"
