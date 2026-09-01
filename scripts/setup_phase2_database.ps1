# One-shot PostgreSQL setup for Phase 1 grounding + Phase 2 knowledge atoms
# Run from repo root in PowerShell:
#   .\scripts\setup_phase2_database.ps1
#   .\scripts\setup_phase2_database.ps1 -DemoOnly

param(
    [switch]$DemoOnly,
    [switch]$ForceDemo
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$args = @("scripts/setup_phase2_database.py")
if ($DemoOnly) { $args += "--demo-only" }
if ($ForceDemo) { $args += "--force-demo" }

python @args
exit $LASTEXITCODE
