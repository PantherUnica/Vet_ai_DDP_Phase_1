# Expose local Doctor UI on a public https URL (Cloudflare quick tunnel).
# Prerequisites: Streamlit already running on port 8501
#   .\scripts\run_doctor_ui_lan.ps1
# Then:
#   .\scripts\deploy_public_tunnel.ps1

param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$candidates = @(
    "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe",
    "$env:ProgramFiles\cloudflared\cloudflared.exe",
    "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe",
    "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe"
)
$cf = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $cf) {
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) { $cf = $cmd.Source }
}
if (-not $cf) {
    Write-Error "cloudflared not found. Install with: winget install --id Cloudflare.cloudflared -e"
}

Write-Host "Opening public tunnel to http://127.0.0.1:$Port ..."
Write-Host "Keep this window open. Share the https://....trycloudflare.com URL that appears."
Write-Host "Note: free quick tunnels are temporary; URL changes each restart."
& $cf tunnel --url "http://127.0.0.1:$Port"
