param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [Parameter(Mandatory=$true)][string]$Python,
  [string]$Artifacts = "artifacts"
)
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repo

if (-not (Test-Path (Join-Path $Artifacts "selections/FROZEN.json"))) {
  throw "Frozen validation selections are required before final evaluation."
}
if (-not (Test-Path "reports/frozen_selections/PUBLISHED.json")) {
  throw "Publish and commit the frozen selections before final evaluation."
}

$runner = (Resolve-Path (Join-Path $PSScriptRoot "run_registered.ps1")).Path
$logRoot = Join-Path $repo "$Artifacts/logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$transcript = Join-Path $logRoot "scheduled-final.transcript.log"
Start-Transcript -Path $transcript -Append
try {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner `
    -DataRoot $DataRoot -Python $Python -Artifacts $Artifacts -Stage final
  if ($LASTEXITCODE -ne 0) {
    throw "Scheduled registered final evaluation failed with exit code $LASTEXITCODE"
  }
} finally {
  Stop-Transcript
}
