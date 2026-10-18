param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [Parameter(Mandatory=$true)][string]$Python,
  [string]$Artifacts = "artifacts"
)
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repo
$frozen = Join-Path $Artifacts "selections/FROZEN.json"
if (Test-Path $frozen) {
  return
}

$runner = (Resolve-Path (Join-Path $PSScriptRoot "run_registered.ps1")).Path

$logRoot = Join-Path $repo "$Artifacts/logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$transcript = Join-Path $logRoot "scheduled-select.transcript.log"
Start-Transcript -Path $transcript -Append
try {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner `
    -DataRoot $DataRoot -Python $Python -Artifacts $Artifacts -Stage select
  if ($LASTEXITCODE -ne 0) {
    throw "Scheduled registered selection failed with exit code $LASTEXITCODE"
  }
} finally {
  Stop-Transcript
}
