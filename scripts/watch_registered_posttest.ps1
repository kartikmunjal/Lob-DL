param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [Parameter(Mandatory=$true)][string]$RawRoot,
  [Parameter(Mandatory=$true)][string]$Python,
  [string]$Artifacts = "artifacts",
  [int]$Replicates = 2000
)
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repo

if (-not (Test-Path (Join-Path $Artifacts "selections/FROZEN.json"))) {
  throw "Frozen validation selections are required before post-test work."
}

$runner = (Resolve-Path (Join-Path $PSScriptRoot "run_posttest.ps1")).Path
$logRoot = Join-Path $repo "$Artifacts/logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$transcript = Join-Path $logRoot "scheduled-posttest.transcript.log"
Start-Transcript -Path $transcript -Append
try {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner `
    -DataRoot $DataRoot -RawRoot $RawRoot -Python $Python `
    -Artifacts $Artifacts -Replicates $Replicates
  if ($LASTEXITCODE -ne 0) {
    throw "Scheduled registered post-test workflow failed with exit code $LASTEXITCODE"
  }
} finally {
  Stop-Transcript
}
