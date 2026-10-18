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
$active = Get-CimInstance Win32_Process | Where-Object {
  ($_.CommandLine -like "*run_registered.ps1*" -and $_.CommandLine -like "*-Stage select*") `
    -or $_.CommandLine -like "*select_neural.py*"
}
if ($active) {
  return
}

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
