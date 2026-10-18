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

$mutex = New-Object System.Threading.Mutex($false, "Global\LobDLRegisteredRun")
$mutexAcquired = $false
try {
  try {
    $mutexAcquired = $mutex.WaitOne(0)
  } catch [System.Threading.AbandonedMutexException] {
    $mutexAcquired = $true
  }
  if (-not $mutexAcquired) {
    Write-Output "Another registered Lob-DL runner already owns the global mutex."
    return
  }

  if (-not (Test-Path (Join-Path $Artifacts "selections/FROZEN.json"))) {
    throw "Frozen validation selections are required before post-test work."
  }

  & (Join-Path $PSScriptRoot "run_secondary.ps1") -DataRoot $DataRoot -Python $Python `
    -Artifacts $Artifacts -Replicates $Replicates
  & (Join-Path $PSScriptRoot "run_transfer_registered.ps1") `
    -RawRoot $RawRoot -FIDataRoot $DataRoot `
    -Python $Python -Artifacts $Artifacts -Replicates $Replicates
  & (Join-Path $PSScriptRoot "report_registered.ps1") `
    -DataRoot $DataRoot -Python $Python `
    -Artifacts $Artifacts -Replicates $Replicates

  & $Python scripts/report_external_anchor.py --artifacts $Artifacts
  if ($LASTEXITCODE -ne 0) {
    throw "External-anchor reporting failed with exit code $LASTEXITCODE"
  }
  & $Python scripts/generate_report_assets.py --artifacts $Artifacts
  if ($LASTEXITCODE -ne 0) {
    throw "Public-asset generation failed with exit code $LASTEXITCODE"
  }
  & $Python scripts/publish_registered_results.py --artifacts $Artifacts
  if ($LASTEXITCODE -ne 0) {
    throw "Registered-result publication failed with exit code $LASTEXITCODE"
  }
} finally {
  if ($mutexAcquired) {
    $mutex.ReleaseMutex()
  }
  $mutex.Dispose()
}
