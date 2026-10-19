param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [Parameter(Mandatory=$true)][string]$Python,
  [string]$Artifacts = "artifacts",
  [int]$MinFreeMemoryMB = 4096
)
$ErrorActionPreference = "Stop"
$freeMemory = (& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
  Select-Object -First 1).Trim()
if ([int]$freeMemory -lt $MinFreeMemoryMB) {
  Write-Output "Selection deferred: GPU has $freeMemory MiB free; requires $MinFreeMemoryMB MiB."
  return
}

$watcher = (Resolve-Path (Join-Path $PSScriptRoot "watch_registered_select.ps1")).Path
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $watcher `
  -DataRoot $DataRoot -Python $Python -Artifacts $Artifacts
if ($LASTEXITCODE -ne 0) {
  throw "GPU-gated selection watcher failed with exit code $LASTEXITCODE"
}
