param(
  [Parameter(Mandatory=$true)][string]$RawRoot,
  [Parameter(Mandatory=$true)][string]$FIDataRoot,
  [string]$Python = "python",
  [string]$Artifacts = "artifacts",
  [int]$Replicates = 2000
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$seeds = @(11, 23, 47, 91, 137)
$horizons = @(10, 20, 50, 100)
$systems = @("B0_majority", "B0_persistence", "B1", "B2", "M1", "M2")

function Invoke-RegisteredPython {
  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Registered Python command failed with exit code ${LASTEXITCODE}: $Python $args"
  }
}

foreach ($horizon in $horizons) {
  foreach ($model in @("B0_majority", "B0_persistence", "B1")) {
    Invoke-RegisteredPython scripts/run_transfer.py --raw $RawRoot `
      --fi-data $FIDataRoot --artifacts $Artifacts --model $model --horizon $horizon
  }
  foreach ($model in @("B2", "M1", "M2")) {
    foreach ($seed in $seeds) {
      Invoke-RegisteredPython scripts/run_transfer.py --raw $RawRoot `
        --fi-data $FIDataRoot --artifacts $Artifacts --model $model `
        --horizon $horizon --seed $seed
    }
  }
}

foreach ($horizon in $horizons) {
  foreach ($labels in @("fixed", "matched")) {
    foreach ($model in $systems) {
      Invoke-RegisteredPython scripts/aggregate_transfer.py --artifacts $Artifacts `
        --model $model --horizon $horizon --labels $labels --replicates $Replicates
    }
    for ($left = 0; $left -lt $systems.Count; $left++) {
      for ($right = $left + 1; $right -lt $systems.Count; $right++) {
        Invoke-RegisteredPython scripts/paired_transfer.py --artifacts $Artifacts `
          --system-a $systems[$left] --system-b $systems[$right] `
          --horizon $horizon --labels $labels --replicates $Replicates
      }
    }
  }
}
