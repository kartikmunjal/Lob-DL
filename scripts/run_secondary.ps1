param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [string]$Python = "python",
  [string]$Artifacts = "artifacts",
  [int]$Replicates = 2000
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$seeds = @(11, 23, 47, 91, 137)
$horizons = @(10, 20, 50, 100)

function Invoke-RegisteredPython {
  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Registered Python command failed with exit code ${LASTEXITCODE}: $Python $args"
  }
}

foreach ($horizon in $horizons) {
  Invoke-RegisteredPython scripts/run_leaky_b1.py --data $DataRoot `
    --horizon $horizon --artifacts $Artifacts
  foreach ($model in @("M1", "M2")) {
    foreach ($depth in @(1, 5)) {
      foreach ($seed in $seeds) {
        Invoke-RegisteredPython scripts/train_neural.py --data $DataRoot `
          --model $model --variant "depth$depth" --horizon $horizon `
          --seed $seed --output $Artifacts
      }
    }
  }
  foreach ($seed in $seeds) {
    Invoke-RegisteredPython scripts/train_m1_refit.py --data $DataRoot `
      --horizon $horizon --seed $seed --artifacts $Artifacts
  }
}

foreach ($horizon in $horizons) {
  foreach ($model in @(
    "B1_leaky", "M1_refit", "M1_depth1", "M1_depth5", "M2_depth1", "M2_depth5"
  )) {
    Invoke-RegisteredPython scripts/aggregate_results.py --artifacts $Artifacts `
      --model $model --horizon $horizon --replicates $Replicates
  }
  foreach ($variant in @("M1_refit", "M1_depth1", "M1_depth5")) {
    Invoke-RegisteredPython scripts/paired_comparison.py --artifacts $Artifacts `
      --system-a M1 --system-b $variant --horizon $horizon --replicates $Replicates
  }
  foreach ($variant in @("M2_depth1", "M2_depth5")) {
    Invoke-RegisteredPython scripts/paired_comparison.py --artifacts $Artifacts `
      --system-a M2 --system-b $variant --horizon $horizon --replicates $Replicates
  }
}
