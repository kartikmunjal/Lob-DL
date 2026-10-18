param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [string]$Python = "python",
  [string]$Artifacts = "artifacts",
  [int]$Replicates = 2000
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$horizons = @(10, 20, 50, 100)

function Invoke-RegisteredPython {
  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Registered Python command failed with exit code ${LASTEXITCODE}: $Python $args"
  }
}
$systems = @(
  "B0:majority",
  "B0:persistence",
  "B1",
  "B1_shipped144",
  "B2",
  "B2_shipped144",
  "M1",
  "M1_anchor",
  "M2"
)

foreach ($horizon in $horizons) {
  Invoke-RegisteredPython scripts/aggregate_results.py --artifacts $Artifacts --model B0 `
    --variant majority --horizon $horizon --replicates $Replicates
  Invoke-RegisteredPython scripts/aggregate_results.py --artifacts $Artifacts --model B0 `
    --variant persistence --horizon $horizon --replicates $Replicates
  foreach ($model in @(
    "B1", "B1_shipped144", "B2", "B2_shipped144", "M1", "M1_anchor", "M2"
  )) {
    Invoke-RegisteredPython scripts/aggregate_results.py --artifacts $Artifacts --model $model `
      --horizon $horizon --replicates $Replicates
  }
  for ($left = 0; $left -lt $systems.Count; $left++) {
    for ($right = $left + 1; $right -lt $systems.Count; $right++) {
      Invoke-RegisteredPython scripts/paired_comparison.py --artifacts $Artifacts `
        --system-a $systems[$left] --system-b $systems[$right] `
        --horizon $horizon --replicates $Replicates
    }
  }
}

Invoke-RegisteredPython scripts/mechanism_analysis.py --data $DataRoot `
  --artifacts $Artifacts --replicates $Replicates
