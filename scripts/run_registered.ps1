param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [string]$Python = "python",
  [string]$Artifacts = "artifacts",
  [ValidateSet("select", "final", "all")][string]$Stage = "all"
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

if (-not (Test-Path "reports/phase1_timing.json")) {
  throw "Commit the registered Phase 1 timing report before starting search."
}

if ($Stage -in @("select", "all")) {
  # Stage 1: validation-only searches. No script here constructs test data.
  foreach ($horizon in $horizons) {
    foreach ($model in @("B1", "B2")) {
      Invoke-RegisteredPython scripts/train_baselines.py --data $DataRoot --model $model `
        --horizon $horizon --phase select --output $Artifacts
    }
  }
  foreach ($model in @("M1", "M2")) {
    foreach ($horizon in $horizons) {
      Invoke-RegisteredPython scripts/select_neural.py --data $DataRoot --model $model `
        --horizon $horizon --output $Artifacts
    }
  }

  # Final scripts reject an absent lock or any subsequent selection-file change.
  Invoke-RegisteredPython scripts/freeze_selections.py --artifacts $Artifacts
}

if ($Stage -eq "select") {
  return
}

# Stage 2: untouched test evaluation after every configuration is frozen.
foreach ($horizon in $horizons) {
  foreach ($rule in @("majority", "persistence")) {
    Invoke-RegisteredPython scripts/train_baselines.py --data $DataRoot --model B0 `
      --horizon $horizon --phase final --rule $rule --output $Artifacts
  }
  Invoke-RegisteredPython scripts/train_baselines.py --data $DataRoot --model B1 `
    --horizon $horizon --phase final --output $Artifacts
  Invoke-RegisteredPython scripts/train_baselines.py --data $DataRoot --model B1 `
    --feature-set shipped144 --horizon $horizon --phase final --output $Artifacts
  foreach ($seed in $seeds) {
    Invoke-RegisteredPython scripts/train_baselines.py --data $DataRoot --model B2 `
      --horizon $horizon --phase final --seed $seed --output $Artifacts
    Invoke-RegisteredPython scripts/train_baselines.py --data $DataRoot --model B2 `
      --feature-set shipped144 --horizon $horizon --phase final `
      --seed $seed --output $Artifacts
  }
}

foreach ($model in @("M1", "M2")) {
  foreach ($horizon in $horizons) {
    foreach ($seed in $seeds) {
      Invoke-RegisteredPython scripts/train_neural.py --data $DataRoot --model $model `
        --horizon $horizon --seed $seed --output $Artifacts
    }
  }
}

# The selected M1 fits serve as anchor fits when the anchor wins; otherwise run it.
foreach ($horizon in $horizons) {
  $selectionPath = Join-Path $Artifacts "selections/M1/h$horizon.json"
  $selected = (Get-Content $selectionPath -Raw | ConvertFrom-Json).selected.parameters
  $anchorWon = ($selected.learning_rate -eq 0.01) -and `
    ($selected.epsilon -eq 1.0) -and ($selected.dropout -eq 0.0)
  if (-not $anchorWon) {
    foreach ($seed in $seeds) {
      Invoke-RegisteredPython scripts/train_neural.py --data $DataRoot --model M1 `
        --variant anchor --horizon $horizon --seed $seed --output $Artifacts
    }
  }
}
