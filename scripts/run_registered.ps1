param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$seeds = @(11, 23, 47, 91, 137)
$horizons = @(10, 20, 50, 100)

foreach ($horizon in $horizons) {
  & $Python scripts/train_baselines.py --data $DataRoot --model B0 --horizon $horizon
  & $Python scripts/train_baselines.py --data $DataRoot --model B0 --horizon $horizon --rule persistence
  & $Python scripts/train_baselines.py --data $DataRoot --model B1 --horizon $horizon
  foreach ($seed in $seeds) {
    & $Python scripts/train_baselines.py --data $DataRoot --model B2 --horizon $horizon --seed $seed
  }
}

foreach ($model in @("M1", "M2")) {
  foreach ($horizon in $horizons) {
    foreach ($seed in $seeds) {
      & $Python scripts/train_neural.py --data $DataRoot --model $model --horizon $horizon --seed $seed
    }
  }
}
