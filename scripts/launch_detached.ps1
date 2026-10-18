param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [Parameter(Mandatory=$true)][string]$Python,
  [string]$Artifacts = "artifacts",
  [ValidateSet("select", "final", "all")][string]$Stage = "select"
)
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$logRoot = Join-Path $repo "$Artifacts/logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$stdout = Join-Path $logRoot "$Stage.stdout.log"
$stderr = Join-Path $logRoot "$Stage.stderr.log"
$arguments = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $PSScriptRoot "run_registered.ps1"),
  "-DataRoot", $DataRoot,
  "-Python", $Python,
  "-Artifacts", $Artifacts,
  "-Stage", $Stage
)
$process = Start-Process powershell.exe -ArgumentList $arguments `
  -WorkingDirectory $repo -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr -PassThru
@{
  pid = $process.Id
  stage = $Stage
  stdout = $stdout
  stderr = $stderr
} | ConvertTo-Json
