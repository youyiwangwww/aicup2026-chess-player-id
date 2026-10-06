param(
    [string]$Config = 'configs/real_quick.yaml',
    [string]$Python = '',
    [switch]$Mock
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($Mock) {
    if ($PSBoundParameters.ContainsKey('Config')) { throw 'Use either -Mock or -Config, not both.' }
    $Config = 'configs/phase25_mock.yaml'
}
if (-not $Mock -and -not $PSBoundParameters.ContainsKey('Config')) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'data\training\train_A.csv'))) {
        Write-Host 'Please place the official train_A.csv at data/training/train_A.csv'
        exit 1
    }
}
if ([string]::IsNullOrWhiteSpace($Python)) {
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $Python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }
}
Push-Location -LiteralPath $projectRoot
try {
    $runArguments = @('-m', 'src.run_experiment', '--config', $Config)
    if ($Mock) { $runArguments += '--create-mock' }
    & $Python @runArguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment stopped (exit code $LASTEXITCODE)." }
}
finally { Pop-Location }
