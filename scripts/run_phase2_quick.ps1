param(
    [string]$Config = 'configs/triplet_quick.yaml',
    [string]$Python = '',
    [switch]$Mock
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($Mock) {
    if ($PSBoundParameters.ContainsKey('Config')) {
        throw 'Use either -Mock or -Config, not both.'
    }
    $Config = 'configs/triplet_mock.yaml'
}
if ([string]::IsNullOrWhiteSpace($Python)) {
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $Python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }
}

function Invoke-Phase2Module([string]$Module, [string[]]$ModuleArguments) {
    Write-Host "Running $Module"
    & $Python -m $Module @ModuleArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Module failed (exit code $LASTEXITCODE). Fix the reported error before continuing."
    }
}

Push-Location -LiteralPath $projectRoot
try {
    & $Python -c 'import sys; print(sys.version)'
    if ($LASTEXITCODE -ne 0) { throw 'Python unavailable. Install Python 3.10+ or pass -Python with a valid executable.' }
    if ($Mock) { Invoke-Phase2Module 'src.create_metric_mock_data' @() }
    $configArguments = @('--config', $Config)
    foreach ($module in @('src.build_metric_split', 'src.player_features', 'src.train_triplet',
                           'src.embed_players', 'src.evaluate', 'src.compare_baselines')) {
        Invoke-Phase2Module $module $configArguments
    }
    Write-Host 'Phase 2 completed. See the configured comparison CSV and split audit.'
}
finally {
    Pop-Location
}
