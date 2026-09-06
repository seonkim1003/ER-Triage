param(
    [string]$Run = 'artifacts/full-random-42',
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $viewerArgs = @('-m', 'ertriage', 'dashboard', '--run', $Run, '--port', $Port)
    $runName = Split-Path $Run -Leaf
    $evaluationFile = Join-Path $PSScriptRoot "artifacts/early-warning-$runName/early_warning.json"
    if (Test-Path -LiteralPath $evaluationFile) {
        $viewerArgs += @('--evaluation', $evaluationFile)
    }
    Write-Host "Open http://127.0.0.1:$Port after the server starts. Press Ctrl+C to stop."
    & "$PSScriptRoot\.venv\Scripts\python.exe" @viewerArgs
    if ($LASTEXITCODE -ne 0) { throw 'Dashboard failed. See the error above.' }
} finally {
    Pop-Location
}
