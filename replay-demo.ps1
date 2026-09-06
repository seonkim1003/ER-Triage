$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot\.venv\Scripts\python.exe" -m ertriage replay --run artifacts/baseline
    if ($LASTEXITCODE -ne 0) { throw 'Replay failed. See the error above.' }
} finally {
    Pop-Location
}
