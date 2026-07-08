$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root ".venv313\Scripts\python.exe"

if (Test-Path -LiteralPath $venvPython) {
  $python = $venvPython
} else {
  $python = "python"
}

Write-Host "Starting ML Learning Lab at http://127.0.0.1:4173/"
& $python (Join-Path $root "backend_server.py") --host 127.0.0.1 --port 4173
