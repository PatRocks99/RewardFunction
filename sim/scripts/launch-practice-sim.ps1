[CmdletBinding()]
param()

$simRoot = Resolve-Path (Join-Path $PSScriptRoot "..\windows_practice\autodrive_simulator")
$exe = Join-Path $simRoot "AutoDRIVE Simulator.exe"

if (-not (Test-Path -LiteralPath $exe)) {
    throw "Simulator executable not found at $exe. Re-download/extract the practice Windows build first."
}

Start-Process -FilePath $exe -WorkingDirectory $simRoot
