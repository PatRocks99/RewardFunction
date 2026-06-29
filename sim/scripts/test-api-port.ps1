[CmdletBinding()]
param(
    [string]$WslIp,
    [int]$Port = 4567,
    [string]$Distro
)

if (-not $WslIp) {
    $scriptDir = Split-Path -Parent $PSCommandPath
    $getIpScript = Join-Path $scriptDir "get-wsl-ip.ps1"
    if ($Distro) {
        $WslIp = & $getIpScript -Distro $Distro
    }
    else {
        $WslIp = & $getIpScript
    }
}

Write-Host "Checking Windows -> WSL API bridge target $WslIp`:$Port ..."
$result = Test-NetConnection -ComputerName $WslIp -Port $Port -InformationLevel Detailed

if ($result.TcpTestSucceeded) {
    Write-Host "OK: port $Port is reachable at $WslIp." -ForegroundColor Green
}
else {
    Write-Host "NOT REACHABLE: start the WSL API first with 'make api', then re-run this check." -ForegroundColor Yellow
}

$result
