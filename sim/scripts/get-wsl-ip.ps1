[CmdletBinding()]
param(
    [string]$Distro
)

$wslArgs = @()
if ($Distro) {
    $wslArgs += @("-d", $Distro)
}
$wslArgs += @("--", "bash", "-lc", "hostname -I | tr ' ' '\n' | head -n 1")

$ip = (& wsl.exe @wslArgs).Trim()
if (-not $ip) {
    throw "Could not discover a WSL IP address. Start Ubuntu/WSL and try again."
}

$ip
