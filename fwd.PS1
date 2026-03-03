# update-wsl-portproxy.ps1
# Requires: PowerShell running as Administrator

$ListenPort = 8765
$WslDistro  = ""   # optional: set to a distro name like "Ubuntu-24.04" to target a specific distro

function Get-WslIPv4 {
    param([string]$Distro)

    # Query WSL for the IPv4 on eth0 (WSL2 default)
    $cmd = if ([string]::IsNullOrWhiteSpace($Distro)) {
        'wsl.exe -e sh -lc "ip -4 addr show dev eth0 | awk ''/inet /{print $2}'' | cut -d/ -f1 | head -n1"'
    } else {
        "wsl.exe -d `"$Distro`" -e sh -lc `"ip -4 addr show dev eth0 | awk '/inet /{print `$2}' | cut -d/ -f1 | head -n1`""
    }

    $ip = (Invoke-Expression $cmd).Trim()
    if (-not $ip -or $ip -notmatch '^\d{1,3}(\.\d{1,3}){3}$') {
        throw "Could not determine WSL IPv4 address. Got: '$ip'"
    }
    return $ip
}

try {
    $WslIP = Get-WslIPv4 -Distro $WslDistro
    Write-Host "WSL IP detected: $WslIP"

    # Delete any existing rules (ignore errors if they don't exist)
    & netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=$ListenPort | Out-Null
    & netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0   listenport=$ListenPort | Out-Null

    # Re-add rules pointing to current WSL IP
    & netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=$ListenPort connectaddress=$WslIP connectport=$ListenPort | Out-Null
    & netsh interface portproxy add v4tov4 listenaddress=0.0.0.0   listenport=$ListenPort connectaddress=$WslIP connectport=$ListenPort | Out-Null

    # Show the resulting mapping
    Write-Host "`nCurrent portproxy rules:"
    & netsh interface portproxy show v4tov4

    Write-Host "`nDone."
}
catch {
    Write-Error $_
    exit 1
}