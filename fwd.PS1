# update-wsl-portproxy.ps1
# Requires: PowerShell running as Administrator
#  powershell.exe -ExecutionPolicy Bypass -File C:\Users\jonlo\fwd.PS1

$ListenPort = 8765
$ListenPort2 = 22
$WslDistro  = ""   # optional: set to a distro name like "Ubuntu-24.04" to target a specific distro

function Get-WslIPv4 {
    param([string]$Distro)

    # One-line output + cut gives a clean IPv4
    $linuxCmd = "ip -o -4 addr show dev eth0 | awk '{print `$4}' | cut -d/ -f1 | head -n1"

    $ip = if ([string]::IsNullOrWhiteSpace($Distro)) {
        (wsl.exe -e sh -lc $linuxCmd).Trim()
    } else {
        (wsl.exe -d $Distro -e sh -lc $linuxCmd).Trim()
    }

    # If it still somehow contains "inet", strip it
    $ip = ($ip -replace '^inet\s+', '').Trim()

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
    & netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0   listenport=$ListenPort2 | Out-Null

    # Re-add rules pointing to current WSL IP
    & netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=$ListenPort connectaddress=$WslIP connectport=$ListenPort | Out-Null
    & netsh interface portproxy add v4tov4 listenaddress=0.0.0.0   listenport=$ListenPort connectaddress=$WslIP connectport=$ListenPort | Out-Null
    & netsh interface portproxy add v4tov4 listenaddress=0.0.0.0   listenport=$ListenPort2 connectaddress=$WslIP connectport=$ListenPort2 | Out-Null

    # Show the resulting mapping
    Write-Host "`nCurrent portproxy rules:"
    & netsh interface portproxy show v4tov4

    Write-Host "`nDone."
}
catch {
    Write-Error $_
    exit 1
}