param(
    [switch]$DisablePerformanceLog
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"
if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

if ($PSVersionTable.PSVersion.Major -lt 7) {
    $pwsh = Get-Command pwsh.exe -ErrorAction SilentlyContinue
    if ($null -ne $pwsh) {
        Write-Host "Current PowerShell is $($PSVersionTable.PSVersion). Relaunching with PowerShell 7..."
        & $pwsh.Source -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath @args
        exit $LASTEXITCODE
    }
    Write-Warning "PowerShell 7 (pwsh.exe) was not found; continuing with Windows PowerShell."
}

$names = @(
    "D2S_FPS_BREAKDOWN",
    "D2S_RUNTIME_FRAME_LOG_REFRESH_S",
    "D2S_SLOW_RUNTIME_LOG_MS"
)
$previous = @{}
foreach ($name in $names) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$runExitCode = 1
try {
    if ($DisablePerformanceLog) {
        Remove-Item Env:D2S_FPS_BREAKDOWN -ErrorAction SilentlyContinue
        Remove-Item Env:D2S_RUNTIME_FRAME_LOG_REFRESH_S -ErrorAction SilentlyContinue
        Remove-Item Env:D2S_SLOW_RUNTIME_LOG_MS -ErrorAction SilentlyContinue
        Write-Host "Performance breakdown logging: disabled"
    }
    else {
        $env:D2S_FPS_BREAKDOWN = "1"
        $env:D2S_RUNTIME_FRAME_LOG_REFRESH_S = "5"
        $env:D2S_SLOW_RUNTIME_LOG_MS = "200"
        Write-Host "Performance breakdown logging: enabled"
        Write-Host "Runtime timing refresh: every 5 seconds"
    }

    Write-Host "Starting Desktop2Stereo Local Viewer..."
    Write-Host "Working directory: $repoRoot"
    & $runBat
    $runExitCode = $LASTEXITCODE
}
finally {
    foreach ($name in $names) {
        $oldValue = $previous[$name]
        if ($null -eq $oldValue) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable($name, $oldValue, "Process")
        }
    }
}

exit $runExitCode
