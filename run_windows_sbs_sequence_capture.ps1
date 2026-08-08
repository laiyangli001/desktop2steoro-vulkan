param(
    [ValidateSet(3, 6, 9, 12)]
    [int]$FrameContexts = 9
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"
if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = Join-Path $repoRoot "artifacts\sbs_sequence_$stamp"
$names = @(
    "D2S_SBS_CAPTURE_DIR",
    "D2S_SBS_CAPTURE_DELAY_SECONDS",
    "D2S_SBS_CAPTURE_SAMPLE_COUNT",
    "D2S_SBS_CAPTURE_IMAGE_COUNT",
    "D2S_SBS_CAPTURE_EYE_WIDTH",
    "D2S_OPENXR_VULKAN_FRAME_CONTEXTS"
)
$previous = @{}
foreach ($name in $names) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    $env:D2S_SBS_CAPTURE_DIR = $outputDir
    $env:D2S_SBS_CAPTURE_DELAY_SECONDS = "15"
    $env:D2S_SBS_CAPTURE_SAMPLE_COUNT = "300"
    $env:D2S_SBS_CAPTURE_IMAGE_COUNT = "6"
    $env:D2S_SBS_CAPTURE_EYE_WIDTH = "640"
    $env:D2S_OPENXR_VULKAN_FRAME_CONTEXTS = [string]$FrameContexts

    Write-Host "SBS sequence capture enabled."
    Write-Host "Wait after first SBS frame: 15 seconds"
    Write-Host "Pacing metadata: 300 new SBS outputs"
    Write-Host "Sparse SBS screenshots: 6"
    Write-Host "OpenXR Vulkan frame contexts: $FrameContexts"
    Write-Host "Output: $outputDir"
    Write-Host "Start video playback before the 15-second delay expires."

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

Write-Host "SBS capture folder: $outputDir"
exit $runExitCode
