param(
    [string]$Font = "$PSScriptRoot\..\..\src\desktop2steoro\xr_viewer\fonts\NotoSansSC-Regular.otf",
    [string]$Charset = "$PSScriptRoot\..\..\src\desktop2steoro\xr_viewer\fonts\overlay_charset.txt",
    [string]$Output = "$PSScriptRoot\..\..\src\desktop2steoro\xr_viewer\fonts\d2s_overlay_msdf",
    [int]$FontSize = 48,
    [int]$TextureSize = 2048,
    [int]$DistanceRange = 4
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$cli = Join-Path $repoRoot ".tools\msdf\node_modules\.bin\msdf-bmfont.cmd"

if (-not (Test-Path -LiteralPath $Font)) {
    throw "MSDF source font not found: $Font"
}
if (-not (Test-Path -LiteralPath $Charset)) {
    throw "MSDF charset file not found: $Charset"
}
if (-not (Test-Path -LiteralPath $cli)) {
    throw "Install the pinned generator first: npm install --no-audit --no-fund msdf-bmfont-xml@2.8.0 --prefix .tools/msdf"
}

$outputDir = Split-Path -Parent $Output
New-Item -ItemType Directory -Force $outputDir | Out-Null

& $cli $Font `
    --output-type json `
    --filename $Output `
    --font-size $FontSize `
    --charset-file $Charset `
    --texture-size "$TextureSize,$TextureSize" `
    --texture-padding 4 `
    --distance-range $DistanceRange `
    --field-type msdf `
    --smart-size

if ($LASTEXITCODE -ne 0) {
    throw "msdf-bmfont failed with exit code $LASTEXITCODE"
}

# msdf-bmfont names the JSON file after the font face. Normalize it so the
# runtime can load one stable asset name regardless of the source font.
$outputJson = "$Output.json"
$fontStem = [IO.Path]::GetFileNameWithoutExtension($Font)
$generatedJson = Join-Path $outputDir "$fontStem.json"
if ((Test-Path -LiteralPath $generatedJson) -and ($generatedJson -ne $outputJson)) {
    Move-Item -LiteralPath $generatedJson -Destination $outputJson -Force
}

Write-Host "Generated MSDF overlay font assets at $Output.*"
