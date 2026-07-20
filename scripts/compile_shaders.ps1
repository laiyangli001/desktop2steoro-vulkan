param(
    [string]$Compiler = "glslc",
    [string]$ShaderRoot = "$PSScriptRoot\..\shaders"
)

$ErrorActionPreference = "Stop"
$compilerCommand = Get-Command $Compiler -ErrorAction SilentlyContinue
if (-not $compilerCommand) {
    throw "Shader compiler '$Compiler' was not found. Install the Vulkan SDK or pass -Compiler."
}

$sourceFiles = Get-ChildItem -LiteralPath $ShaderRoot -Filter *.comp -File
if (-not $sourceFiles) {
    throw "No GLSL compute shaders found in $ShaderRoot."
}

foreach ($source in $sourceFiles) {
    $output = Join-Path $source.DirectoryName ($source.BaseName + ".spv")
    & $compilerCommand.Source -std=450 $source.FullName -o $output
    if ($LASTEXITCODE -ne 0) {
        throw "Shader compilation failed: $($source.Name)"
    }
    Write-Host "Compiled $($source.Name) -> $([IO.Path]::GetFileName($output))"
}
