# Fetch JPL DE440 ephemeris kernel (de440.bsp, ~32 MB) into data/.
# Requires internet access. Used by the JPL regression test.
$ErrorActionPreference = "Stop"
$dataDir = Join-Path $PSScriptRoot "..\data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
$url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"
$out = Join-Path $dataDir "de440.bsp"
if (Test-Path $out) {
    Write-Host "de440.bsp already present at $out"
} else {
    Write-Host "Downloading $url ..."
    Invoke-WebRequest -Uri $url -OutFile $out
    Write-Host "Downloaded to $out"
}
