# Apply the fusion patch series (Windows / PowerShell)
#
#   pwsh apply-fusion.ps1 -WorkDir D:\src\llm [-Base <commit>]
#
param(
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [string]$Base = "9a9394a895b96003ca842a6041cb28ac49a108f7"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path (Join-Path $WorkDir ".git"))) {
    Write-Host "[1/4] cloning PrismML-Eng/llama.cpp -> $WorkDir"
    git clone --filter=blob:limit=204800 https://github.com/PrismML-Eng/llama.cpp.git $WorkDir
}

Push-Location $WorkDir
try {
    Write-Host "[2/4] fetching + checking out base $Base"
    git fetch --no-tags origin prism
    git checkout -B fusion-local $Base

    $patches = Get-ChildItem -Path $here -Filter *.patch | Sort-Object Name
    Write-Host "[3/4] applying $($patches.Count) patches"
    git am --3way @($patches.FullName)
    if ($LASTEXITCODE -ne 0) { throw "git am failed - resolve manually (git am --abort to bail out)" }

    Write-Host "[4/4] done. Example build:"
    Write-Host @"
cmake -S . -B build-hip -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DAMDGPU_TARGETS=gfx1030 `
  -DGGML_NATIVE=OFF `
  -DCMAKE_C_COMPILER="<rocm>\bin\clang.exe" `
  -DCMAKE_CXX_COMPILER="<rocm>\bin\clang++.exe" `
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_CURL=OFF `
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF `
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON `
  -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT="`$PWD"
ninja -C build-hip -j 16 llama-server llama-kvmem-server llama-kvmem-cli
"@
}
finally {
    Pop-Location
}
