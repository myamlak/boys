# Second Paper A sweep, end to end.
#
# Builds only the harness translation unit, against a clean checkout of boys
# v5.0.0.  Run from a Visual Studio developer shell.  This sweep measures no
# duration, so it has no canary; it runs only after the coordinator has freed
# the machine, and its figures are deterministic arithmetic from a fixed
# argument list.
#
#   powershell -ExecutionPolicy Bypass -File run.ps1

#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$BoysTree = $(if ($env:BOYS_TREE) { $env:BOYS_TREE } else { "C:\Users\myaml\dev\bladawiec\external\boys" }),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

$tag = (& git -C $BoysTree describe --tags --exact-match HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or -not ($tag -match '^v5\.0\.0$')) {
    throw "boys tree is not at the exact tag v5.0.0: $BoysTree (tag: $tag)"
}
$status = & git -C $BoysTree status --porcelain
if ($LASTEXITCODE -ne 0 -or $status) {
    throw "boys tree at v5.0.0 is not clean; a dirty tree cannot be cited: $BoysTree"
}

$BuildDir = Join-Path $Here "build"
$ResultsDir = Join-Path $Here "results"
$DataDir = Join-Path $Here "data"
New-Item -ItemType Directory -Force -Path $BuildDir, $ResultsDir, $DataDir | Out-Null

$Include = Join-Path $BoysTree "include"
$CompileFlags = @(
    "/nologo", "/std:c++latest", "/O2", "/EHsc", "/fp:precise", "/arch:AVX2",
    "/I", $Include
)
$Source = Join-Path $Here "sweep2.cpp"
$Object = Join-Path $BuildDir "sweep2.obj"
$Exe = Join-Path $BuildDir "sweep2.exe"

$FlagLine = (@($CompileFlags) + @($Source) + @($Object, $Exe)) -join " "
& $Python (Join-Path $Here "../common/boysprovenance.py") `
    --boys-tree $BoysTree --harness sweep2 --harness-dir $Here `
    --flags $FlagLine --out (Join-Path $ResultsDir "provenance.json")
if ($LASTEXITCODE -ne 0) { throw "provenance failed" }

& $Python (Join-Path $Here "grid.py") --boys-tree $BoysTree --out (Join-Path $DataDir "grid.csv")
if ($LASTEXITCODE -ne 0) { throw "grid failed" }

& $Python (Join-Path $Here "fits.py") --boys-tree $BoysTree `
    --out (Join-Path $DataDir "fits.csv") `
    --validation (Join-Path $DataDir "fits_validation.txt") --dps 70
if ($LASTEXITCODE -ne 0) { throw "fits failed" }

& $Python (Join-Path $Here "../common/reference.py") `
    --boys-tree $BoysTree --grid (Join-Path $DataDir "grid.csv") `
    --out (Join-Path $DataDir "reference.csv") `
    --validation (Join-Path $DataDir "reference_validation.txt") `
    --dps 70 --digits 60
if ($LASTEXITCODE -ne 0) { throw "reference failed" }

& cl @CompileFlags "/Fo$Object" "/Fe$Exe" $Source
if ($LASTEXITCODE -ne 0) { throw "compile failed" }

& $Exe --grid (Join-Path $DataDir "grid.csv") --fits (Join-Path $DataDir "fits.csv") `
    --paths (Join-Path $ResultsDir "paths.csv") `
    --cells (Join-Path $ResultsDir "cells.csv") `
    --admission (Join-Path $ResultsDir "admission.csv")
if ($LASTEXITCODE -ne 0) { throw "sweep2 failed" }

& $Python (Join-Path $Here "analyze.py") `
    --tree $Here --boys-tree $BoysTree `
    --provenance (Join-Path $ResultsDir "provenance.json") `
    --validation (Join-Path $DataDir "reference_validation.txt")
if ($LASTEXITCODE -ne 0) { throw "analysis failed" }

Write-Host ""
Write-Host "second sweep complete"
Write-Host "  paths:    $(Join-Path $ResultsDir 'paths.csv')"
Write-Host "  evidence: $(Join-Path $ResultsDir 'summary.txt')"
Write-Host "  transfers: $(Join-Path $ResultsDir 'transfer_cells.csv')"
