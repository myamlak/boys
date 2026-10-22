# First Paper A sweep, end to end.
#
# Builds nothing but the harness translation unit, and only against a clean
# checkout of boys v5.0.0.  Run from a Visual Studio developer shell (or any
# shell where `cl` resolves) on a quiet machine; the run takes roughly five to
# fifteen minutes, dominated by the 60-digit reference.
#
#   powershell -ExecutionPolicy Bypass -File run.ps1
#   powershell -ExecutionPolicy Bypass -File run.ps1 -BoysTree C:\path\to\boys

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
$Source = Join-Path $Here "sweep1.cpp"
$Object = Join-Path $BuildDir "sweep1.obj"
$Exe = Join-Path $BuildDir "sweep1.exe"

# Provenance first, so a run that dies later still leaves the revision and
# flags on disk.  The boysprovenance digest walk ignores build artifacts.
$FlagLine = (@($CompileFlags) + @($Source) + @($Object, $Exe)) -join " "
& $Python (Join-Path $Here "../common/boysprovenance.py") `
    --boys-tree $BoysTree --harness sweep1 --harness-dir $Here `
    --flags $FlagLine --out (Join-Path $ResultsDir "provenance.json")
if ($LASTEXITCODE -ne 0) { throw "provenance failed" }

# The grid reads the committed gate table of this exact tree.
& $Python (Join-Path $Here "grid.py") --boys-tree $BoysTree --out (Join-Path $DataDir "grid.csv")
if ($LASTEXITCODE -ne 0) { throw "grid failed" }

# The reference is the tree's own generator, cross-checked by every route the
# tree carries.  60 digits for a double-precision comparison; see reference.py.
& $Python (Join-Path $Here "../common/reference.py") `
    --boys-tree $BoysTree --grid (Join-Path $DataDir "grid.csv") `
    --out (Join-Path $DataDir "reference.csv") `
    --validation (Join-Path $DataDir "reference_validation.txt") `
    --dps 60 --digits 60
if ($LASTEXITCODE -ne 0) { throw "reference failed" }

& cl @CompileFlags "/Fo$Object" "/Fe$Exe" $Source
if ($LASTEXITCODE -ne 0) { throw "compile failed" }

& $Exe --grid (Join-Path $DataDir "grid.csv") `
    --cells (Join-Path $ResultsDir "cells.csv") `
    --admission (Join-Path $ResultsDir "admission.csv") `
    --gain (Join-Path $ResultsDir "replica_gain.csv") `
    --batch full
if ($LASTEXITCODE -ne 0) { throw "sweep1 failed" }

& $Python (Join-Path $Here "analyze.py") `
    --tree $Here --boys-tree $BoysTree `
    --provenance (Join-Path $ResultsDir "provenance.json") `
    --validation (Join-Path $DataDir "reference_validation.txt")
if ($LASTEXITCODE -ne 0) { throw "analysis failed" }

$env:BOYS_SWEEP_TREE = $Here
& $Python (Join-Path $Here "../../make_tables.py")
if ($LASTEXITCODE -ne 0) { throw "make_tables failed" }

Write-Host ""
Write-Host "first sweep complete"
Write-Host "  cells:    $(Join-Path $ResultsDir 'cells.csv')"
Write-Host "  evidence: $(Join-Path $ResultsDir 'summary.txt')"
Write-Host "  tables:   ../../tables/"
