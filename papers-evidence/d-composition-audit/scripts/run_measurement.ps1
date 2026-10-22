# Run the composed-bounds interaction measurement against a pinned boys build.
#
# Machine rule (coordinator): this is a timing-sensitive measurement. Do not run
# it until the coordinator signals. Run it on a quiet machine, alone: no build,
# lane, generator or other timing run concurrently. Read and report the canary.
# The canary is checked before the measurement (a wide gate stops the run) and
# again after it; both readings are written to environment.txt.
#
# The build, the per-lane argument emission and the reference generation all
# happen before the canary gate, never during the measurement. The generator
# reads the arguments the driver emitted, so it computes F_n at each lane's
# exact narrowed input rather than at the decimal. If MSBuild leaves persistent
# nodes running, the gate will read wide; wait for them to exit and rerun.
# Under the owner's 2026-10-20 ruling the canary is a hint: with
# BOYS_CANARY_HINT_ONLY=1 a wide reading is recorded and the run proceeds under
# AMENDMENT-2026-10-20, with its rows labelled timing-provisional.
#
# Usage, only after the coordinator's signal:
#   powershell -File scripts/run_measurement.ps1 -CoordinatorSignal
#   powershell -File scripts/run_measurement.ps1 -CoordinatorSignal -AllowOtherRevision
#
# The stability design repeats the run on both multiply-add routes. The second
# Windows configuration is selected with -MulAddRoute separate; it writes to
# results-msvc-separate and builds in build-separate, and it reuses the primary
# run's reference at the same narrowed inputs:
#   powershell -File scripts/run_measurement.ps1 -CoordinatorSignal `
#       -MulAddRoute separate -Reference results\reference.csv
#
# The script records the library revision and dirty state, the generator and
# build type, the compiler flags (compile_commands.json when the generator
# writes it) and the driver's own environment file beside the results.

[CmdletBinding()]
param(
    [string]$BoysSource = "C:\Users\myaml\dev\bladawiec\external\boys",
    [string]$ExpectedRevision = "482fd3388c276091834a6c12a642d9de2481c9f2",
    [string]$OutDir = "",
    [string]$BuildDir = "",
    [string]$Reference = "",
    [ValidateSet("fused", "separate")]
    [string]$MulAddRoute = "fused",
    [string]$Python = "python",
    [switch]$AllowOtherRevision,
    [switch]$CoordinatorSignal
)

$ErrorActionPreference = "Stop"

# The four stability configurations are separate builds and separate result
# directories. The primary one keeps the original paths so the paper's figures
# and the existing smoke workflow do not move; the other three get a suffix that
# names the configuration.
if (-not $OutDir) {
    $OutDir = if ($MulAddRoute -eq "separate") {
        Join-Path $PSScriptRoot "..\results-msvc-separate"
    } else {
        Join-Path $PSScriptRoot "..\results"
    }
}
if (-not $BuildDir) {
    $BuildDir = if ($MulAddRoute -eq "separate") {
        Join-Path $PSScriptRoot "..\build-separate"
    } else {
        Join-Path $PSScriptRoot "..\build"
    }
}
$separateValue = if ($MulAddRoute -eq "separate") { "ON" } else { "OFF" }

if (-not $CoordinatorSignal) {
    throw "wait for the coordinator's signal: the interaction probe is timing-sensitive and runs on a quiet machine, alone. Pass -CoordinatorSignal only when it has been given."
}

if (-not (Test-Path (Join-Path $BoysSource "CMakeLists.txt"))) {
    throw "BoysSource does not look like the boys source: $BoysSource"
}

$revision = (& git -C $BoysSource rev-parse HEAD).Trim()
$dirty = @(& git -C $BoysSource status --porcelain).Count

if (($revision -ne $ExpectedRevision) -and (-not $AllowOtherRevision)) {
    throw "boys is at $revision, expected $ExpectedRevision. Pass -AllowOtherRevision to measure a different revision."
}

Write-Host "boys revision: $revision (dirty paths: $dirty)"
if ($dirty -ne 0) {
    Write-Warning "the boys tree has uncommitted changes; the figures belong to the working tree, not to $revision"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$arguments = Join-Path $OutDir "arguments.csv"
$reference = Join-Path $OutDir "reference.csv"
$cells = Join-Path $OutDir "cells.csv"
$environment = Join-Path $OutDir "environment.txt"
$summary = Join-Path $OutDir "summary.json"

# The summary is the analysis engine; a logic failure there would waste the one
# probe run. Exercise it on a synthetic fixture with a planted interaction and a
# planted violation before any build or measurement starts. This is not a
# library measurement and touches no Boys output.
Write-Host "self-testing the summary analysis (synthetic, no library run)..."
& $Python (Join-Path $PSScriptRoot "selftest_summarise.py")
if ($LASTEXITCODE -ne 0) {
    throw "summary self-test failed with exit code $LASTEXITCODE; fix it before the probe"
}
& $Python (Join-Path $PSScriptRoot "selftest_stability.py")
if ($LASTEXITCODE -ne 0) {
    throw "stability self-test failed with exit code $LASTEXITCODE; fix it before the probe"
}

$ninja = Get-Command ninja -ErrorAction SilentlyContinue
$cl = Get-Command cl -ErrorAction SilentlyContinue

# Ninja is usable only when the MSVC environment is already live: ninja on PATH
# alone is not enough, because cl must also resolve. The Visual Studio
# generator finds its own toolchain, so it is the safe default on this host.
$useNinja = ($null -ne $ninja) -and ($null -ne $cl)

$cmakeArgs = @(
    "-S", $PSScriptRoot,
    "-B", $BuildDir,
    "-DBOYS_SOURCE_DIR=$BoysSource",
    "-DBOYS_MULADD_SEPARATE=$separateValue",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
)

# CMake refuses to change an existing build directory's generator, and a reused
# build directory is the normal case after a smoke run. Honour the cached one.
$cachePath = Join-Path $BuildDir "CMakeCache.txt"
$cachedGenerator = $null
if (Test-Path $cachePath) {
    $generatorLine = Get-Content $cachePath | Select-String -Pattern '^CMAKE_GENERATOR:INTERNAL='
    if ($null -ne $generatorLine) {
        $text = $generatorLine.Line
        $cachedGenerator = $text.Substring($text.IndexOf("=") + 1)
    }
}

if ($null -ne $cachedGenerator) {
    $generator = $cachedGenerator
} elseif ($useNinja) {
    $generator = "Ninja"
} else {
    $generatorHelp = (& cmake --help) -join "`n"
    if ($generatorHelp -match "Visual Studio 18 2026") {
        $generator = "Visual Studio 18 2026"
    } elseif ($generatorHelp -match "Visual Studio 17 2022") {
        $generator = "Visual Studio 17 2022"
    } else {
        throw "no usable Ninja toolchain (cl is not on PATH) and no supported Visual Studio generator; run from a developer prompt or install one"
    }
}

$isNinja = $generator -match '^Ninja'
$cmakeArgs += @("-G", $generator)
if ($isNinja) {
    $cmakeArgs += @("-DCMAKE_BUILD_TYPE=Release")
} else {
    $cmakeArgs += @("-A", "x64")
}

Write-Host "configuring ($generator)..."
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    throw "cmake configure failed with exit code $LASTEXITCODE"
}

# A single-config build with an empty build type is -O0. Refuse it.
if ($isNinja) {
    $cache = Get-Content (Join-Path $BuildDir "CMakeCache.txt")
    $buildTypeLine = $cache | Select-String -Pattern '^CMAKE_BUILD_TYPE:STRING='
    $buildType = ""
    if ($null -ne $buildTypeLine) {
        $line = $buildTypeLine.Line
        $buildType = $line.Substring($line.IndexOf("=") + 1)
    }
    if ($buildType -notmatch "Release|RelWithDebInfo") {
        throw "build type is '$buildType', not an optimized one; refusing an -O0 measurement"
    }
} else {
    $buildType = "Release (multi-config)"
}

Write-Host "building measure-interaction..."
& cmake --build $BuildDir --target measure-interaction --config Release
if ($LASTEXITCODE -ne 0) {
    throw "build failed with exit code $LASTEXITCODE"
}

if ($isNinja) {
    $exe = Join-Path $BuildDir "measure-interaction.exe"
} else {
    $exe = Join-Path $BuildDir "Release\measure-interaction.exe"
}
if (-not (Test-Path $exe)) {
    throw "driver not found at $exe"
}

# The driver narrows each nominal argument to its lane's own format. Emitting
# that narrowed input is what lets the reference be computed at the value the
# lane actually sees, so the arithmetic is compared and not the narrowing.
Write-Host "emitting per-lane arguments..."
& $exe --emit-arguments $arguments
if ($LASTEXITCODE -ne 0) {
    throw "argument emission failed with exit code $LASTEXITCODE"
}

if ($Reference) {
    # The stability configurations measure the same narrowed grid, so one
    # 60-digit reference serves them all. Copy it into this configuration's
    # directory so the manifest covers the exact file the run read.
    if (-not (Test-Path $Reference)) {
        throw "reference '$Reference' does not exist"
    }
    Write-Host "reusing reference '$Reference' (same narrowed inputs, checked by the driver)..."
    # PowerShell refuses to copy a file onto itself; the primary configuration's
    # default reference IS this configuration's directory.
    if ([IO.Path]::GetFullPath($Reference) -ne [IO.Path]::GetFullPath($reference)) {
        Copy-Item $Reference $reference -Force
    }
} else {
    Write-Host "generating reference at each lane's exact narrowed input..."
    & $Python (Join-Path $PSScriptRoot "gen_reference.py") --arguments $arguments --out $reference
    if ($LASTEXITCODE -ne 0) {
        throw "reference generator failed with exit code $LASTEXITCODE (2 means mpmath is missing or the arguments are bad; do not use python -I, it hides user-site mpmath)"
    }
}

# Machine rule: the canary gate runs after the build and the generator have
# exited and before any cell is measured. A wide gate stops the run; the
# driver's measurement mode repeats the gate before its first cell and reads it
# again after the last.
Write-Host "canary gate (quiet machine required)..."
& $exe --canary-only
if ($LASTEXITCODE -ne 0) {
    throw "canary gate is wide (exit $LASTEXITCODE); the machine is not quiet. Wait for other work to finish and rerun."
}

# Nothing else runs here until the measurement process exits: no generator, no
# build, no lane, no other timing run.
Write-Host "measuring (alone)..."
& $exe --arguments $arguments --reference $reference --out $cells --env $environment
if ($LASTEXITCODE -ne 0) {
    throw "measurement failed with exit code $LASTEXITCODE (a wide canary leaves the rows provisional; nothing ships from them)"
}

# The stability design rests on the route actually delivered, not on the
# configure-time spelling. A contracting build turns the separate selection into
# fused steps, and a run that called itself separate would then compare two
# copies of one arithmetic.
$routeLine = Get-Content $environment | Select-String -Pattern '^muladd_fp64=' | Select-Object -First 1
if ($null -eq $routeLine) {
    throw "environment.txt has no muladd_fp64 line; the route in force is unrecorded"
}
$routeInForce = $routeLine.Line.Substring($routeLine.Line.IndexOf("=") + 1)
if ($routeInForce -ne $MulAddRoute) {
    throw "this run asked for the $MulAddRoute route but the build reports $routeInForce in force; the stability premise is not met"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$papersRevision = (& git -C $repoRoot rev-parse HEAD).Trim()
$papersDirty = @(& git -C $repoRoot status --porcelain).Count
$cmakeVersion = (& cmake --version | Select-Object -First 1)

$facts = @(
    "",
    "papers_revision=$papersRevision",
    "papers_dirty_paths=$papersDirty",
    "boys_revision=$revision",
    "boys_dirty_paths=$dirty",
    "target=msvc",
    "muladd_route=$MulAddRoute",
    "stability_configuration=msvc-$MulAddRoute",
    "cmake_generator=$generator",
    "cmake_build_type=$buildType",
    "cmake_version=$cmakeVersion",
    "python=$Python",
    "source_date_utc=$((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))",
    "machine_rule=quiet machine, alone; no build, lane, generator or other timing run concurrently",
    "canary_policy=hint-only (owner ruling 2026-10-20; see AMENDMENT-2026-10-20.md)",
    "amendment=AMENDMENT-2026-10-20.md",
    "coordinator_signal=received",
    "canary_read_by=driver (canary: pre-measurement gate, post-measurement reading; see canary_* lines)",
    "manifest=manifest.sha256"
)
Add-Content -Path $environment -Value $facts -Encoding utf8

$compileCommands = Join-Path $BuildDir "compile_commands.json"
if (Test-Path $compileCommands) {
    Copy-Item $compileCommands (Join-Path $OutDir "compile_commands.json") -Force
}

Write-Host "summarising..."
& $Python (Join-Path $PSScriptRoot "summarise_interaction.py") --cells $cells --out $summary
if ($LASTEXITCODE -ne 0) {
    throw "summary failed with exit code $LASTEXITCODE"
}

# Provenance: hash the driver and every measured result file. environment.txt is
# not in the manifest: it records the manifest's own SHA-256, and a file cannot
# contain the digest of a manifest that hashes it. Verify environment ->
# manifest_sha256 -> manifest.sha256 -> the files it lists.
$manifest = Join-Path $OutDir "manifest.sha256"
$hashTargets = @($exe, $arguments, $reference, $cells, $summary)
if (Test-Path $compileCommands) {
    $hashTargets += (Join-Path $OutDir "compile_commands.json")
}
$hashLines = foreach ($target in $hashTargets) {
    $hash = (Get-FileHash -Algorithm SHA256 -Path $target).Hash.ToLowerInvariant()
    "$hash  $([System.IO.Path]::GetFileName($target))"
}
Set-Content -Path $manifest -Value $hashLines -Encoding utf8
$manifestHash = (Get-FileHash -Algorithm SHA256 -Path $manifest).Hash.ToLowerInvariant()
Add-Content -Path $environment -Value "manifest_sha256=$manifestHash" -Encoding utf8

Write-Host "results: $OutDir"
Write-Host "manifest: $manifest"
