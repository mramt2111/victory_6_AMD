<#
.SYNOPSIS
    One-command launcher for VIctory Unified with tuned RDNA3/4 and Native DXR 1.1 defaults.

.DESCRIPTION
    Starts an Xbox Series title under VIctory Unified with the tuned
    performance flags (including GPU-side 64 KiB+ buffer compares, Ports 5/6/7
    CBV/guard/journal paths, background heap worker, scoped UAV barriers,
    RDNA Wave32 lowering, and optional Native DXR 1.1 hardware ray tracing).

    Auto-detects game_scarlett_master.exe in the current directory or parent
    directory when -Exe is omitted, and builds victory.exe automatically if
    not yet compiled.

.EXAMPLE
    .\scripts\play-rdna.ps1
.EXAMPLE
    .\scripts\play-rdna.ps1 -Exe D:\Games\GTA6\game_scarlett_master.exe -Log .\run.log
.EXAMPLE
    .\scripts\play-rdna.ps1 -Safe
.EXAMPLE
    .\scripts\play-rdna.ps1 -Fps 60 -NativeRT -Output 1440p
#>
param(
    [string]$Exe = '',
    [ValidateSet('series-x', 'series-s')][string]$Console = 'series-x',
    [switch]$Safe,
    [switch]$RayTracing,
    [switch]$NativeRT,
    [int]$Fps = 0,
    [string]$Output = '',
    [double]$Scale = 0,
    [string]$Log = '',
    [ValidateSet('release', 'diag')][string]$Build = 'release',
    [switch]$Rebuild,
    [hashtable]$More = @{}
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

if ($Log) {
    $Log = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Log)
    [System.IO.File]::WriteAllText($Log, '', [System.Text.Encoding]::Unicode)
}

function Write-LaunchMessage {
    param([string]$Message, [ConsoleColor]$Color = 'Gray')
    Write-Host $Message -ForegroundColor $Color
    if ($Log) { Add-Content -LiteralPath $Log -Value $Message -Encoding Unicode }
}

function Invoke-LaunchProgram {
    param([string]$Program, [string[]]$ProgramArguments)
    $global:LASTEXITCODE = 1
    if ($Log) {
        # Windows PowerShell wraps native stderr as ErrorRecords when it is redirected.
        # Keep those messages in the log, even with a terminating preference in the caller.
        $writer = [System.IO.StreamWriter]::new($Log, $true, [System.Text.Encoding]::Unicode)
        $writer.AutoFlush = $true
        try {
            $ErrorActionPreference = 'Continue'
            $PSNativeCommandUseErrorActionPreference = $false
            & $Program @ProgramArguments 2>&1 | ForEach-Object {
                $line = [string]$_
                $writer.WriteLine($line)
                $line
            }
        } finally {
            $writer.Dispose()
        }
    } else {
        & $Program @ProgramArguments
    }
}

# 1. Locate game executable automatically if not passed explicitly.
if (-not $Exe) {
    $candidates = @(
        (Join-Path (Get-Location) 'game_scarlett_master.exe'),
        (Join-Path $repoRoot '..\game_scarlett_master.exe'),
        (Join-Path $repoRoot 'game_scarlett_master.exe')
    )
    $Exe = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $Exe) {
        Write-LaunchMessage "Could not find game_scarlett_master.exe automatically." Red
        Write-LaunchMessage "Pass it explicitly: .\scripts\play-rdna.ps1 -Exe <path\to\game_scarlett_master.exe>"
        exit 1
    }
}
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
    Write-LaunchMessage "Executable not found: $Exe" Red
    exit 1
}
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$gameDir = Split-Path -Parent $Exe

# 2. Ensure victory.exe is built (build automatically on first run or -Rebuild).
$victory = Join-Path $repoRoot "target\$Build\victory.exe"
if ($Rebuild -or -not (Test-Path -LiteralPath $victory -PathType Leaf)) {
    Write-LaunchMessage "Building VIctory Unified ($Build profile)..." Cyan
    $cargo = Get-Command cargo.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $cargo) {
        Write-LaunchMessage "Cargo was not found. Install Rust with rustup, then reopen the launcher." Red
        exit 1
    }
    Push-Location $repoRoot
    try {
        if ($Build -eq 'diag') {
            Invoke-LaunchProgram $cargo.Source @('build', '--locked', '--profile', 'diag', '-p', 'victory', '--features', 'diagnostics')
        } else {
            Invoke-LaunchProgram $cargo.Source @('build', '--locked', '--release', '-p', 'victory')
        }
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Pop-Location
    }
}
if (-not (Test-Path -LiteralPath $victory -PathType Leaf)) {
    Write-LaunchMessage "Build output was not found: $victory" Red
    exit 1
}

# 3. Clean stale VICTORY_* vars from the shell (keep user path overrides).
$presetPath = Join-Path $PSScriptRoot 'rdna-preset.json'
$preset = Get-Content -LiteralPath $presetPath -Raw | ConvertFrom-Json
foreach ($entry in @(Get-ChildItem Env:VICTORY_* -ErrorAction SilentlyContinue | Where-Object { $_.Name -notin $preset.keep })) {
    Remove-Item "Env:$($entry.Name)"
}

# 4. Apply unified RDNA performance settings shared with the Linux launcher.
$settings = [ordered]@{}
$variant = if ($Safe) { $preset.safe } else { $preset.fast }
foreach ($entry in (@($preset.base.PSObject.Properties) + @($variant.PSObject.Properties))) {
    $settings[$entry.Name] = $entry.Value
}

if ($RayTracing -or $NativeRT) {
    $settings['VICTORY_BVH_BUILDER'] = '1'
}

if ($NativeRT) {
    $settings['VICTORY_NATIVE_RT'] = '1'
    $settings['VICTORY_NATIVE_RT_SUPPRESS_BUILDER'] = '1'
}

if ($Fps -gt 0) {
    $interval = [math]::Round(1000.0 / $Fps, 3)
    $settings['VICTORY_FRAME_INTERVAL_MS'] = $interval.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

if ($Output) {
    $resolvedOutput = switch ($Output.ToLowerInvariant()) {
        '720p'  { '1280x720' }
        '936p'  { '1664x936' }
        '1080p' { '1920x1080' }
        '1440p' { '2560x1440' }
        '4k'    { '3840x2160' }
        '2160p' { '3840x2160' }
        default { $Output }
    }
    $settings['VICTORY_OUTPUT'] = $resolvedOutput
}

if ($Scale -gt 0) {
    $settings['VICTORY_RENDER_SCALE'] = $Scale.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

foreach ($k in $More.Keys) {
    $settings[$k] = [string]$More[$k]
}

foreach ($k in $settings.Keys) {
    Set-Item "Env:$k" $settings[$k]
}

$mode = if ($Safe) { 'SAFE' } else { 'RDNA FAST' }
$rtTag = if ($NativeRT) { ' | Native DXR 1.1' } elseif ($RayTracing) { ' | BVH RT' } else { '' }
Write-LaunchMessage "Launching VIctory Unified [$mode | $Console | $Build$rtTag]" Green
Write-LaunchMessage "  Binary : $victory"
Write-LaunchMessage "  Game   : $Exe"
if ($Fps -gt 0) { Write-LaunchMessage "  Target : $Fps FPS ($($settings['VICTORY_FRAME_INTERVAL_MS']) ms)" }
if ($Output) { Write-LaunchMessage "  Output : $($settings['VICTORY_OUTPUT'])" }
if ($Log) { Write-LaunchMessage "  Log    : $Log" }

# Run from the game directory so relative package/RPF paths resolve cleanly.
Push-Location -LiteralPath $gameDir
try {
    Invoke-LaunchProgram $victory @('run', '--console', $Console, '--execute', $Exe)
    $runExitCode = $LASTEXITCODE
    Write-LaunchMessage "VIctory Unified exited with code $runExitCode."
} finally {
    Pop-Location
}
exit $runExitCode
