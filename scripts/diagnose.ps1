<#
.SYNOPSIS
    One-click Windows PC hardware and VIctory environment diagnostic tool.

.DESCRIPTION
    Reports OS build, CPU cores, total/free RAM, GPU model & VRAM, driver version,
    victory.exe presence, victory_shader_cache size, and recommends a preset from
    presets/ (rdna4-4k, rdna2-1440p, or low-end-8gb).
#>
param(
    [string]$GameDir = ''
)

$ErrorActionPreference = 'SilentlyContinue'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '  VIctory Windows PC System Diagnostics' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

# 1. OS Build
$os = Get-CimInstance Win32_OperatingSystem
$osCaption = if ($os) { $os.Caption } else { 'Unknown Windows' }
$osVersion = if ($os) { $os.Version } else { 'Unknown' }
$osBuild   = if ($os) { $os.BuildNumber } else { 'Unknown' }
Write-Host ('[OS]       {0} (Version {1}, Build {2})' -f $osCaption, $osVersion, $osBuild)

# 2. CPU Cores
$cpus = @(Get-CimInstance Win32_Processor)
$cpuName = if ($cpus.Count -gt 0) { ($cpus[0].Name).Trim() } else { 'Unknown CPU' }
$physCores = ($cpus | Measure-Object -Property NumberOfCores -Sum).Sum
$logCores  = ($cpus | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
if (-not $logCores) { $logCores = [Environment]::ProcessorCount }
Write-Host ('[CPU]      {0} ({1} physical cores / {2} logical threads)' -f $cpuName, $physCores, $logCores)

# 3. RAM (Total / Free)
$totalRamGb = if ($os) { [math]::Round(($os.TotalVisibleMemorySize / 1MB), 2) } else { 0 }
$freeRamGb  = if ($os) { [math]::Round(($os.FreePhysicalMemory / 1MB), 2) } else { 0 }
Write-Host ('[RAM]      {0} GiB total / {1} GiB free' -f $totalRamGb, $freeRamGb)

# 4. GPU Model, True VRAM (>4 GiB via registry when available), & Driver Version
$regVramByName = @{}
$regBase = 'HKLM:\SYSTEM\ControlSet001\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
if (Test-Path $regBase) {
    Get-ChildItem $regBase -ErrorAction SilentlyContinue | ForEach-Object {
        $props = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
        if ($props -and $props.DriverDesc -and $props.'HardwareInformation.qwMemorySize') {
            $bytes = [uint64]$props.'HardwareInformation.qwMemorySize'
            if ($bytes -gt 0) {
                $regVramByName[$props.DriverDesc.Trim()] = $bytes
            }
        }
    }
}

$gpus = @(Get-CimInstance Win32_VideoController)
$bestGpuName = 'Unknown GPU'
$bestVramGb  = 0.0
if ($gpus.Count -eq 0) {
    Write-Host '[GPU]      No display adapter detected' -ForegroundColor Yellow
} else {
    foreach ($gpu in $gpus) {
        $name = if ($gpu.Name) { $gpu.Name.Trim() } else { 'Unknown Adapter' }
        $vramBytes = [uint64]0
        if ($regVramByName.ContainsKey($name)) {
            $vramBytes = $regVramByName[$name]
        } elseif ($gpu.AdapterRAM) {
            $vramBytes = [uint64]$gpu.AdapterRAM
        }
        $vramGb = [math]::Round(($vramBytes / 1GB), 2)
        $drvVer = if ($gpu.DriverVersion) { $gpu.DriverVersion } else { 'Unknown' }
        $drvDate = if ($gpu.DriverDate) { $gpu.DriverDate.ToString('yyyy-MM-dd') } else { 'Unknown' }
        Write-Host ('[GPU]      {0}' -f $name)
        Write-Host ('           VRAM: {0} GiB | Driver: {1} ({2})' -f $vramGb, $drvVer, $drvDate)
        if ($vramGb -gt $bestVramGb) {
            $bestVramGb = $vramGb
            $bestGpuName = $name
        }
    }
}

# 5. victory.exe Presence
$exeCandidates = @()
if ($env:VICTORY_EXE) { $exeCandidates += $env:VICTORY_EXE }
$exeCandidates += (
    (Join-Path $repoRoot 'target\release\victory.exe'),
    (Join-Path $repoRoot 'target\diag\victory.exe'),
    (Join-Path $repoRoot 'victory.exe')
)
$foundExe = $null
foreach ($cand in $exeCandidates) {
    if ($cand -and (Test-Path -LiteralPath $cand -PathType Leaf)) {
        $foundExe = (Resolve-Path -LiteralPath $cand).Path
        break
    }
}
if (-not $foundExe) {
    $cmd = Get-Command 'victory.exe' -ErrorAction SilentlyContinue
    if ($cmd) { $foundExe = $cmd.Source }
}
if ($foundExe) {
    $exeInfo = Get-Item -LiteralPath $foundExe
    $exeMb = [math]::Round(($exeInfo.Length / 1MB), 2)
    Write-Host ('[BINARY]   Found: {0} ({1} MiB)' -f $foundExe, $exeMb) -ForegroundColor Green
} else {
    Write-Host '[BINARY]   victory.exe NOT found. Build with: cargo build --release --locked -p victory' -ForegroundColor Yellow
}

# 6. victory_shader_cache Size
$cacheCandidates = @()
if ($env:VICTORY_CACHE_DIR) { $cacheCandidates += $env:VICTORY_CACHE_DIR }
if ($GameDir) { $cacheCandidates += (Join-Path $GameDir 'victory_shader_cache') }
if ($env:VICTORY_GAME_DIR) { $cacheCandidates += (Join-Path $env:VICTORY_GAME_DIR 'victory_shader_cache') }
$cacheCandidates += (Join-Path $repoRoot 'victory_shader_cache')
if ($env:LOCALAPPDATA) { $cacheCandidates += (Join-Path $env:LOCALAPPDATA 'VIctory\cache-pareto') }

$foundCache = $null
foreach ($dir in $cacheCandidates) {
    if ($dir -and (Test-Path -LiteralPath $dir -PathType Container)) {
        $foundCache = (Resolve-Path -LiteralPath $dir).Path
        break
    }
}
if ($foundCache) {
    $cacheFiles = @(Get-ChildItem -LiteralPath $foundCache -Recurse -File -ErrorAction SilentlyContinue)
    $cacheBytes = ($cacheFiles | Measure-Object -Property Length -Sum).Sum
    if (-not $cacheBytes) { $cacheBytes = 0 }
    $cacheMb = [math]::Round(($cacheBytes / 1MB), 2)
    $hasPsoBook = Test-Path -LiteralPath (Join-Path $foundCache 'known-pipelines.bin') -PathType Leaf
    $psoNote = if ($hasPsoBook) { 'known-pipelines.bin present' } else { 'known-pipelines.bin not recorded yet' }
    Write-Host ('[CACHE]    {0} ({1} files, {2} MiB; {3})' -f $foundCache, $cacheFiles.Count, $cacheMb, $psoNote)
} else {
    Write-Host '[CACHE]    victory_shader_cache not found yet (run Preload-Shaders.bat <game_dir>)' -ForegroundColor Yellow
}

# 7. Preset Recommendation (rdna4-4k / rdna2-1440p / low-end-8gb)
$isHighEndGpu = ($bestVramGb -ge 15.5) -or ($bestGpuName -match 'RX 90[0-9]{2}|RX 7900|RTX 4090|RTX 5080|RTX 5090')
$isMidGpu     = ($bestVramGb -ge 11.5) -or ($bestGpuName -match 'RX 6800|RX 6900|RX 7700|RX 7800|RTX 3080|RTX 4070|RTX 4080')

if ($isHighEndGpu -and $totalRamGb -ge 24 -and $logCores -ge 12) {
    $recommended = 'rdna4-4k'
    $playMode = 'Play-VIctory.cmd native-rt'
    $reason = '16+ GiB VRAM class GPU with ample RAM and CPU threads'
} elseif ($isMidGpu -and $totalRamGb -ge 15.5 -and $logCores -ge 8) {
    $recommended = 'rdna2-1440p'
    $playMode = 'Play-VIctory.cmd pareto-residency'
    $reason = '12+ GiB VRAM class GPU with 16+ GiB RAM'
} else {
    $recommended = 'low-end-8gb'
    $playMode = 'Play-VIctory.cmd low-end'
    $reason = 'Conservative RAM/VRAM/CPU budgets recommended (<=12 GiB VRAM, <=16 GiB RAM, or <=8 threads)'
}

$presetPath = Join-Path $repoRoot ('presets\' + $recommended + '.json')
Write-Host ''
Write-Host '------------------------------------------------------------' -ForegroundColor Cyan
Write-Host ('Recommended Preset : {0}' -f $recommended) -ForegroundColor Green
Write-Host ('Preset File        : {0}' -f $presetPath)
Write-Host ('Reason             : {0}' -f $reason)
Write-Host ('Suggested Launcher : {0}' -f $playMode)
Write-Host '------------------------------------------------------------' -ForegroundColor Cyan
exit 0
