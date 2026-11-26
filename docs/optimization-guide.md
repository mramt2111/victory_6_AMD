# Windows PC Optimization Guide

This guide covers the turnkey Windows workflow for diagnosing your PC, pre-translating and warming shaders, and choosing the right VIctory runtime switches.

## 1. Check Your Hardware First (`Diagnose-System.bat`)

Double-click `Diagnose-System.bat` (or run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\diagnose.ps1`) from the repository root. It reports:

- Windows OS version and build number
- CPU model, physical cores, and logical thread count
- Total and available system RAM
- GPU model, true VRAM capacity (including >4 GiB registry lookup), and driver version
- Whether `victory.exe` is built or on `PATH`
- Current `victory_shader_cache` location, size, and whether `known-pipelines.bin` has been recorded
- A recommended preset from `presets/` (`rdna4-4k`, `rdna2-1440p`, or `low-end-8gb`)

## 2. Preload Shaders & Warm PSOs (`Preload-Shaders.bat` and `victory.exe warm`)

First-run stutter comes from two separate steps: translating console shaders into DXIL, and compiling D3D12 Pipeline State Objects (PSOs) in your GPU driver.

1. **Pre-translate shaders before launching:** Drag and drop your game folder onto `Preload-Shaders.bat`, or run:
   ```cmd
   Preload-Shaders.bat "D:\Games\YourTitle"
   ```
   This scans the game directory, translates discovered shaders into `<game_dir>\victory_shader_cache`, and automatically sets `VICTORY_CACHE_DIR` to that folder.
2. **Record and warm D3D12 PSOs:** Launch the title once using `Play-VIctory.cmd` (which enables `VICTORY_PIPELINE_BOOK=1` when `victory_shader_cache` is present) so VIctory records encountered pipelines into `known-pipelines.bin`.
3. **Compile recorded PSOs ahead of time:** Re-run `Preload-Shaders.bat` or invoke `victory.exe warm` directly:
   ```cmd
   set VICTORY_CACHE_DIR=D:\Games\YourTitle\victory_shader_cache
   target\release\victory.exe warm
   ```
   `victory.exe warm` compiles every recorded pipeline in `known-pipelines.bin` into the on-disk D3D12 pipeline library so subsequent launches load ready-made PSOs instead of compiling mid-frame.

## 3. Launch with `Play-VIctory.cmd`

`Play-VIctory.cmd` applies the tuned pareto-residency switch set and automatically picks up `<game_dir>\victory_shader_cache` when present. Point its required executable path environment variable to your own title binary, then run one of its three modes:

| Command | Best For | Key Behavior |
| --- | --- | --- |
| `Play-VIctory.cmd` (or `pareto-residency`) | Most 12–16 GB VRAM Windows PCs | Balanced pareto-residency defaults with `VICTORY_UP_QUEUE=1` and on-demand texture mip streaming (`VICTORY_LEVELS_ON_DEMAND=1`). |
| `Play-VIctory.cmd native-rt` | High-end RDNA 3 / RDNA 4 GPUs (16+ GB VRAM) | Enables hardware DXR 1.1 ray tracing (`VICTORY_NATIVE_RT=1`, `VICTORY_BVH_BUILDER=1`, `VICTORY_NATIVE_RT_SUPPRESS_BUILDER=1`). |
| `Play-VIctory.cmd low-end` | 8 GB VRAM GPUs, <=16 GB RAM, or <=8-thread CPUs | Activates `VICTORY_LOW_END=1`, `VICTORY_OUTPUT=1280x720`, `VICTORY_TEXTURE_MAX=512`, `VICTORY_WINDOW_BUDGET_MB=512`, and disables RT and copy-queue uploads. |

For custom hardware profiles, see the JSON presets in `presets/`:

- `presets/rdna4-4k.json` — 4K RDNA 4 / RDNA 3 with Wave32, GPU upload heap, predictive PSOs, and native RT
- `presets/rdna2-1440p.json` — 1440p RDNA 2 / mid-range raster-only profile
- `presets/low-end-8gb.json` — 720p memory-constrained 8 GB VRAM profile
- `presets/linux-wine.json` — Conservative 1080p Wine/Proton (vkd3d-proton) compatibility profile

## 4. Key Performance Switches

Set any of these environment variables before running `Play-VIctory.cmd` to override its defaults:

- **`VICTORY_LOW_END`** (`1` or `0`): Forces or disables low-end RAM/VRAM/CPU scaling. By default, VIctory auto-detects low-end hardware when physical RAM is `<= 16 GiB`, DXGI local VRAM budget is `<= 12 GiB`, or logical CPU count is `<= 8`. Set `VICTORY_LOW_END=1` to force conservative memory budgets, or `VICTORY_LOW_END=0` to opt out on borderline systems.
- **`VICTORY_UP_QUEUE`** (`1` or `0`): Offloads CPU-modified buffer uploads onto a dedicated D3D12 copy queue instead of blocking the direct graphics queue. Recommended (`1`) on mid-to-high-end discrete GPUs; disable (`0`) on 8 GB / low-end systems or under Wine/Proton if you observe queue contention.
- **`VICTORY_GPU_UPLOAD_HEAP`** (`1` or `0`): Enables D3D12 GPU Upload Heap (Resizable BAR / SAM direct VRAM write) for upload buffers when supported by the adapter and Windows driver, reducing staging copies over PCIe. Pair with ReBAR enabled in UEFI on RDNA 3 / RDNA 4 GPUs.
- **`VICTORY_PREDICT_PSO`** (`1` or `0`): Speculatively pre-compiles likely pipeline state variants in the background ahead of first draw to cut remaining runtime shader-compile hitches on multi-core CPUs.

## 5. Quick Troubleshooting

- **Stuttering on first launch:** Run `Preload-Shaders.bat`, play past the main menu once to generate `known-pipelines.bin`, then run `Preload-Shaders.bat` (or `victory.exe warm`) again.
- **Out-of-VRAM crashes or heavy paging:** Switch to `Play-VIctory.cmd low-end` or set `VICTORY_LOW_END=1` with `VICTORY_TEXTURE_MAX=512` and `VICTORY_WINDOW_BUDGET_MB=512`.
- **Suspected GPU hang / device lost:** Enable lightweight DRED diagnostics with `VICTORY_D3D12_DRED=light` and set `VICTORY_LOG=info` before reproducing.
