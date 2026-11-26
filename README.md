# VIctory Unified (v3)

A unified, hardened Windows x64 compatibility layer for running user-supplied Xbox Series X|S titles natively on PC, combining the strongest components of four community forks with 63+ verified memory-safety, concurrency, shader, and pipeline fixes.

> **Legal & Asset Policy:** This repository contains **no game files, proprietary code, or assets**. You must supply your own legal Xbox Series dump. This project is independent and is not affiliated with or endorsed by any game developer or publisher.

---

## Highlights

- **Hardened Core (`codec-d--cool_fox` + 63+ verified fixes):** Eliminates critical use-after-free, data-race, VEH, FFI-panic, and OOB bugs across `victory-gpu`, `victory-hle`, `victory-loader`, `victory-shader`, `victory-pipeline`, and `apps/victory`, plus BC7 rotation-mode and `xgamesave` error-4395 fixes.
- **Native DXR 1.1 Hardware Ray Tracing (`maren-halv--VIctory-DXR`):** Full RDNA2-to-DXIL SM 6.6 `RayQuery` lifter and host BLAS/TLAS acceleration-structure manager (`VICTORY_NATIVE_RT=1`), with `SKIP_RT_RESOLVE` / `SKIP_ACTIVATE_RAYS` / `ambient.rs` / auto-`AnyHit` fixes and `VICTORY_NATIVE_RT_SUPPRESS_BUILDER=1` to avoid dual-running software BVH compute shaders.
- **PC AMD FidelityFX FSR 4.1.2 Upscaling (`VIctory-Belobug-AMD`):** Routes console FidelityFX upscaling directly to AMD's signed PC `amd_fidelityfx_upscaler_dx12.dll` (`VICTORY_FIDELITYFX_DIR`, `VICTORY_FSR_4K=quality|balanced|performance`) for 4K output.
- **Fast Split-Memory Pipeline:**
  - Renderer Ports 5–7 (`VICTORY_PORT_CBV=1`, `VICTORY_PORT_GUARD=1`, `VICTORY_PORT_JOURNAL=1`) and hybrid in-place heaps (`VICTORY_INPLACE=2`).
  - Single-pass AVX2 descriptor compare (`VICTORY_SYNC_IN_PLACE`), coalesced barriers (`VICTORY_JOIN_BARRIERS=1`), scoped UAV barriers (`VICTORY_SCOPED_UAV_BARRIERS=1`), background bindless texture worker (`VICTORY_HEAP_WORKER=1`), and parallel texture readbacks (`texback::land_many`).
  - Lowered default PSO wait cap (32 ms) with `THREAD_PRIORITY_BELOW_NORMAL`, crash-safe `pipeline/known/writer.rs`, and driver shader-compiler crash guard (`VICTORY_COMPILER_GUARD` / `VICTORY_WARM_BAD`).
  - VRAM-aware dynamic texture mip cap (`VICTORY_TEXTURE_MAX`) and wall-clock texture re-check cap (`VICTORY_TEXTURE_LOOK_MS`).
- **Display, FPS, HUD & Launchers (`XaviFortes--VIctory-RDNA` + `VIctory-Belobug-AMD`):**
  - Frame-rate unlocker (`VICTORY_FRAME_INTERVAL_MS`, `VICTORY_TITLE_FPS`), resolution/scale patcher (`VICTORY_OUTPUT`, `VICTORY_RENDER_SCALE`, `VICTORY_UPSCALER`), wall-clock ambient settle (`VICTORY_AMBIENT_SETTLE`), X3D V-Cache CCD pinning (`VICTORY_CACHE_CORES`, `VICTORY_LANE_CORE`), and upgraded live F3 telemetry HUD (`VICTORY_PANEL=1`).
  - Cross-platform Tkinter GUI launcher (`scripts/launcher.py`, `launch.bat`, `launch.sh`) plus optimized `Play-VIctory.cmd`, `scripts/play.ps1`, and `scripts/play-rdna.ps1`.

---

## System Requirements

| Component | Minimum / Recommended |
|---|---|
| **OS** | Windows 10/11 x64 (or Linux via Wine + VKD3D-Proton) |
| **CPU** | x86_64 with **AVX2** (AMD X3D V-Cache or high-IPC 8+ core recommended) |
| **GPU** | Direct3D 12 with **Shader Model 6.6** and **Mesh Shaders** (DXR 1.1 for `VICTORY_NATIVE_RT=1`; **Resizable BAR enabled in BIOS** recommended) |
| **VRAM** | 16 GB+ recommended (auto-capped via `VICTORY_TEXTURE_MAX` on 16 GB cards) |
| **System RAM** | **64 GB recommended** (32 GB minimum; shadow copies + command-list allocators use 30–46 GB private commit) |
| **Build Tools** | Rust toolchain (pinned in `rust-toolchain.toml`) + MSVC C++ Build Tools & Windows SDK |

---

## Quick Start (Build & Pre-Warm)

```powershell
# 1. Copy and edit configuration (keep game & cache paths outside the repo)
copy victory.example.toml victory.toml

# 2. Run workspace tests and build release binary
cargo test --workspace --locked
cargo build --release --locked -p victory

# 3. Pre-translate shaders and warm PSO cache ahead of time (strongly recommended)
target\release\victory.exe shaders <path-to-shader-archives>
target\release\victory.exe warm --used
```

---

## Launching

### Desktop GUI Launcher (Windows & Linux/Wine)
```powershell
.\launch.bat
```
```bash
./launch.sh
```
Supports console profile (`series-x` / `series-s`), FPS target, output resolution, Safe Mode, Native DXR 1.1 RT, and custom `VICTORY_*` overrides.

### PowerShell Launcher (`scripts/play.ps1`)
Defaults to the optimized `release` build with Ports 5–7, barrier coalescing, scoped UAV barriers, and background texture uploads enabled:
```powershell
# Standard optimized launch
scripts\play.ps1 -Exe D:\Games\MyDump\game_scarlett_master.exe

# Native DXR 1.1 Hardware Ray Tracing (suppresses software BVH dual-run)
scripts\play.ps1 -Exe D:\Games\MyDump\game_scarlett_master.exe -NativeRT

# PC AMD FidelityFX FSR 4.1.2 (4K output; requires signed amd_fidelityfx_upscaler_dx12.dll)
$env:VICTORY_FIDELITYFX_DIR = 'D:\Tools\FidelityFX\signedbin'
scripts\play.ps1 -Exe D:\Games\MyDump\game_scarlett_master.exe -Fsr4K quality

# Safe fallback mode (disables experimental fast paths for troubleshooting)
scripts\play.ps1 -Exe D:\Games\MyDump\game_scarlett_master.exe -Safe
```

### Batch Launcher (`Play-VIctory.cmd`)
```powershell
$env:VICTORY_GAME_DIR = 'D:\Games\MyDump'
.\Play-VIctory.cmd pareto-residency
# or with Native DXR 1.1:
.\Play-VIctory.cmd native-rt
```

---

## Key Environment Switches

| Switch | Default | Purpose |
|---|---|---|
| `VICTORY_NATIVE_RT` | `0` | Enables native D3D12 DXR 1.1 `RayQuery` lifting & host BLAS/TLAS manager |
| `VICTORY_NATIVE_RT_SUPPRESS_BUILDER` | `0` (`1` in `-NativeRT`) | Suppresses console software BVH compute shaders when running Native DXR |
| `VICTORY_NATIVE_RT_ANYHIT` | Auto under pure DXR | Lifts closest-hit alpha-test/foliage `AnyHit` loops (`0` forces off) |
| `VICTORY_FSR_4K` | Off | PC FidelityFX 4K preset (`quality`, `balanced`, `performance`); requires `VICTORY_FIDELITYFX_DIR` |
| `VICTORY_OUTPUT` | Native | Overrides render output mode (`1280x720`, `1664x936`, `1920x1080`, `2560x1440`, `3840x2160`) |
| `VICTORY_FRAME_INTERVAL_MS` | `33.333` | Target frame interval in ms (`16.667` = 60 FPS cap, `0` = uncapped) |
| `VICTORY_TEXTURE_MAX` | Auto (1024/512/256) | Dynamic VRAM-aware max dimension for streamed 2D textures (`0` = uncapped) |
| `VICTORY_TEXTURE_LOOK_MS` | `4000` | Wall-clock cap for re-checking missing/partially-mapped texture slots |
| `VICTORY_SHOW_AFTER` / `VICTORY_UPLOAD_AHEAD` | `1` / `1` | Defers publishing new texture descriptors until copy-queue upload finishes |
| `VICTORY_JOIN_BARRIERS` / `VICTORY_SCOPED_UAV_BARRIERS` | `1` (presets) / `1` | Coalesces consecutive D3D12 barriers and scopes UAV barriers to target resources |
| `VICTORY_SYNC_IN_PLACE` | `1` | Single-pass AVX2 in-place descriptor heap comparison |
| `VICTORY_PORT_CBV` / `VICTORY_PORT_GUARD` / `VICTORY_PORT_JOURNAL` | `1` (presets) | Enables Renderer Ports 5–7 (raw in-place CBVs, shared guard locks, 64-shard journal) |
| `VICTORY_HEAP_WORKER` | `1` (presets) | Offloads bindless texture detiling/uploads to a background worker thread |
| `VICTORY_COMPILER_GUARD` / `VICTORY_WARM_BAD` | `1` / `1` | Catches GPU driver shader-compiler crashes and skips recorded bad PSOs |
| `VICTORY_PANEL` | `0` | Enables live F3 performance HUD (FPS, frame-time graph, VRAM/RAM, per-core CPU) |
| `VICTORY_AMBIENT_SETTLE` | `250,3` | Wall-clock ambient indirect lighting settle time when RT is off (`0` = legacy) |

---

## Credits & Upstream Lineage

Built on the shoulders of the VIctory community:

- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)** — core loader, HLE runtime, D3D12 backend, and GFX10→DXIL shader recompiler.
- **[codec-d/cool_fox](https://github.com/codec-d/cool_fox)** — NVIDIA/Wave32 fixes, residency controls, and diagnostics.
- **[maren-halv/VIctory-DXR](https://github.com/maren-halv/VIctory-DXR)** — native DXR 1.1 `RayQuery` lifter, host AS manager, RDNA4 fixes, `heap_worker`, and Ports 5–7.
- **[XaviFortes/VIctory-RDNA](https://github.com/XaviFortes/VIctory-RDNA)** — resolution/FPS patchers, AVX2 sync-in-place, barrier coalescing, parallel texback, X3D pinning, and desktop GUI launcher.
- **[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD)** & **[DexNom/VIctory](https://github.com/DexNom/VIctory)** — PC FidelityFX FSR 4.1.2 bridge, PSO compiler crash guard, `VICTORY_TEXTURE_MAX`, ambient indirect lighting, and F3 HUD.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**, **[myokex/victory-amd](https://github.com/myokex/victory-amd)**, **[FQY55kai/ZJOU-VICTORY](https://github.com/FQY55kai/ZJOU-VICTORY)**, **Geckomacabre**, and **cxy0072019-sketch**.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) and [docs/asset-policy.md](docs/asset-policy.md).
