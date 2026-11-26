# VIctory Unified

A unified best-of-three fork of [VIctory](https://github.com/Marvisak/VIctory), a Direct3D 12 and runtime compatibility layer for running Xbox Series X|S binaries natively on Windows using files supplied by the user.

This build combines three community branches into a single tree:
- **`codec-d--cool_fox` + 63 safety fixes:** hardened runtime, Ports 5/6/7 (`VICTORY_PORT_CBV`, `VICTORY_PORT_GUARD`, `VICTORY_PORT_JOURNAL`), background heap worker (`VICTORY_HEAP_WORKER`), scoped UAV barriers (`VICTORY_SCOPED_UAV_BARRIERS`), and RDNA Wave32 lowering (`VICTORY_RDNA_WAVE32`).
- **`maren-halv--VIctory-DXR`:** Native hardware DirectX Raytracing (DXR 1.1 `RayQuery`, `VICTORY_NATIVE_RT=1`) — replacing slow software BVH traversal loops and emulated BVH build shaders with real host GPU acceleration structures and hardware RT cores.
- **`XaviFortes--VIctory-RDNA`:** Output resolution & render-scale controls (`VICTORY_OUTPUT`, `VICTORY_RENDER_SCALE`), ambient lighting tuning (`VICTORY_AMBIENT`), X3D V-Cache core affinity, switch documentation linter (`xtask`), and the cross-platform desktop GUI launcher (`launch.bat`, `launch.sh`, `scripts/launcher.py`, `scripts/play-rdna.ps1`).

> **Legal & Asset Policy**
>
> **This repository contains no copyrighted game files, executables, shaders, textures, audio, or assets.** You must supply your own legally obtained dump of any title you run. This project is an independent open-source compatibility layer under **GPL-3.0** and is not affiliated with or endorsed by any console manufacturer, game developer, or publisher. See [docs/asset-policy.md](docs/asset-policy.md).

---

## Key Features

### Native Hardware DXR 1.1 Ray Tracing (`VICTORY_NATIVE_RT=1`)
- **`XboxBvhBuilder.dll` HLE (`crates/victory-hle/src/modules/bvh_builder.rs`):** Intercepts console BVH builder APIs directly rather than emulating dozens of BVH-building compute shaders in software.
- **Host DXR Acceleration Structure Manager (`crates/victory-gpu/src/dxr.rs`):** Builds and updates native D3D12 multi-geometry static BLASes, dynamic BLASes, and `PREFER_FAST_TRACE` TLASes directly on host RT hardware with topology validation, safe deferred resource retention, and null-TLAS fallback handling.
- **RDNA 2 → DXIL SM 6.6 `RayQuery` Lifter (`crates/victory-shader/src/emit/rayquery_lift.rs`, `lower.rs`):** Detects `image_bvh*_intersect_ray` traversal loops (opaque, front/back-track, and alpha-tested any-hit loops), coalesces `AllocateRayQuery` handles across disjoint loops, and lifts them into native DXIL Shader Model 6.6 `RayQuery` (`TraceRayInline` / `Proceed` / `Committed*` / `Candidate*`) instructions. Closest-hit `AnyHit` loops default to the inline fallback path on RDNA 4 (opt-in via `VICTORY_NATIVE_RT_ANYHIT=1`) so opaque/shadow passes capture the full hardware speedup without divergent `Proceed()` stalls.
- **AMD RDNA 4 Stability (`amdxc64.dll`):** Fixes buffer `getDimensions` lowering crashes and validates lifted DXR shaders through `dxil.dll` and host D3D12 PSO creation.

---

## Performance & Benchmarks

Measured on **AMD Ryzen 7 7800X3D · AMD Radeon RX 9070 XT (16 GB, RDNA 4) · 32 GB RAM · Windows 11**.

### Native DXR 1.1 vs. Software BVH (Head-to-Head `VICTORY_GPU_TIMING=1` + `DRED`, Series S Profile)

| Metric | Software BVH (`VICTORY_BVH_BUILDER=1`) | Native DXR 1.1 (`VICTORY_NATIVE_RT=1`) | Delta |
|---|---:|---:|---:|
| Total GPU card time / frame | `128.6 ms` | **`80.6 ms`** | **-37.3% (-48.0 ms)** |
| Indirect compute time / frame | `105.4 ms` | **`64.2 ms`** | **-39.1% (-41.2 ms)** |
| Shadow RT pass (`f607780aa8e502d0`) | `8.20 ms` | **`~0.1 ms`** | **~80× faster** |
| RT pass (`caf96a3229145fa6`) | `21.04 ms` | **`10.39 ms`** | **-50.6%** |
| RT pass (`36f04a4ed0ad1d9b`) | `5.01 ms` | **`2.19 ms`** | **-56.3%** |
| RT pass (`3592426e3e9142d8`) | `6.01 ms` | **`3.30 ms`** | **-45.1%** |
| Resident VRAM / faults | ~12–16 GB | **`~10.2 GB` (`0` faults)** | **~2+ GB lower** |

### Configuration FPS Summary (Measured & Estimated)

| Configuration | Build | Median / Typical FPS | Peak FPS | Notes |
|---|---|---:|---:|---|
| **Series S — RT Off** | `--release` (measured, warm) | **~18.8 fps** | **29.3 fps** | CPU `preparing`-bound (`~45.5 ms` prep, `0.5 ms` GPU wait) |
| **Series X (1440p) — RT Off** | `--release` (measured) | **~11.8 fps** | **24.6 fps** | CPU `preparing`-bound (`~58.0 ms` prep, `0.3 ms` GPU wait) |
| **Series S — Software RT** | `--release` (measured, `d1686f1`, 165s) | **~11.7 fps** settled (`12.0 fps` overall med) | **18.8 fps** | `1.4 ms` med / `4.7 ms` settled GPU wait, `54.4 ms` prep, `11.3 GB` max VRAM (`128.6 ms` timed `diag`) |
| **Series X (1440p) — Software RT** | `--release` (measured) | **~8.2 fps** | **19.9 fps** | Software BVH emulation (`175.8 ms` GPU in timed `diag`) |
| **Series S — Native DXR 1.1 RT** | `--release` (measured, `c9de853`) | **~12.0–15.1 fps** settled (`12.1 fps` overall med) | **25.1 fps** | CPU `preparing` cut to **`32.5 ms`** (from `58–69 ms`), `0` faults (`12.1 fps` med / `25.1 fps` peak across 21 world samples; `0.0 ms` GPU wait when within local VRAM budget) |
| **Series X (1440p) — Native DXR 1.1 RT** | `--release` (estimated) | **~11–20+ fps** (est.) | **~20+ fps** (est.) | Pending dedicated Series X `--release` Native RT benchmark pass |

*Note: `diag` runs with `VICTORY_GPU_TIMING=1` and `VICTORY_D3D12_DRED=1` incur heavy instrumentation overhead (~3–6 fps) and are used strictly for apples-to-apples per-shader GPU timing. Use `--release` without timing/DRED flags for gameplay.*

### Renderer, Residency & Frame-Time Improvements (Inherited from Upstream Forks)
- **Async list recording & parallel submission prep:** `VICTORY_THIN`, `VICTORY_ASYNC_SUBMIT`, `VICTORY_JOIN_PARTS`, AVX2 page comparisons, and parallel descriptor/refresh passes.
- **Background GPU read-back & on-card steps:** `VICTORY_RELAY`, `VICTORY_LAND`, `VICTORY_CARD_STEPS`, and texture read-back.
- **32 GB RAM residency & two-plane composition:** Fallback heaps, commit-pressure backoff, `TablePlan` caching, Wave32 lowering support, and Series S two-plane composition.
- **Frame record & replay diagnostics:** Per-pass GPU timing (`VICTORY_GPU_TIMING=1`) and replay tooling ([docs/replay.md](docs/replay.md)).

---

## Requirements

- **OS:** Windows 10/11 x64 (or Linux + Wine via `launch.sh` / `scripts/launcher.py`)
- **CPU:** x86-64 with AVX2
- **GPU:** Direct3D 12 GPU with **Shader Model 6.6**, **Mesh Shaders**, and **DXR Tier 1.1** (for `VICTORY_NATIVE_RT=1`)
- **Build tools:** Rust toolchain (pinned in `rust-toolchain.toml`) and MSVC C++ Build Tools with Windows SDK
- **Title files:** Your own dump stored **outside** this repository

---

## Building & Running

```powershell
# Release build (recommended for gameplay)
cargo build --release --locked -p victory

# Diagnostics build (with frame/GPU timing instrumentation)
cargo build --profile diag --locked -p victory --features diagnostics
```

Configure local paths outside the repository (see `victory.example.toml`):

```powershell
copy victory.example.toml victory.toml
```

### Desktop GUI Launcher (Windows & Linux/Wine)

Double-click `launch.bat` on Windows (or run `./launch.sh` on Linux) to open the desktop library launcher with controls for Console Profile, Output Resolution (`VICTORY_OUTPUT`), Target FPS (`VICTORY_FRAME_INTERVAL_MS`), Safe Mode, BVH Ray Tracing, Native DXR 1.1 RT (`VICTORY_NATIVE_RT=1`), and custom `VICTORY_*` overrides.

### PowerShell Launcher

```powershell
# Tuned RDNA + unified defaults (auto-builds victory.exe if needed)
scripts\play-rdna.ps1 -Exe <path-to-your-title-executable>

# With native DXR 1.1 hardware ray tracing, 1440p output, and 60 FPS cap
scripts\play-rdna.ps1 -Exe <path-to-your-title-executable> -NativeRT -Output 1440p -Fps 60

# Safe fallback mode (disables experimental optimizations)
scripts\play-rdna.ps1 -Exe <path-to-your-title-executable> -Safe
```

Fast-pipeline launcher presets default to release builds (without `VICTORY_D3D12_DRED=light` overhead) with barrier coalescing (`VICTORY_JOIN_BARRIERS=1`), in-place sync (`VICTORY_SYNC_IN_PLACE`), texture-watch pacing (`VICTORY_TEXTURE_LOOK_MS`), and software BVH builder suppression under Native DXR (`VICTORY_NATIVE_RT_SUPPRESS_BUILDER=1`). Optional ahead-of-time shader/pipeline pre-warming with `victory warm`:

```powershell
target\release\victory.exe shaders <path-to-shader-archives>
target\release\victory.exe warm --used
```

---

## Key Environment Switches

| Variable | Effect |
|---|---|
| `VICTORY_NATIVE_RT=1` | Enable native D3D12 DXR 1.1 `RayQuery` lifting & host AS management |
| `VICTORY_NATIVE_RT_SUPPRESS_BUILDER=1` | Suppress software BVH builder shaders when `VICTORY_NATIVE_RT=1` is active (auto-set in Native RT presets) |
| `VICTORY_JOIN_BARRIERS=1` | Coalesce redundant D3D12 resource barriers across adjacent commands (enabled in fast presets) |
| `VICTORY_SYNC_IN_PLACE` | Sync descriptor heap pages in place (on by default; set `0` to copy pages first) |
| `VICTORY_TEXTURE_LOOK_MS` | Poll interval in ms for texture slots waiting on backing memory (default `4000`) |
| `VICTORY_NATIVE_RT_ANYHIT=1` | Force closest-hit `AnyHit` loops onto `RayQuery` `Proceed()` (default off on RDNA 4) |
| `VICTORY_BVH_BUILDER=1` | Enable BVH builder support (paired with `VICTORY_NATIVE_RT=1` or software fallback) |
| `VICTORY_OUTPUT=2560x1440` | Override output resolution (`1280x720`, `1664x936`, `1920x1080`, `2560x1440`, `3840x2160`) |
| `VICTORY_RENDER_SCALE=0.75` | Override internal render scale (`0.5` to `1.0`) |
| `VICTORY_FRAME_INTERVAL_MS=16.667` | Cap frame pacing interval in milliseconds (e.g. `16.667` for 60 FPS) |
| `VICTORY_AMBIENT=0.8` | Tune ambient lighting scale |
| `VICTORY_RDNA_WAVE32=1` | Enable RDNA Wave32 shader lowering (`1` or `2`) |
| `VICTORY_D3D12_DRED=1` / `light` | Enable D3D12 DRED breadcrumbs for GPU fault diagnostics |
| `VICTORY_GPU_TIMING=1` | Log per-command GPU execution times (pair with `diag` build + DRED) |
| `VICTORY_FSR_4K` | Enable PC FidelityFX 4.1.2 upscaling to 3840×2160 (`quality`, `balanced`, or `performance`; also `-Fsr4K` in `scripts\play.ps1`) |
| `VICTORY_FIDELITYFX_DIR` | Folder containing signed `amd_fidelityfx_upscaler_dx12.dll` (or `[paths].fidelityfx_dir` in `victory.toml`) |
| `VICTORY_COMPILER_GUARD` | Catch driver PSO compiler access violations on worker threads and replace the worker (on by default; `0` to disable) |
| `VICTORY_WARM_BAD` | Skip pipeline keys recorded in `warm-bad.txt` that previously crashed the driver compiler (on by default; `0` to disable) |
| `VICTORY_TEXTURE_MAX` | Dynamic mip cap in pixels for streamed placed 2D textures (`0` uncapped; unset auto-caps at `1024`, `512` at ≥92% VRAM, `256` at ≥97%) |
| `VICTORY_PANEL=1` | Show the dark F3 telemetry HUD at startup (live 172-frame graph, GPU/VRAM/RAM breakdown, per-core CPU bars; toggle anytime with `F3`) |
| `VICTORY_CACHE_DIR` | Override local shader/pipeline cache directory (default under `%LOCALAPPDATA%`) |

For additional switches and architecture notes, see [docs/architecture.md](docs/architecture.md), [docs/fork-ray-tracing.md](docs/fork-ray-tracing.md), [docs/findings.md](docs/findings.md), and [docs/README-fxceless.md](docs/README-fxceless.md).

---

## Credits & License

Licensed under the **GNU General Public License v3.0** ([LICENSE](LICENSE)).

Built upon the work of the VIctory community:
- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)** — core loader, runtime HLE, D3D12 backend, and shader recompiler.
- **codec-d**, **maren-halv**, **XaviFortes**, **fxceless**, **avatarmanz**, **DexNom**, **Belobug**, **romainhedouin**, **ZJOU**, **myokex** (with Spexivus, joroyt, tightfall), **Joshua Odom**, **Geckomacabre**, and **cxy0072019-sketch** — upstream renderer, DXR 1.1, RDNA/launcher, residency, read-back, Wave32, and AMD/NVIDIA improvements.
