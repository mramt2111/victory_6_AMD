> [!NOTE]
> **Testing in Progress:** This unified build is currently in active Windows CI and community GPU testing (with a focus on lower-end / 8 GB VRAM PCs and overall stability). Please include your `Diagnose-System.bat` output when sharing benchmarks or bug reports.

## What's New in `victory_6_AMD` (Unified Low-End & Stability Build)

This build combines features from `cool_fox`, `VIctory-DXR`, `victory-rdna`, and `Belobug-AMD` into one maintained base, adding 63 audited bug fixes, adaptive 8 GB GPU scaling, four custom runtime algorithms, and turnkey Windows shader-preloading tools.

### Low-End PC & Memory Stability
- **Adaptive `VICTORY_LOW_END` mode**: Automatically scales staging pools, worker threads, and DXR slab sizes (`32 MiB`) for 8 GB VRAM / lower-RAM systems.
- **VRAM & memory bounds**: Adds VRAM-budget-aware `VICTORY_TEXTURE_MAX`, `VICTORY_WINDOW_BUDGET_MB` LRU window eviction, `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` heaps (with automatic `E_INVALIDARG` fallback), and fixes the `HostHeap::release()` COM leak.
- **Lower RAM & lock contention**: Releases `Pending` graphics state and shader vectors after PSO compilation, adds lock-free `AtomicBool` fast paths in `Shadow`, and batches `K32QueryWorkingSetEx` working-set queries (256 blocks per call).

### D3D12, AMD RDNA & GPU Pipeline
- **Async copy-queue uploads (`VICTORY_UP_QUEUE=1`)**: Dedicated `D3D12_COMMAND_LIST_TYPE_COPY` uploader moving buffer transfers off the graphics queue.
- **AMD ReBAR / SAM Upload Heaps (`VICTORY_GPU_UPLOAD_HEAP=1`)**: Detects `D3D12_FEATURE_D3D12_OPTIONS16::GPUUploadHeapSupported` at startup with an automatic atomic fallback latch to standard `D3D12_HEAP_TYPE_UPLOAD` if a driver rejects it.
- **Zero-copy PSO & root signature caching**: Process-wide BLAKE3-keyed root signature cache and zero-copy DXIL `stage_bytes` lookups.
- **Long-session frame pacing**: `BufferSweep` per-part dirty-bucket caching, barrier coalescing (`VICTORY_JOIN_BARRIERS`), scoped UAV barriers (`VICTORY_SCOPED_UAV_BARRIERS=1`), and parallel texture readbacks.
- **AMD features & HUD**: FidelityFX FSR 4.1.2 bridge, `VICTORY_RDNA_WAVE32`, AMD RDNA 4 `getDimensions` stability workaround, and an upgraded dark F3 HUD with zero-allocation redraws.

### Custom Runtime Algorithms
- **H-RLA**: Binary monoid run-length summary tree over `DynamicBits` (`O(1)` exact rejection, `O(log(N/64))` contiguous descriptor allocation) + `strides_shadow` PCIe write-elision.
- **IT-DSE**: Lock-free per-buffer EWMA dirty-stride estimator dynamically choosing optimal block granularity in `4 KiB–256 KiB`.
- **ED-SAP**: Exponentially decayed (`T½ = 120s`) PSO co-occurrence tracker (`VICTORY_PREDICT_PSO`) with Smith's-rule compile-time utility ranking for background pre-warming.
- **BM-Swizzle**: Branchless F2 carry-masked Xbox 64 KiB tile stepping with x86-64 BMI2 `_pdep_u32` fast path.

### DXR 1.1, Shader Cache & HLE
- **Hardened DXR 1.1**: Batched `ResourceBarrier` calls, unified hardware DXR 1.1 handshake (`VICTORY_NATIVE_RT`), fail-safe `XboxBvhBuilder.dll` mandatory import validation, and executable `BuildProfile` checks.
- **Versioned zero-read-back shader cache (`VSH3`)**: Keyed on `TRANSLATOR_VERSION = 2` (no `Cargo.lock` wipes), fast vs. slow `REQUESTED` PSO queues, and `VICTORY_COMPILER_GUARD` / `VICTORY_WARM_BAD` crash isolation.
- **Optimized HLE & CPU decoders**: Reusable 4 MiB DirectStorage scratch buffers, `RwLock` drive/audio heaps, zero-allocation steady-state GameInput polling & XMA audio pumping, branchless Morton BCPack decoding, and a 9-bit XMA VLC lookup table.

---

## Quick Start (Windows PC)

1. **Check your system & recommended preset**:
   ```cmd
   Diagnose-System.bat
