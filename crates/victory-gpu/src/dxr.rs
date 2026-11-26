//! Native DXR Tier 1.1 acceleration structure manager.
//!
//! Opt-in via `VICTORY_NATIVE_RT=1`. Builds native host BLASes (dynamic from
//! guest geometry descriptors, static extracted on the GPU from rehydrated
//! console BVH4 buffers in `crate::memory`) and host TLASes (patching guest
//! instance descriptors' BLAS VAs through a GPU hash table).
//!
//! All `ID3D12GraphicsCommandList4` raytracing work is queued onto the guest
//! command list as owned [`DxrOp`] records and replayed on the submission
//! thread inside `Host::submit` right after `card_before` uploads dirty guest
//! buffers and arenas — never on `VICTORY_THIN` recording/worker threads.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::mem::{ManuallyDrop, offset_of, size_of};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, s};

use crate::host::{Host, HostList, Prelude, compile, host, mapped, name, not_resident, resident, transition};
use crate::object::This;

/// Size of each 256B-aligned AS backing slab (64 MiB).
const AS_SLAB_SIZE: u64 = 64 << 20;
/// Required DXR alignment for acceleration structures and scratch buffers.
const AS_ALIGN: u64 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT as u64;
/// Default initial reservation for a pre-bound TLAS slot so `HostHeap` SRVs
/// keep a stable GPU VA once the first `BuildTlas` runs.
const DEFAULT_TLAS_RESERVE: u64 = 8 << 20;
/// Minimum scratch buffer chunk allocated per pooled DXR submission.
const MIN_SCRATCH_SIZE: u64 = 16 << 20;
/// Upper bound on static BVH4 leaf nodes extracted per BLAS.
const MAX_STATIC_NODES: u64 = 262_144;
/// Maximum valid unpacked BVH4 size (matches `memory.rs::PACKED`).
const MAX_PACKED_BVH_BYTES: u64 = 64 << 20;
/// Maximum readback bytes copied per submission for async static BLAS upgrades (8 MiB).
const MAX_READBACK_BYTES_PER_SUB: u64 = 8 << 20;
/// Maximum geometries accepted per upgraded static BLAS.
const MAX_UPGRADE_GEOMS: usize = 2048;
/// Maximum triangle slots (including sparse NaN holes) per upgraded static BLAS (524,288 tris = 18 MiB).
const MAX_UPGRADE_TRIS: usize = 524_288;
/// Maximum upgraded static BLASes drained per submission.
const MAX_UPGRADES_PER_SUB: usize = 8;
/// Maximum vertex bytes uploaded in one multi-upgrade batch (16 MiB; a single valid upgrade up to 18 MiB still runs).
const MAX_UPGRADE_BATCH_BYTES: u64 = 16 << 20;
/// Maximum queued static upgrades awaiting GPU build (prevents unbounded RAM growth).
const MAX_PENDING_UPGRADES: usize = 256;
/// Maximum static BLAS compactions drained per submission.
const MAX_COMPACTIONS_PER_SUB: usize = 16;
/// Maximum queued static BLAS compactions awaiting GPU copy.
const MAX_PENDING_COMPACTIONS: usize = 8192;
/// Maximum 8-byte compacted-size postbuild queries recorded per submission (8192 × 8B = 64 KiB).
const COMPACT_INFO_SLOTS: usize = 8192;
const COMPACT_INFO_BYTES: u64 = (COMPACT_INFO_SLOTS * 8) as u64;
/// Pooled per-submission auxiliary buffers larger than this are dropped on fence completion.
const MAX_POOLED_AUX_BYTES: u64 = 4 << 20;
/// Pooled per-submission scratch buffers larger than this are dropped on fence completion.
const MAX_POOLED_SCRATCH_BYTES: u64 = 32 << 20;
/// Pooled per-submission static-extraction vertex buffers larger than this are dropped on fence completion.
const MAX_POOLED_EXTRACT_VERTS_BYTES: u64 = 4 << 20;
/// Pooled per-submission patched host-instance buffers larger than this are dropped on fence completion.
const MAX_POOLED_HOST_INSTANCES_BYTES: u64 = 2 << 20;
/// Maximum pooled DXR command lists kept in `DxrState::free_lists`.
const MAX_FREE_LISTS: usize = 16;
/// Maximum pooled DXR submission containers kept in `DxrState::free_submissions`.
const MAX_FREE_SUBS: usize = 4;
/// Maximum cached dynamic BLAS `GetRaytracingAccelerationStructurePrebuildInfo` entries.
const MAX_PREBUILD_CACHE: usize = 4096;
/// Maximum queued async BVH4 readback decode batches (bounded RAM, non-blocking relay).
const MAX_DECODE_QUEUE: usize = 16;
/// Maximum individual dirty hash-table slots tracked before falling back to `[min..=max]`.
const MAX_DIRTY_HASH_SLOTS: usize = 64;
/// Coalesce dirty hash-table runs separated by at most this many 16B slots (4 KiB).
const HASH_DIRTY_MERGE_GAP: usize = 256;
/// Open-addressed GPU hash table capacity (65536 × 16B = 1 MiB).
const HASH_CAPACITY: usize = 65_536;
const HASH_MASK: u32 = (HASH_CAPACITY - 1) as u32;
const HASH_TABLE_BYTES: u64 = (HASH_CAPACITY * 16) as u64;

/// Whether `VICTORY_NATIVE_RT=1` was requested in the environment.
pub fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NATIVE_RT").is_some_and(|v| v == "1"))
}

/// Whether native DXR is active (`VICTORY_NATIVE_RT=1` and host supports
/// DXR Tier 1.1 + `ID3D12Device5`).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| requested() && host().is_some_and(Host::raytracing_tier_1_1))
}

/// Whether dynamic `BuildRaytracingAccelerationStructureBatched2X` and
/// `CopyRaytracingAccelerationStructure` calls should skip forwarding to
/// `XboxBvhBuilder.dll` and fulfill prebuild/postbuild directly from host DXR
/// (`VICTORY_NATIVE_RT_SUPPRESS_BUILDER=1`, default OFF during dual-run bring-up).
pub fn suppress_builder() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        enabled()
            && std::env::var_os("VICTORY_NATIVE_RT_SUPPRESS_BUILDER")
                .is_some_and(|v| v == "1")
    })
}

/// Whether opaque static BLAS geometries should also set
/// `D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION`
/// (`VICTORY_NATIVE_RT_NO_DUP_OPAQUE=1`, default OFF) for A/B testing
/// `FORCE_NON_OPAQUE` ray-query loops.
fn no_dup_opaque() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_NATIVE_RT_NO_DUP_OPAQUE").is_some_and(|v| v == "1")
    })
}

/// Whether static BLAS builds set `ALLOW_COMPACTION`, query postbuild compacted sizes,
/// and asynchronously compact into smaller slab slices (`VICTORY_NATIVE_RT_COMPACT=0` disables).
fn compact_static_blas() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_NATIVE_RT_COMPACT").is_none_or(|v| v != "0")
    })
}

fn static_blas_build_flags() -> D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS {
    if compact_static_blas() {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION
    } else {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
    }
}

/// Returns `true` when compacting a static BLAS from `uncompacted_size` to
/// `compacted_size` (rounded up to 256B `AS_ALIGN`) saves meaningful slab VRAM.
fn should_compact(uncompacted_size: u64, compacted_size: u64) -> bool {
    if compacted_size == 0 {
        return false;
    }
    let aligned = compacted_size.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
    let Some(saved) = uncompacted_size.checked_sub(aligned) else {
        return false;
    };
    saved > 4096 && (saved.saturating_mul(10) >= uncompacted_size || saved >= 64 * 1024)
}

fn static_geom_flags(opaque: bool) -> D3D12_RAYTRACING_GEOMETRY_FLAGS {
    if opaque {
        if no_dup_opaque() {
            D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
                | D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION
        } else {
            D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
        }
    } else {
        D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION
    }
}

fn dynamic_geom_flags(raw_flags: u32) -> D3D12_RAYTRACING_GEOMETRY_FLAGS {
    let opaque = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE.0 as u32;
    let no_dup = D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION.0 as u32;
    let flags = if (raw_flags & opaque) == 0 {
        raw_flags | no_dup
    } else {
        raw_flags
    };
    D3D12_RAYTRACING_GEOMETRY_FLAGS(flags as i32)
}

/// Guest `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS` (`0x18` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestBuildAsInputs {
    pub kind: u32,
    pub flags: u32,
    pub num_descs: u32,
    pub descs_layout: u32,
    pub ptr_or_va: u64,
}
const _: () = assert!(size_of::<GuestBuildAsInputs>() == 0x18);

/// Guest `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC` (`0x30` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestBuildAsDesc {
    pub dest_va: u64,
    pub inputs: GuestBuildAsInputs,
    pub source_va: u64,
    pub scratch_va: u64,
}
const _: () = assert!(size_of::<GuestBuildAsDesc>() == 0x30);
const _: () = assert!(offset_of!(GuestBuildAsDesc, inputs) == 0x08);
const _: () = assert!(offset_of!(GuestBuildAsDesc, source_va) == 0x20);
const _: () = assert!(offset_of!(GuestBuildAsDesc, scratch_va) == 0x28);

/// Guest `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC` (`0x10` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestPostbuildInfoDesc {
    pub dest_buffer: u64,
    pub info_type: u32,
    pub _pad: u32,
}
const _: () = assert!(size_of::<GuestPostbuildInfoDesc>() == 0x10);

/// Guest `D3D12XBOX_XDXR_STREAM_DESC` (`0x10` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestStreamDesc {
    pub size_in_bytes: u64,
    pub data: *const c_void,
}
const _: () = assert!(size_of::<GuestStreamDesc>() == 0x10);

/// Guest `D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2` (`0x50` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestBatchedDesc2 {
    pub desc: GuestBuildAsDesc,
    pub num_postbuild_info_descs: u32,
    pub _pad: u32,
    pub postbuild_info_descs: *const GuestPostbuildInfoDesc,
    pub stream_desc: GuestStreamDesc,
}
const _: () = assert!(size_of::<GuestBatchedDesc2>() == 0x50);
const _: () = assert!(offset_of!(GuestBatchedDesc2, num_postbuild_info_descs) == 0x30);
const _: () = assert!(offset_of!(GuestBatchedDesc2, postbuild_info_descs) == 0x38);
const _: () = assert!(offset_of!(GuestBatchedDesc2, stream_desc) == 0x40);

/// Guest `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO` (`0x18` bytes).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GuestPrebuildInfo {
    pub result_data_max_size_in_bytes: u64,
    pub scratch_data_size_in_bytes: u64,
    pub update_scratch_data_size_in_bytes: u64,
}
const _: () = assert!(size_of::<GuestPrebuildInfo>() == 0x18);

/// Guest `D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC` (`0x30` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestGeometryTrianglesDesc {
    pub transform3x4: u64,
    pub index_format: u32,
    pub vertex_format: u32,
    pub index_count: u32,
    pub vertex_count: u32,
    pub index_buffer: u64,
    pub vertex_buffer_va: u64,
    pub vertex_buffer_stride: u64,
}
const _: () = assert!(size_of::<GuestGeometryTrianglesDesc>() == 0x30);

/// Guest `D3D12_RAYTRACING_GEOMETRY_AABBS_DESC` (`0x18` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestGeometryAabbsDesc {
    pub aabb_count: u64,
    pub aabbs_va: u64,
    pub aabbs_stride: u64,
}
const _: () = assert!(size_of::<GuestGeometryAabbsDesc>() == 0x18);

#[repr(C)]
#[derive(Clone, Copy)]
pub union GuestGeometryUnion {
    pub triangles: GuestGeometryTrianglesDesc,
    pub aabbs: GuestGeometryAabbsDesc,
}

/// Guest `D3D12_RAYTRACING_GEOMETRY_DESC` (`0x38` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestGeometryDesc {
    pub kind: u32,
    pub flags: u32,
    pub data: GuestGeometryUnion,
}
const _: () = assert!(size_of::<GuestGeometryDesc>() == 0x38);

/// Owned geometry specification copied out of guest memory during command recording.
#[derive(Clone)]
pub(crate) enum GeometryKind {
    Triangles {
        transform3x4: u64,
        index_format: u32,
        vertex_format: u32,
        index_count: u32,
        vertex_count: u32,
        index_buffer: u64,
        vertex_buffer_va: u64,
        vertex_buffer_stride: u64,
    },
    Aabbs {
        aabb_count: u64,
        aabbs_va: u64,
        aabbs_stride: u64,
    },
}

#[derive(Clone)]
pub(crate) struct GeometrySpec {
    pub(crate) flags: u32,
    pub(crate) kind: GeometryKind,
}

#[derive(Clone, Copy)]
pub(crate) struct PostbuildSpec {
    pub(crate) dest_buffer: u64,
    pub(crate) info_type: u32,
}

/// Queued DXR operation attached to a guest command list and replayed on `victory-submit`.
#[derive(Clone)]
pub(crate) enum DxrOp {
    BuildBlas {
        dest_va: u64,
        source_va: u64,
        flags: u32,
        geometries: Vec<GeometrySpec>,
        postbuild: Vec<PostbuildSpec>,
    },
    BuildTlas {
        dest_va: u64,
        source_va: u64,
        flags: u32,
        num_instances: u32,
        instance_descs_va: u64,
        postbuild: Vec<PostbuildSpec>,
    },
    CopyAs {
        dest_va: u64,
        source_va: u64,
        mode: u32,
    },
    ExtractStaticBlas {
        dest_va: u64,
        size_bytes: u64,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AsKind {
    Blas,
    Tlas,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlabSlice {
    slab_idx: usize,
    offset: u64,
    size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RetiredSlice {
    release_after_tlas_epoch: u64,
    release_after_sub_seq: u64,
    slice: SlabSlice,
}

#[derive(Clone)]
struct HostAs {
    buffer: ID3D12Resource,
    slice: SlabSlice,
    gpu_va: u64,
    kind: AsKind,
    flags: u32,
    num_instances: u32,
    built: bool,
    generation: u64,
    topo_hash: u64,
    update_scratch_bytes: u64,
}

// SAFETY: `ID3D12Resource` is free-threaded; entries are synchronized by `RwLock`.
unsafe impl Send for HostAs {}
unsafe impl Sync for HostAs {}

struct Slab<B = ID3D12Resource> {
    buffer: B,
    gpu_base: u64,
    size: u64,
    used: u64,
    /// Sorted, non-overlapping, non-adjacent `(offset, size)` free ranges below `used`.
    free: Vec<(u64, u64)>,
}

// SAFETY: `ID3D12Resource` is free-threaded; accessed under `DXR_STATE` mutex.
unsafe impl Send for Slab<ID3D12Resource> {}

struct SlabAllocator<B = ID3D12Resource> {
    slabs: Vec<Slab<B>>,
    retired: VecDeque<RetiredSlice>,
}

impl<B> Default for SlabAllocator<B> {
    fn default() -> Self {
        Self {
            slabs: Vec::new(),
            retired: VecDeque::new(),
        }
    }
}

impl<B: Clone> SlabAllocator<B> {
    fn alloc_with<F>(&mut self, size: u64, make_slab: F) -> Option<(B, SlabSlice, u64)>
    where
        F: FnOnce(u64, usize) -> Option<(B, u64)>,
    {
        let need = size.max(AS_ALIGN).next_multiple_of(AS_ALIGN);

        // Best-fit search across reclaimed free ranges before bump-allocating.
        let mut best: Option<(usize, usize, u64)> = None;
        'search: for (slab_idx, slab) in self.slabs.iter().enumerate() {
            for (free_idx, &(_, range_size)) in slab.free.iter().enumerate() {
                if range_size >= need {
                    if best.is_none_or(|(_, _, best_sz)| range_size < best_sz) {
                        best = Some((slab_idx, free_idx, range_size));
                        if range_size == need {
                            break 'search;
                        }
                    }
                }
            }
        }
        if let Some((slab_idx, free_idx, _)) = best {
            let slab = &mut self.slabs[slab_idx];
            let (offset, range_size) = slab.free[free_idx];
            if range_size == need {
                slab.free.remove(free_idx);
            } else {
                slab.free[free_idx] = (offset + need, range_size - need);
            }
            let slice = SlabSlice {
                slab_idx,
                offset,
                size: need,
            };
            return Some((slab.buffer.clone(), slice, slab.gpu_base + offset));
        }

        // Fall back to bump allocation in an existing slab's tail.
        if let Some((slab_idx, slab)) = self
            .slabs
            .iter_mut()
            .enumerate()
            .rev()
            .find(|(_, s)| s.used + need <= s.size)
        {
            let offset = slab.used;
            slab.used += need;
            let slice = SlabSlice {
                slab_idx,
                offset,
                size: need,
            };
            return Some((slab.buffer.clone(), slice, slab.gpu_base + offset));
        }

        // Allocate a fresh backing slab.
        let slab_size = need.max(AS_SLAB_SIZE).next_multiple_of(AS_ALIGN);
        let slab_idx = self.slabs.len();
        let (buffer, gpu_base) = make_slab(slab_size, slab_idx)?;
        self.slabs.push(Slab {
            buffer: buffer.clone(),
            gpu_base,
            size: slab_size,
            used: need,
            free: Vec::new(),
        });
        let slice = SlabSlice {
            slab_idx,
            offset: 0,
            size: need,
        };
        Some((buffer, slice, gpu_base))
    }

    fn retire(&mut self, slice: SlabSlice, release_after_tlas_epoch: u64, release_after_sub_seq: u64) {
        if slice.size == 0 {
            return;
        }
        self.retired.push_back(RetiredSlice {
            release_after_tlas_epoch,
            release_after_sub_seq,
            slice,
        });
    }

    fn drain_retired(&mut self, completed_tlas_epoch: u64, completed_sub_seq: u64) -> usize {
        let mut freed = 0usize;
        let slabs = &mut self.slabs;
        self.retired.retain(|r| {
            if completed_tlas_epoch >= r.release_after_tlas_epoch && completed_sub_seq >= r.release_after_sub_seq {
                Self::free_in_slabs(slabs, r.slice);
                freed += 1;
                false
            } else {
                true
            }
        });
        freed
    }

    #[cfg(test)]
    fn free(&mut self, slice: SlabSlice) {
        Self::free_in_slabs(&mut self.slabs, slice);
    }

    fn free_in_slabs(slabs: &mut [Slab<B>], slice: SlabSlice) {
        if slice.size == 0 {
            return;
        }
        let Some(slab) = slabs.get_mut(slice.slab_idx) else {
            return;
        };
        let mut start = slice.offset;
        let mut end = match slice.offset.checked_add(slice.size) {
            Some(e) if e <= slab.used => e,
            _ => {
                debug_assert!(false, "dxr: slab free out of bounds");
                return;
            }
        };
        let idx = slab.free.partition_point(|&(off, _)| off < start);
        if idx > 0 {
            let (p_off, p_sz) = slab.free[idx - 1];
            if p_off + p_sz > start {
                debug_assert!(false, "dxr: overlapping slab free with previous range");
                return;
            }
        }
        if idx < slab.free.len() && end > slab.free[idx].0 {
            debug_assert!(false, "dxr: overlapping slab free with next range");
            return;
        }

        let merge_prev = idx > 0 && slab.free[idx - 1].0 + slab.free[idx - 1].1 == start;
        let merge_next = idx < slab.free.len() && end == slab.free[idx].0;
        let insert_idx = if merge_prev {
            start = slab.free[idx - 1].0;
            idx - 1
        } else {
            idx
        };
        if merge_next {
            end = slab.free[idx].0 + slab.free[idx].1;
        }

        match (merge_prev, merge_next) {
            (true, true) => {
                slab.free.remove(idx);
                if end == slab.used {
                    slab.free.remove(insert_idx);
                    slab.used = start;
                } else {
                    slab.free[insert_idx] = (start, end - start);
                }
            }
            (true, false) => {
                if end == slab.used {
                    slab.free.remove(insert_idx);
                    slab.used = start;
                } else {
                    slab.free[insert_idx] = (start, end - start);
                }
            }
            (false, true) => {
                if end == slab.used {
                    slab.free.remove(idx);
                    slab.used = start;
                } else {
                    slab.free[idx] = (start, end - start);
                }
            }
            (false, false) => {
                if end == slab.used {
                    slab.used = start;
                } else {
                    slab.free.insert(idx, (start, end - start));
                }
            }
        }
    }

    /// Returns `(total_allocated_bytes, available_bytes)` where `available_bytes`
    /// is free-list ranges plus unused bump tails across all slabs.
    fn stats(&self) -> (u64, u64) {
        let mut total = 0u64;
        let mut free = 0u64;
        for s in &self.slabs {
            total = total.saturating_add(s.size);
            let tail = s.size.saturating_sub(s.used);
            let recycled: u64 = s.free.iter().map(|&(_, sz)| sz).sum();
            free = free.saturating_add(tail).saturating_add(recycled);
        }
        (total, free)
    }
}

static DXR_SLAB_BYTES: AtomicU64 = AtomicU64::new(0);

/// Local VRAM reserved by the texture queue for Native DXR AS slabs not yet
/// allocated (`VICTORY_CARD_ROOM_FOR_DXR_MB`, default 1,536 MiB when
/// `VICTORY_NATIVE_RT=1`, 0 otherwise), releasing automatically as slabs grow.
pub(crate) fn dxr_room_kept() -> u64 {
    if !enabled() {
        return 0;
    }
    static MOST: OnceLock<u64> = OnceLock::new();
    let most = *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_ROOM_FOR_DXR_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(1536)
            << 20
    });
    most.saturating_sub(DXR_SLAB_BYTES.load(Ordering::Relaxed))
}

impl SlabAllocator<ID3D12Resource> {
    fn alloc(&mut self, host: &Host, size: u64) -> Option<(ID3D12Resource, SlabSlice, u64)> {
        self.alloc_with(size, |slab_size, slab_idx| {
            let buffer = create_as_slab(host, slab_size, slab_idx)?;
            DXR_SLAB_BYTES.fetch_add(slab_size, Ordering::Relaxed);
            // SAFETY: live committed buffer resource.
            let gpu_base = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, gpu_base))
        })
    }
}

fn create_as_slab(host: &Host, size: u64, index: usize) -> Option<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain committed AS slab creation in the required initial state.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            not_resident(),
            &desc,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            None,
            &mut buffer,
        )
    };
    let made = made
        .and_then(|()| buffer.as_ref().map_or(Ok(()), |b| resident(&host.device, b)))
        .and_then(|()| mapped(&host.device));
    match made {
        Ok(()) => {
            let buf = buffer?;
            name(&buf, || format!("victory dxr as slab {index} ({} MiB)", size >> 20));
            Some(buf)
        }
        Err(e) => {
            tracing::warn!(error = %e, size, "dxr: failed to allocate AS slab");
            None
        }
    }
}

struct StaticReadbackRecord {
    guest_key: u64,
    generation: u64,
    offset: u64,
    len: u64,
}

struct CompactQueryRecord {
    guest_key: u64,
    generation: u64,
    uncompacted_size: u64,
    slot: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UpgradedGeom {
    vert_offset: u32,
    vert_count: u32,
    opaque: bool,
}

#[derive(Debug)]
struct PendingStaticUpgrade {
    guest_key: u64,
    generation: u64,
    geometries: Vec<UpgradedGeom>,
    vertices: Vec<[f32; 3]>,
}

/// Host resources and pooled command lists owned by one in-flight submission.
pub(crate) struct DxrSubmission {
    lists: Vec<Prelude>,
    scratch: Option<(ID3D12Resource, u64)>,
    host_instances: Option<(ID3D12Resource, u64)>,
    extract_verts: Option<(ID3D12Resource, u64)>,
    hash_upload: Option<(ID3D12Resource, usize)>,
    /// Temporary `DEFAULT` UAV buffer for unpacked 64-byte TriangleNode dumps.
    node_staging: Option<(ID3D12Resource, u64)>,
    /// `READBACK` buffer receiving packed per-BLAS node dumps for this submission.
    node_readback: Option<(ID3D12Resource, u64)>,
    readback_used: u64,
    readback_cap: u64,
    readbacks: Vec<StaticReadbackRecord>,
    /// Pooled upload + default vertex buffers for draining `pending_static_upgrades`.
    upgrade_upload: Option<(ID3D12Resource, usize, u64)>,
    upgrade_verts: Option<(ID3D12Resource, u64)>,
    /// Pooled 64 KiB `DEFAULT` UAV and `READBACK` buffers for 8-byte compacted-size queries.
    compact_info_uav: Option<(ID3D12Resource, u64)>,
    compact_info_rb: Option<(ID3D12Resource, u64)>,
    compact_queries: Vec<CompactQueryRecord>,
    kept: Vec<ID3D12Resource>,
    /// Keeps full `Shadow` objects (including any backing `ID3D12Heap` for placed
    /// in-place buffers) alive until this submission's GPU fence completes.
    kept_shadows: Vec<Arc<crate::shadow::Shadow>>,
    /// Monotonic DXR submission sequence number assigned when submitted (0 if unused).
    sub_seq: u64,
    /// Monotonic `tlas_build_epoch` achieved by this submission (0 if no `BuildTlas` ran).
    tlas_epoch: u64,
}

// SAFETY: D3D12 objects are free-threaded; a `DxrSubmission` has one owner at a time.
unsafe impl Send for DxrSubmission {}

impl DxrSubmission {
    fn new() -> Self {
        Self {
            lists: Vec::new(),
            scratch: None,
            host_instances: None,
            extract_verts: None,
            hash_upload: None,
            node_staging: None,
            node_readback: None,
            readback_used: 0,
            readback_cap: 0,
            readbacks: Vec::new(),
            upgrade_upload: None,
            upgrade_verts: None,
            compact_info_uav: None,
            compact_info_rb: None,
            compact_queries: Vec::new(),
            kept: Vec::new(),
            kept_shadows: Vec::new(),
            sub_seq: 0,
            tlas_epoch: 0,
        }
    }

    fn ensure_scratch(&mut self, host: &Host, need: u64) -> Option<(&ID3D12Resource, u64)> {
        let want = need.max(MIN_SCRATCH_SIZE).next_multiple_of(AS_ALIGN);
        if self.scratch.as_ref().is_none_or(|(_, sz)| *sz < want) {
            if let Some((old, _)) = self.scratch.take() {
                self.kept.push(old);
            }
            let buf = host.uav_buffer(want)?;
            name(&buf, || format!("victory dxr scratch ({} MiB)", want >> 20));
            self.scratch = Some((buf, want));
        }
        let (buf, _) = self.scratch.as_ref()?;
        // SAFETY: live buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        Some((buf, gpu))
    }

    fn ensure_host_instances(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, u64)> {
        let want = need.max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.host_instances.as_ref().is_none_or(|(_, sz)| *sz < want) {
            if let Some((old, _)) = self.host_instances.take() {
                self.kept.push(old);
            }
            let buf = host.uav_buffer(want)?;
            name(&buf, || format!("victory dxr host instances ({} KiB)", want >> 10));
            self.host_instances = Some((buf, want));
        }
        let (buf, _) = self.host_instances.as_ref()?;
        // SAFETY: live buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        Some((buf.clone(), gpu))
    }

    fn ensure_extract_verts(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, u64)> {
        let want = need.max(256 * 1024).next_multiple_of(AS_ALIGN);
        if self.extract_verts.as_ref().is_none_or(|(_, sz)| *sz < want) {
            if let Some((old, _)) = self.extract_verts.take() {
                self.kept.push(old);
            }
            let buf = host.uav_buffer(want)?;
            name(&buf, || format!("victory dxr extracted verts ({} KiB)", want >> 10));
            self.extract_verts = Some((buf, want));
        }
        let (buf, _) = self.extract_verts.as_ref()?;
        // SAFETY: live buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        Some((buf.clone(), gpu))
    }

    fn ensure_hash_upload(&mut self, host: &Host) -> Option<(ID3D12Resource, *mut u8)> {
        if self.hash_upload.is_none() {
            let (buf, ptr) = host.upload_buffer(HASH_TABLE_BYTES)?;
            name(&buf, || "victory dxr blas hash upload".into());
            self.hash_upload = Some((buf, ptr as usize));
        }
        let (buf, ptr) = self.hash_upload.as_ref()?;
        Some((buf.clone(), *ptr as *mut u8))
    }

    fn ensure_node_staging(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, u64)> {
        let want = need.max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.node_staging.as_ref().is_none_or(|(_, sz)| *sz < want) {
            if let Some((old, _)) = self.node_staging.take() {
                self.kept.push(old);
            }
            let buf = host.uav_buffer(want)?;
            name(&buf, || format!("victory dxr node staging ({} KiB)", want >> 10));
            self.node_staging = Some((buf, want));
        }
        let (buf, _) = self.node_staging.as_ref()?;
        // SAFETY: live buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        Some((buf.clone(), gpu))
    }

    fn prepare_node_readback(&mut self, host: &Host, need: u64) {
        self.readbacks.clear();
        self.readback_used = 0;
        self.readback_cap = 0;
        if need == 0 {
            return;
        }
        let want = need.min(MAX_READBACK_BYTES_PER_SUB).max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.node_readback.as_ref().is_none_or(|(_, sz)| *sz < want) {
            if let Some((old, _)) = self.node_readback.take() {
                self.kept.push(old);
            }
            if let Some(buf) = host.read_back_buffer(want) {
                name(&buf, || format!("victory dxr node readback ({} KiB)", want >> 10));
                self.node_readback = Some((buf, want));
            }
        }
        if let Some((_, sz)) = self.node_readback.as_ref() {
            self.readback_cap = (*sz).min(MAX_READBACK_BYTES_PER_SUB);
        }
    }

    fn ensure_upgrade_upload(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let want = need.max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.upgrade_upload.as_ref().is_none_or(|(_, _, sz)| *sz < want) {
            if let Some((old, _, _)) = self.upgrade_upload.take() {
                self.kept.push(old);
            }
            let (buf, ptr) = host.upload_buffer(want)?;
            name(&buf, || format!("victory dxr upgrade upload ({} KiB)", want >> 10));
            self.upgrade_upload = Some((buf, ptr as usize, want));
        }
        let (buf, ptr, _) = self.upgrade_upload.as_ref()?;
        Some((buf.clone(), *ptr as *mut u8))
    }

    fn ensure_upgrade_verts(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, u64)> {
        let want = need.max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.upgrade_verts.as_ref().is_none_or(|(_, sz)| *sz < want) {
            if let Some((old, _)) = self.upgrade_verts.take() {
                self.kept.push(old);
            }
            let buf = host.uav_buffer(want)?;
            name(&buf, || format!("victory dxr upgrade verts ({} KiB)", want >> 10));
            self.upgrade_verts = Some((buf, want));
        }
        let (buf, _) = self.upgrade_verts.as_ref()?;
        // SAFETY: live buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        Some((buf.clone(), gpu))
    }

    fn ensure_compact_info(&mut self, host: &Host) -> Option<(ID3D12Resource, u64, ID3D12Resource)> {
        if self.compact_info_uav.is_none() {
            let buf = host.uav_buffer(COMPACT_INFO_BYTES)?;
            name(&buf, || "victory dxr compact info uav".into());
            self.compact_info_uav = Some((buf, COMPACT_INFO_BYTES));
        }
        if self.compact_info_rb.is_none() {
            let buf = host.read_back_buffer(COMPACT_INFO_BYTES)?;
            name(&buf, || "victory dxr compact info readback".into());
            self.compact_info_rb = Some((buf, COMPACT_INFO_BYTES));
        }
        let (uav, _) = self.compact_info_uav.as_ref()?;
        let (rb, _) = self.compact_info_rb.as_ref()?;
        // SAFETY: live UAV buffer.
        let uav_gpu = unsafe { uav.GetGPUVirtualAddress() };
        Some((uav.clone(), uav_gpu, rb.clone()))
    }
}

/// Tracks modified slots in `DxrState::blas_table` so `record_build_tlas` copies only
/// dirty slot runs (or `[min_slot..=max_slot]` on bursts) instead of the full 1 MiB table.
struct HashDirtyTracker {
    full: bool,
    has_range: bool,
    overflowed: bool,
    min_slot: usize,
    max_slot: usize,
    slots: Vec<usize>,
}

impl HashDirtyTracker {
    fn full() -> Self {
        Self {
            full: true,
            has_range: false,
            overflowed: false,
            min_slot: 0,
            max_slot: 0,
            slots: Vec::new(),
        }
    }

    fn is_dirty(&self) -> bool {
        self.full || self.has_range
    }

    fn mark_full(&mut self) {
        self.full = true;
        self.has_range = false;
        self.overflowed = false;
        self.slots.clear();
    }

    fn mark_slot(&mut self, slot: usize) {
        if self.full {
            return;
        }
        if !self.has_range {
            self.has_range = true;
            self.min_slot = slot;
            self.max_slot = slot;
        } else {
            self.min_slot = self.min_slot.min(slot);
            self.max_slot = self.max_slot.max(slot);
        }
        if !self.overflowed {
            if self.slots.len() < MAX_DIRTY_HASH_SLOTS {
                self.slots.push(slot);
            } else {
                self.slots.clear();
                self.overflowed = true;
            }
        }
    }

    /// Returns inclusive `(start_slot, end_slot)` dirty ranges and clears the tracker.
    fn take_ranges(&mut self) -> Vec<(usize, usize)> {
        if self.full {
            self.full = false;
            self.has_range = false;
            self.overflowed = false;
            self.slots.clear();
            return vec![(0, HASH_CAPACITY - 1)];
        }
        if !self.has_range {
            return Vec::new();
        }
        self.has_range = false;
        if self.overflowed {
            self.overflowed = false;
            self.slots.clear();
            return vec![(self.min_slot, self.max_slot)];
        }
        self.slots.sort_unstable();
        self.slots.dedup();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for &slot in &self.slots {
            if let Some(last) = ranges.last_mut()
                && slot.saturating_sub(last.1) <= HASH_DIRTY_MERGE_GAP
            {
                last.1 = slot;
            } else {
                ranges.push((slot, slot));
            }
        }
        self.slots.clear();
        ranges
    }
}

struct DxrState {
    slabs: SlabAllocator,
    /// Open-addressed CPU mirror of `(guest_blas_va >> 8 -> host_blas_va)`.
    /// Each slot is `[key_lo, key_hi, val_lo, val_hi]`.
    blas_table: Vec<[u32; 4]>,
    hash_dirty: HashDirtyTracker,
    hash_gpu: Option<ID3D12Resource>,
    /// Mapped shared UAV buffer `[patched_ok, missing_blas, null_blas, total]` for diagnostics.
    patch_stats: Option<(ID3D12Resource, usize, u64)>,
    /// Monotonic generation stamped on `HostAs` entries to reject stale async upgrades.
    next_gen: u64,
    /// Monotonic submission sequence numbers (submitted vs GPU-completed).
    sub_seq: u64,
    completed_sub_seq: u64,
    /// Monotonic TLAS rebuild epochs (submitted vs GPU-completed) gating BLAS slice reclamation.
    tlas_build_epoch: u64,
    completed_tlas_epoch: u64,
    /// Cached dynamic BLAS `GetRaytracingAccelerationStructurePrebuildInfo` keyed by `(flags, topo_hash)`
    /// -> `(ResultDataMaxSizeInBytes, ScratchDataSizeInBytes, UpdateScratchDataSizeInBytes)`.
    prebuild_cache: HashMap<(u32, u64), (u64, u64, u64)>,
    /// Newly pre-allocated TLAS VAs that have not been built yet; initialized with an empty
    /// 0-instance TLAS before any shader can trace them.
    pending_empty_tlas: Vec<u64>,
    /// Decoded multi-geometry static BLAS upgrades waiting for GPU build on the next submission.
    pending_static_upgrades: VecDeque<PendingStaticUpgrade>,
    /// Static BLASes `(guest_key, generation, compacted_size)` awaiting async compaction copy.
    pending_compactions: VecDeque<(u64, u64, u64)>,
    /// Pooled command lists and reusable submission containers returned after GPU completion.
    free_lists: Vec<Prelude>,
    free_submissions: Vec<DxrSubmission>,
}

// SAFETY: D3D12 objects are free-threaded; accessed strictly under `DXR_STATE` mutex.
unsafe impl Send for DxrState {}

impl DxrState {
    fn new() -> Self {
        Self {
            slabs: SlabAllocator::default(),
            blas_table: vec![[0u32; 4]; HASH_CAPACITY],
            hash_dirty: HashDirtyTracker::full(),
            hash_gpu: None,
            patch_stats: None,
            next_gen: 0,
            sub_seq: 0,
            completed_sub_seq: 0,
            tlas_build_epoch: 0,
            completed_tlas_epoch: 0,
            prebuild_cache: HashMap::new(),
            pending_empty_tlas: Vec::new(),
            pending_static_upgrades: VecDeque::new(),
            pending_compactions: VecDeque::new(),
            free_lists: Vec::new(),
            free_submissions: Vec::new(),
        }
    }

    fn bump_gen(&mut self) -> u64 {
        self.next_gen = self.next_gen.wrapping_add(1).max(1);
        self.next_gen
    }

    /// Defers reclamation of a replaced AS slab slice until it is guaranteed unreachable
    /// on the GPU (at least 2 completed TLAS rebuild epochs for BLAS slices, or 2 completed
    /// submissions for non-TLAS-referenced slices like replaced TLASes).
    fn retire_as(&mut self, old: &HostAs) {
        let release_after_sub_seq = self.sub_seq.saturating_add(2);
        let release_after_tlas_epoch = if old.kind == AsKind::Blas {
            self.tlas_build_epoch.saturating_add(2)
        } else {
            0
        };
        self.slabs
            .retire(old.slice, release_after_tlas_epoch, release_after_sub_seq);
    }

    fn insert_blas_hash(&mut self, guest_va: u64, host_va: u64) {
        let key = guest_va >> 8;
        if key == 0 {
            return;
        }
        let key_lo = key as u32;
        let key_hi = (key >> 32) as u32;
        let val_lo = host_va as u32;
        let val_hi = (host_va >> 32) as u32;
        let start = hash_key(key_lo, key_hi) & HASH_MASK;
        for step in 0..64u32 {
            let idx = ((start + step) & HASH_MASK) as usize;
            let entry = &mut self.blas_table[idx];
            if (entry[0] == key_lo && entry[1] == key_hi) || (entry[0] | entry[1]) == 0 {
                if *entry != [key_lo, key_hi, val_lo, val_hi] {
                    *entry = [key_lo, key_hi, val_lo, val_hi];
                    self.hash_dirty.mark_slot(idx);
                }
                return;
            }
        }
        // shortcut: hash table saturated past 64-probe chain; overwrite initial bucket,
        // upgrade to dynamic resize if active BLASes ever approach 48K.
        let idx = start as usize;
        self.blas_table[idx] = [key_lo, key_hi, val_lo, val_hi];
        self.hash_dirty.mark_slot(idx);
    }

    fn ensure_hash_gpu(&mut self, host: &Host) -> Option<ID3D12Resource> {
        if self.hash_gpu.is_none() {
            let buf = host.uav_buffer(HASH_TABLE_BYTES)?;
            name(&buf, || "victory dxr blas hash table".into());
            self.hash_gpu = Some(buf);
            self.hash_dirty.mark_full();
        }
        self.hash_gpu.clone()
    }

    fn ensure_patch_stats(&mut self, host: &Host) -> Option<(ID3D12Resource, u64)> {
        if self.patch_stats.is_none() {
            let (buf, ptr) = host.shared_buffer(64)?;
            // SAFETY: fresh mapped 64-byte buffer.
            unsafe { std::ptr::write_bytes(ptr, 0, 64) };
            // SAFETY: live buffer.
            let gpu = unsafe { buf.GetGPUVirtualAddress() };
            name(&buf, || "victory dxr patch stats".into());
            self.patch_stats = Some((buf, ptr as usize, gpu));
        }
        let (buf, _, gpu) = self.patch_stats.as_ref()?;
        Some((buf.clone(), *gpu))
    }

    fn take_list(&mut self, host: &Host) -> Option<Prelude> {
        self.free_lists.pop().or_else(|| host.make_prelude())
    }

    fn take_submission(&mut self) -> DxrSubmission {
        self.free_submissions.pop().unwrap_or_else(DxrSubmission::new)
    }
}

fn hash_key(lo: u32, hi: u32) -> u32 {
    let mut x = lo ^ hi.wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    x
}

static AS_MAP: OnceLock<RwLock<HashMap<u64, HostAs>>> = OnceLock::new();
static DXR_STATE: Mutex<Option<DxrState>> = Mutex::new(None);

fn as_map() -> &'static RwLock<HashMap<u64, HostAs>> {
    AS_MAP.get_or_init(|| RwLock::new(HashMap::new()))
}

fn lock_state() -> std::sync::MutexGuard<'static, Option<DxrState>> {
    let mut guard = DXR_STATE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(DxrState::new());
    }
    guard
}

/// Diagnostic counters: `[dynamic_blas, static_blas, upgraded_static, tlas, copies, skipped, tlas_srvs, upg_geoms, upg_non_opaque, compacted]`.
static STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
const STAT_DYNAMIC_BLAS: usize = 0;
const STAT_STATIC_BLAS: usize = 1;
const STAT_UPGRADED_STATIC: usize = 2;
const STAT_TLAS: usize = 3;
const STAT_COPIES: usize = 4;
const STAT_SKIPPED: usize = 5;
const STAT_TLAS_SRVS: usize = 6;
const STAT_UPG_GEOMS: usize = 7;
const STAT_UPG_NON_OPAQUE: usize = 8;
const STAT_COMPACTED: usize = 9;

/// Logs native DXR statistics periodically alongside the host frame-rate line.
pub(crate) fn log_stats(frames: f64) {
    if !enabled() || frames <= 0.0 {
        return;
    }
    let [dyn_blas, stat_blas, upgraded, tlas, copies, skipped, srvs, upg_geoms, upg_non_opaque, compacted] =
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(|i| STATS[i].swap(0, Ordering::Relaxed));
    let total_as = as_map().read().unwrap_or_else(|e| e.into_inner()).len();
    let (patched, missing, nulls, slab_total, slab_free) = {
        let guard = DXR_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let (slab_total, slab_free) = guard.as_ref().map_or((0, 0), |s| s.slabs.stats());
        let (p, m, n) = if let Some((_, ptr, _)) = guard.as_ref().and_then(|s| s.patch_stats.as_ref()) {
            let base = *ptr as *mut u32;
            // SAFETY: mapped 64-byte shared buffer written by GPU atomics.
            unsafe {
                let p = base.add(0).read_volatile();
                let m = base.add(1).read_volatile();
                let n = base.add(2).read_volatile();
                base.add(0).write_volatile(0);
                base.add(1).write_volatile(0);
                base.add(2).write_volatile(0);
                base.add(3).write_volatile(0);
                (u64::from(p), u64::from(m), u64::from(n))
            }
        } else {
            (0, 0, 0)
        };
        (p, m, n, slab_total, slab_free)
    };
    tracing::info!(
        "native DXR per frame: dyn_blas={:.1} static_blas={:.1} upgraded_static={:.1} compacted={:.1} tlas={:.1} copies={:.1} skipped={:.1} tlas_srvs={:.1} \
         (instances patched={:.0} missing_blas={:.0} null_blas={:.0}; resident_as={total_as}; slab_mb={:.1} slab_free_mb={:.1}; upg_geoms={upg_geoms} non_opaque={upg_non_opaque})",
        dyn_blas as f64 / frames,
        stat_blas as f64 / frames,
        upgraded as f64 / frames,
        compacted as f64 / frames,
        tlas as f64 / frames,
        copies as f64 / frames,
        skipped as f64 / frames,
        srvs as f64 / frames,
        patched as f64 / frames,
        missing as f64 / frames,
        nulls as f64 / frames,
        slab_total as f64 / (1024.0 * 1024.0),
        slab_free as f64 / (1024.0 * 1024.0),
    );
}

/// Sentinel guest key for the shared valid 0-instance empty TLAS used whenever
/// a shader binds a null (`Location = 0`) RTAS SRV.
const EMPTY_TLAS_KEY: u64 = u64::MAX & !0xFF;

/// Ensures `guest_va` has a stable host TLAS slot of at least `min_bytes`.
pub(crate) fn ensure_tlas_slot(guest_va: u64, min_bytes: u64) -> Option<(u64, ID3D12Resource)> {
    let key = guest_va & !0xFF;
    if key == 0 {
        return None;
    }
    {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = map.get(&key)
            && entry.kind == AsKind::Tlas
            && entry.slice.size >= min_bytes
        {
            return Some((entry.gpu_va, entry.buffer.clone()));
        }
    }
    let host = host()?;
    let mut state_guard = lock_state();
    let state = state_guard.as_mut()?;
    let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = map.get(&key)
        && entry.kind == AsKind::Tlas
        && entry.slice.size >= min_bytes
    {
        return Some((entry.gpu_va, entry.buffer.clone()));
    }
    let reserve = if key == EMPTY_TLAS_KEY {
        min_bytes.max(4096)
    } else {
        min_bytes.max(DEFAULT_TLAS_RESERVE)
    };
    let (buffer, slice, gpu_va) = state.slabs.alloc(host, reserve)?;
    let generation = state.bump_gen();
    if let Some(old) = map.insert(
        key,
        HostAs {
            buffer: buffer.clone(),
            slice,
            gpu_va,
            kind: AsKind::Tlas,
            flags: 0,
            num_instances: 0,
            built: false,
            generation,
            topo_hash: 0,
            update_scratch_bytes: 0,
        },
    ) {
        state.retire_as(&old);
    }
    if !state.pending_empty_tlas.contains(&key) {
        state.pending_empty_tlas.push(key);
    }
    Some((gpu_va, buffer))
}

/// Returns a valid 0-instance host empty TLAS `(gpu_va, backing_slab)` so null
/// RTAS SRVs never execute `TraceRayInline` against `Location = 0`.
pub(crate) fn empty_tlas_srv() -> Option<(u64, ID3D12Resource)> {
    if !enabled() {
        return None;
    }
    ensure_tlas_slot(EMPTY_TLAS_KEY, 4096)
}

/// Returns `(host_tlas_gpu_va, backing_slab_buffer)` for a guest TLAS SRV
/// (`D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE`).
pub(crate) fn tlas_srv(guest_tlas_va: u64) -> Option<(u64, ID3D12Resource)> {
    if !enabled() {
        return None;
    }
    let res = if guest_tlas_va == 0 {
        ensure_tlas_slot(EMPTY_TLAS_KEY, 4096)?
    } else {
        ensure_tlas_slot(guest_tlas_va, DEFAULT_TLAS_RESERVE)?
    };
    STATS[STAT_TLAS_SRVS].fetch_add(1, Ordering::Relaxed);
    Some(res)
}

fn has_pending_work() -> bool {
    DXR_STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|s| {
            !s.pending_empty_tlas.is_empty()
                || !s.pending_static_upgrades.is_empty()
                || !s.pending_compactions.is_empty()
        })
}

// ============================================================================
// GPU Compute Shaders (compiled once via D3DCompile / cs_5_1)
// ============================================================================

/// Patches guest `D3D12_RAYTRACING_INSTANCE_DESC` (64B stride) BLAS VAs through
/// the open-addressed GPU hash table `(guest_blas_va >> 8 -> host_blas_va)`.
/// Missing BLASes get `AccelerationStructure = 0` and `InstanceMask = 0`.
const PATCH_INSTANCES_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Params : register(b0) {
    uint num_instances;
    uint src_offset;
    uint hash_mask;
    uint _pad;
};
ByteAddressBuffer guest_inst : register(t0);
ByteAddressBuffer blas_table : register(t1);
RWByteAddressBuffer host_inst : register(u0);
RWByteAddressBuffer stats : register(u1);

uint hash_key(uint lo, uint hi) {
    uint x = lo ^ (hi * 0x9E3779B9u);
    x ^= x >> 16u;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13u;
    x *= 0xC2B2AE35u;
    x ^= x >> 16u;
    return x;
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= num_instances) return;
    uint src = src_offset + i * 64u;
    uint dst = i * 64u;
    uint4 r0 = guest_inst.Load4(src + 0u);
    uint4 r1 = guest_inst.Load4(src + 16u);
    uint4 r2 = guest_inst.Load4(src + 32u);
    uint4 r3 = guest_inst.Load4(src + 48u);

    uint blas_lo = r3.z;
    uint blas_hi = r3.w;
    uint host_lo = 0u;
    uint host_hi = 0u;
    bool found = false;

    if ((blas_lo | blas_hi) != 0u) {
        uint key_lo = (blas_lo >> 8u) | (blas_hi << 24u);
        uint key_hi = (blas_hi >> 8u);
        uint slot = hash_key(key_lo, key_hi) & hash_mask;
        [loop]
        for (uint step = 0u; step < 64u; ++step) {
            uint4 e = blas_table.Load4(((slot + step) & hash_mask) * 16u);
            if (e.x == key_lo && e.y == key_hi) {
                host_lo = e.z;
                host_hi = e.w;
                found = true;
                break;
            }
            if ((e.x | e.y) == 0u) {
                break;
            }
        }
        if (found) {
            stats.InterlockedAdd(0u, 1u);
        } else {
            stats.InterlockedAdd(4u, 1u);
        }
    } else {
        stats.InterlockedAdd(8u, 1u);
    }
    stats.InterlockedAdd(12u, 1u);

    if (found) {
        r3.z = host_lo;
        r3.w = host_hi;
    } else {
        r3.x &= 0x00FFFFFFu;
        r3.z = 0u;
        r3.w = 0u;
    }
    host_inst.Store4(dst + 0u, r0);
    host_inst.Store4(dst + 16u, r1);
    host_inst.Store4(dst + 32u, r2);
    host_inst.Store4(dst + 48u, r3);
}
"#;

/// Extracts flat `float3` triangle vertices from unpacked 64-byte TriangleNodes
/// in a rehydrated console BVH4 living in `crate::memory`'s paged arenas, and
/// optionally dumps the raw 64-byte TriangleNodes plus 16-byte header to `u1`
/// for zero-stall async CPU decoding into multi-geometry BLASes.
///
/// Header fields: `dw[3]` = `numElements`, `dw[13]` = `offsetToElements` (bytes).
/// Each 64-byte TriangleNode stores:
/// - bytes `0..48`: 4 vertices (`v0..v3`, 3×f32 each)
/// - `dw[12]`: `GeometryIndexAndFlags` (bit 0 = OPAQUE, bit 7 = tri1 present, bits 31:8 = GeometryIndex)
/// - `dw[13]`: `tri0` `PrimitiveIndex`
/// - `dw[14]`: `tri1` `PrimitiveIndex`
/// - `dw[15]`: per-triangle permutation bytes matching `bvh.rs::TRIANGLE_VERTICES`
///   (`tri0: [0,1,2]`, `tri1: [1,3,2]`, winding `[first, i, j]` where `first = (3-i-j)&3`).
/// Unused or out-of-range slots emit degenerate NaN triangles (`0x7FC00000`) which
/// DXR treats as inactive primitives.
const EXTRACT_BVH4_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1), DescriptorTable(UAV(u0, numDescriptors=unbounded, space=1))"
cbuffer Params : register(b0) {
    uint base_lo;
    uint base_hi;
    uint max_nodes;
    uint size_bytes;
    uint dump_nodes;
    uint copy_limit;
};
ByteAddressBuffer directory : register(t0);
RWByteAddressBuffer out_verts : register(u0);
RWByteAddressBuffer out_nodes : register(u1);
RWByteAddressBuffer mem[] : register(u0, space1);

uint load_u32(uint byte_off) {
    uint lo = base_lo + byte_off;
    uint hi = base_hi + (lo < base_lo ? 1u : 0u);
    uint page = ((lo >> 16u) | (hi << 16u)) & 0x1FFFFFu;
    uint2 entry = directory.Load2(page * 8u);
    uint copy = entry.x & 0x7FFFFFFFu;
    if (copy == 0u || copy > copy_limit) return 0u;
    return mem[copy].Load(entry.y + (lo & 0xFFFFu));
}

void write_nan_tri(uint dst) {
    uint3 nan3 = uint3(0x7FC00000u, 0x7FC00000u, 0x7FC00000u);
    out_verts.Store3(dst + 0u, nan3);
    out_verts.Store3(dst + 12u, nan3);
    out_verts.Store3(dst + 24u, nan3);
}

void write_tri(uint dst, uint3 verts[4], uint map_nibbles, uint id_byte) {
    uint i = id_byte & 3u;
    uint j = (id_byte >> 2u) & 3u;
    if (i >= 3u || j >= 3u || i == j) {
        write_nan_tri(dst);
        return;
    }
    uint f = (3u - i - j) & 3u;
    uint v0 = (map_nibbles >> (4u * f)) & 3u;
    uint v1 = (map_nibbles >> (4u * i)) & 3u;
    uint v2 = (map_nibbles >> (4u * j)) & 3u;
    out_verts.Store3(dst + 0u, verts[v0]);
    out_verts.Store3(dst + 12u, verts[v1]);
    out_verts.Store3(dst + 24u, verts[v2]);
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint node_idx = id.x;
    if (node_idx >= max_nodes) return;
    uint dst0 = node_idx * 72u;
    uint dst1 = dst0 + 36u;

    uint num_elements = load_u32(12u);
    uint offset_to_elements = load_u32(52u);
    bool hdr_ok = (offset_to_elements >= 64u && size_bytes >= 64u && offset_to_elements <= size_bytes - 64u);
    uint max_by_size = hdr_ok ? ((size_bytes - offset_to_elements) >> 6u) : 0u;
    uint eff_elements = min(num_elements, max_by_size);
    if (node_idx == 0u && dump_nodes > 0u) {
        out_nodes.Store4(0u, uint4(eff_elements, offset_to_elements, hdr_ok ? dump_nodes : 0u, size_bytes));
    }
    if (!hdr_ok || node_idx >= eff_elements) {
        write_nan_tri(dst0);
        write_nan_tri(dst1);
        return;
    }

    uint node_off = offset_to_elements + node_idx * 64u;
    uint lo = base_lo + node_off;
    uint hi = base_hi + (lo < base_lo ? 1u : 0u);
    uint page = ((lo >> 16u) | (hi << 16u)) & 0x1FFFFFu;
    uint2 entry = directory.Load2(page * 8u);
    uint copy = entry.x & 0x7FFFFFFFu;
    if (copy == 0u || copy > copy_limit) {
        write_nan_tri(dst0);
        write_nan_tri(dst1);
        if (node_idx < dump_nodes) {
            uint n_dst = 16u + node_idx * 64u;
            out_nodes.Store4(n_dst + 0u, uint4(0u, 0u, 0u, 0u));
            out_nodes.Store4(n_dst + 16u, uint4(0u, 0u, 0u, 0u));
            out_nodes.Store4(n_dst + 32u, uint4(0u, 0u, 0u, 0u));
            out_nodes.Store4(n_dst + 48u, uint4(0u, 0u, 0u, 0xFFFFFFFFu));
        }
        return;
    }
    uint off = entry.y + (lo & 0xFFFFu);
    uint4 w0 = mem[copy].Load4(off + 0u);
    uint4 w1 = mem[copy].Load4(off + 16u);
    uint4 w2 = mem[copy].Load4(off + 32u);
    uint4 w3 = mem[copy].Load4(off + 48u);

    if (node_idx < dump_nodes) {
        uint n_dst = 16u + node_idx * 64u;
        out_nodes.Store4(n_dst + 0u, w0);
        out_nodes.Store4(n_dst + 16u, w1);
        out_nodes.Store4(n_dst + 32u, w2);
        out_nodes.Store4(n_dst + 48u, w3);
    }

    uint3 verts[4];
    verts[0] = uint3(w0.x, w0.y, w0.z);
    verts[1] = uint3(w0.w, w1.x, w1.y);
    verts[2] = uint3(w1.z, w1.w, w2.x);
    verts[3] = uint3(w2.y, w2.z, w2.w);

    uint dw12 = w3.x;
    uint dw15 = w3.w;

    // Triangle 0: TRIANGLE_VERTICES[0] = [0, 1, 2] -> nibbles 0x210
    write_tri(dst0, verts, 0x210u, dw15 & 0xFFu);

    // Triangle 1 (when bit 7 of GeometryIndexAndFlags is set): [1, 3, 2] -> nibbles 0x231
    if ((dw12 & 0x80u) != 0u) {
        write_tri(dst1, verts, 0x231u, (dw15 >> 8u) & 0xFFu);
    } else {
        write_nan_tri(dst1);
    }
}
"#;

struct ComputePass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 root signatures and pipeline states are free-threaded.
unsafe impl Send for ComputePass {}
unsafe impl Sync for ComputePass {}

fn make_compute_pass(host: &Host, hlsl: &str, label: &str) -> Option<ComputePass> {
    let code = match compile(hlsl, s!("main"), s!("cs_5_1")) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, label, "dxr: compute shader compile failed");
            return None;
        }
    };
    // SAFETY: bytecode lives through the calls; root signature is embedded in it.
    unsafe {
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).ok()?;
        let pso: ID3D12PipelineState = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
            .ok()?;
        name(&pso, || label.to_owned());
        Some(ComputePass { root, pso })
    }
}

fn patch_instances_pass(host: &Host) -> Option<&'static ComputePass> {
    static PASS: OnceLock<Option<ComputePass>> = OnceLock::new();
    PASS.get_or_init(|| make_compute_pass(host, PATCH_INSTANCES_HLSL, "victory dxr: patch instances"))
        .as_ref()
}

fn extract_bvh4_pass(host: &Host) -> Option<&'static ComputePass> {
    static PASS: OnceLock<Option<ComputePass>> = OnceLock::new();
    PASS.get_or_init(|| make_compute_pass(host, EXTRACT_BVH4_HLSL, "victory dxr: extract bvh4"))
        .as_ref()
}

// ============================================================================
// Public entry points called from `victory-hle::modules::bvh_builder`
// ============================================================================

/// Queries host DXR prebuild info from guest inputs (used when
/// `VICTORY_NATIVE_RT_SUPPRESS_BUILDER=1`).
pub fn prebuild_info(inputs: &GuestBuildAsInputs) -> Option<GuestPrebuildInfo> {
    if !enabled() {
        return None;
    }
    let host = host()?;
    let device5 = host.device.cast::<ID3D12Device5>().ok()?;
    let (host_inputs, _geoms) = translate_prebuild_inputs(inputs)?;
    let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: valid device5, translated inputs, and output pointer.
    unsafe {
        device5.GetRaytracingAccelerationStructurePrebuildInfo(&host_inputs, &mut info);
    }
    Some(GuestPrebuildInfo {
        result_data_max_size_in_bytes: info.ResultDataMaxSizeInBytes,
        scratch_data_size_in_bytes: info.ScratchDataSizeInBytes,
        update_scratch_data_size_in_bytes: info.UpdateScratchDataSizeInBytes,
    })
}

/// Queues a batch of dynamic BLAS/TLAS builds from `XboxBvhBuilder::BuildRaytracingAccelerationStructureBatched2X`.
pub fn enqueue_build_batched(cmd_list: *const c_void, num_descs: u32, descs: *const GuestBatchedDesc2) {
    if !enabled() || cmd_list.is_null() || descs.is_null() || num_descs == 0 {
        return;
    }
    // SAFETY: caller passes `num_descs` batched descriptors.
    let slice = unsafe { std::slice::from_raw_parts(descs, num_descs as usize) };
    for entry in slice {
        let desc = &entry.desc;
        if desc.dest_va == 0 {
            continue;
        }
        let postbuild = if entry.num_postbuild_info_descs > 0 && !entry.postbuild_info_descs.is_null() {
            // SAFETY: caller passes `num_postbuild_info_descs` descriptors.
            let pb = unsafe {
                std::slice::from_raw_parts(entry.postbuild_info_descs, entry.num_postbuild_info_descs as usize)
            };
            pb.iter()
                .map(|p| PostbuildSpec {
                    dest_buffer: p.dest_buffer,
                    info_type: p.info_type,
                })
                .collect()
        } else {
            Vec::new()
        };
        let op = match desc.inputs.kind {
            0 => DxrOp::BuildTlas {
                dest_va: desc.dest_va,
                source_va: desc.source_va,
                flags: desc.inputs.flags,
                num_instances: desc.inputs.num_descs,
                instance_descs_va: desc.inputs.ptr_or_va,
                postbuild,
            },
            1 => {
                let geometries = copy_guest_geometries(&desc.inputs);
                DxrOp::BuildBlas {
                    dest_va: desc.dest_va,
                    source_va: desc.source_va,
                    flags: desc.inputs.flags,
                    geometries,
                    postbuild,
                }
            }
            _ => continue,
        };
        crate::list::queue_dxr_op(cmd_list as This, op);
    }
}

/// Queues an AS copy from `XboxBvhBuilder::CopyRaytracingAccelerationStructure`.
pub fn enqueue_copy(cmd_list: *const c_void, dest_va: u64, source_va: u64, mode: u32) {
    if !enabled() || cmd_list.is_null() || dest_va == 0 || source_va == 0 {
        return;
    }
    crate::list::queue_dxr_op(
        cmd_list as This,
        DxrOp::CopyAs {
            dest_va,
            source_va,
            mode,
        },
    );
}

/// Queues static BLAS extraction + DXR build for a batch of rehydrated console BVH4s from
/// `XboxBvhBuilder::RehydrateRaytracingAccelerationStructureBatched2X`.
pub fn enqueue_rehydrate_batched(
    cmd_list: *const c_void,
    num_descs: u32,
    descs: *const GuestBuildAsDesc,
    sizes: *const u64,
) {
    if !enabled() || cmd_list.is_null() || descs.is_null() || num_descs == 0 {
        return;
    }
    // SAFETY: caller passes `num_descs` descriptors (and optional `sizes` array).
    let slice = unsafe { std::slice::from_raw_parts(descs, num_descs as usize) };
    let size_slice = if sizes.is_null() {
        None
    } else {
        // SAFETY: parallel array of `num_descs` byte lengths.
        Some(unsafe { std::slice::from_raw_parts(sizes, num_descs as usize) })
    };
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    for (i, desc) in slice.iter().enumerate() {
        if desc.dest_va == 0 {
            continue;
        }
        let raw_arg = size_slice.map_or(0, |s| s[i]);
        let probed_dest = probe_packed_bvh_size(desc.dest_va);
        let probed_src = (desc.source_va != 0)
            .then(|| probe_packed_bvh_size(desc.source_va))
            .flatten();
        // `RehydrateRaytracingAccelerationStructureBatched2X`'s 5th parameter may be
        // either byte sizes or GPU VAs/flags; never trust values outside `(128..=64 MiB)`.
        let arg_size = (128..=MAX_PACKED_BVH_BYTES).contains(&raw_arg).then_some(raw_arg);
        let size_bytes = probed_dest.or(probed_src).or(arg_size).unwrap_or(0);
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                dest = format_args!("{:#x}", desc.dest_va),
                src = format_args!("{:#x}", desc.source_va),
                raw_arg = format_args!("{:#x}", raw_arg),
                probed_dest = ?probed_dest,
                probed_src = ?probed_src,
                size_bytes,
                "dxr: rehydrate static BLAS sample"
            );
        }
        crate::list::queue_dxr_op(
            cmd_list as This,
            DxrOp::ExtractStaticBlas {
                dest_va: desc.dest_va,
                size_bytes,
            },
        );
    }
}

/// Infers unpacked byte size from a console packed BVH4 header (`dw[0] & 0xFFFF == 6`,
/// marks `0x8100`, `dw[1]` = node count minus one; matches `memory.rs::packed_len`).
fn probe_packed_bvh_size(va: u64) -> Option<u64> {
    if va == 0 || (va & 0xFF) != 0 {
        return None;
    }
    let mut header = [0u8; 8];
    crate::host::copy_mapped(header.as_mut_ptr(), va as *const u8, 8, false);
    let words = u64::from_le_bytes(header);
    let version = words as u32;
    let marks = version >> 16;
    let nodes = words >> 32;
    let len = nodes.saturating_add(1).saturating_mul(64);
    ((version & 0xFFFF) == 6 && (marks & 0x8100) == 0x8100 && nodes > 0 && len <= MAX_PACKED_BVH_BYTES).then_some(len)
}

/// Probes either a packed BVH4 header or an already-unpacked BVH4 header (`dw[3]` =
/// `num_elements`, `dw[13]` = `offset_to_elements`) at `va` to recover `(size_bytes, max_nodes)`.
fn probe_bvh_bounds(va: u64, hint_bytes: u64) -> Option<(u64, u64)> {
    let mut bytes = if (128..=MAX_PACKED_BVH_BYTES).contains(&hint_bytes) {
        hint_bytes
    } else {
        probe_packed_bvh_size(va).unwrap_or(0)
    };
    // Also read the 64-byte header: if the BVH is already unpacked in guest memory,
    // `dw[3]` is exact `numElements` and `dw[13]` is `offsetToElements` in bytes.
    let mut hdr = [0u8; 64];
    crate::host::copy_mapped(hdr.as_mut_ptr(), va as *const u8, 64, false);
    let dw = |idx: usize| u32::from_le_bytes(hdr[idx * 4..idx * 4 + 4].try_into().unwrap());
    let num_elements = u64::from(dw(3));
    let off_elements = u64::from(dw(13));
    if bytes == 0 && off_elements >= 64 && num_elements > 0 && num_elements <= MAX_STATIC_NODES {
        let unpacked = off_elements.saturating_add(num_elements.saturating_mul(64));
        if unpacked <= MAX_PACKED_BVH_BYTES {
            bytes = unpacked;
        }
    }
    if bytes < 128 || bytes > MAX_PACKED_BVH_BYTES {
        return None;
    }
    // Use exact `num_elements` when header looks unpacked; otherwise conservative upper bound
    // (at most `size_bytes / 64`, capped at `MAX_STATIC_NODES`).
    let max_nodes = if off_elements >= 64
        && off_elements < bytes
        && num_elements > 0
        && off_elements.saturating_add(num_elements.saturating_mul(64)) <= bytes
    {
        num_elements.min(MAX_STATIC_NODES)
    } else {
        (bytes / 64).clamp(1, MAX_STATIC_NODES)
    };
    Some((bytes, max_nodes))
}

fn copy_guest_geometries(inputs: &GuestBuildAsInputs) -> Vec<GeometrySpec> {
    let count = inputs.num_descs as usize;
    if count == 0 || inputs.ptr_or_va == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let g = if inputs.descs_layout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS.0 as u32 {
            // SAFETY: guest passes an array of `count` geometry descriptor pointers.
            let ptr = unsafe { (inputs.ptr_or_va as *const *const GuestGeometryDesc).add(i).read_unaligned() };
            if ptr.is_null() {
                continue;
            }
            // SAFETY: non-null guest geometry descriptor.
            unsafe { ptr.read_unaligned() }
        } else {
            // SAFETY: guest passes a contiguous array of `count` geometry descriptors.
            unsafe { (inputs.ptr_or_va as *const GuestGeometryDesc).add(i).read_unaligned() }
        };
        let kind = match g.kind {
            0 => {
                // SAFETY: `kind == 0` selects `triangles`.
                let t = unsafe { g.data.triangles };
                GeometryKind::Triangles {
                    transform3x4: t.transform3x4,
                    index_format: t.index_format,
                    vertex_format: t.vertex_format,
                    index_count: t.index_count,
                    vertex_count: t.vertex_count,
                    index_buffer: t.index_buffer,
                    vertex_buffer_va: t.vertex_buffer_va,
                    vertex_buffer_stride: t.vertex_buffer_stride,
                }
            }
            1 => {
                // SAFETY: `kind == 1` selects `aabbs`.
                let a = unsafe { g.data.aabbs };
                GeometryKind::Aabbs {
                    aabb_count: a.aabb_count,
                    aabbs_va: a.aabbs_va,
                    aabbs_stride: a.aabbs_stride,
                }
            }
            _ => continue,
        };
        out.push(GeometrySpec { flags: g.flags, kind });
    }
    out
}

fn translate_prebuild_inputs(
    inputs: &GuestBuildAsInputs,
) -> Option<(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS, Vec<D3D12_RAYTRACING_GEOMETRY_DESC>)> {
    let flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(inputs.flags as i32);
    match inputs.kind {
        0 => Some((
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: flags,
                NumDescs: inputs.num_descs,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: 0,
                },
            },
            Vec::new(),
        )),
        1 => {
            let specs = copy_guest_geometries(inputs);
            let mut geoms = Vec::with_capacity(specs.len());
            for spec in &specs {
                geoms.push(host_geom_desc_for_prebuild(spec));
            }
            let ptr = if geoms.is_empty() {
                std::ptr::null()
            } else {
                geoms.as_ptr()
            };
            Some((
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: flags,
                    NumDescs: geoms.len() as u32,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: ptr,
                    },
                },
                geoms,
            ))
        }
        _ => None,
    }
}

fn host_geom_desc_for_prebuild(spec: &GeometrySpec) -> D3D12_RAYTRACING_GEOMETRY_DESC {
    match spec.kind {
        GeometryKind::Triangles {
            transform3x4,
            index_format,
            vertex_format,
            index_count,
            vertex_count,
            index_buffer,
            vertex_buffer_va,
            vertex_buffer_stride,
        } => D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: dynamic_geom_flags(spec.flags),
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                    Transform3x4: u64::from(transform3x4 != 0) * 16,
                    IndexFormat: DXGI_FORMAT(crate::resource::pc_format(index_format) as i32),
                    VertexFormat: DXGI_FORMAT(crate::resource::pc_format(vertex_format) as i32),
                    IndexCount: index_count,
                    VertexCount: vertex_count,
                    IndexBuffer: u64::from(index_buffer != 0) * 16,
                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        StartAddress: u64::from(vertex_buffer_va != 0) * 16,
                        StrideInBytes: vertex_buffer_stride,
                    },
                },
            },
        },
        GeometryKind::Aabbs {
            aabb_count,
            aabbs_va,
            aabbs_stride,
        } => D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS,
            Flags: dynamic_geom_flags(spec.flags),
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                AABBs: D3D12_RAYTRACING_GEOMETRY_AABBS_DESC {
                    AABBCount: aabb_count,
                    AABBs: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        StartAddress: u64::from(aabbs_va != 0) * 16,
                        StrideInBytes: aabbs_stride,
                    },
                },
            },
        },
    }
}

// ============================================================================
// List-recording preparation (schedules shadow refreshes before `card_before`)
// ============================================================================

/// Schedules shadow buffer refreshes/write-backs on `list` for guest buffers
/// read or written by `op`, so `Host::submit` uploads them via `card_before`
/// before DXR replay runs.
pub(crate) fn prepare_list_op(list: &HostList, op: &DxrOp) {
    match op {
        DxrOp::BuildBlas { geometries, postbuild, .. } => {
            for g in geometries {
                match g.kind {
                    GeometryKind::Triangles {
                        transform3x4,
                        index_format,
                        index_count,
                        vertex_count,
                        index_buffer,
                        vertex_buffer_va,
                        vertex_buffer_stride,
                        ..
                    } => {
                        if vertex_buffer_va != 0 && vertex_count > 0 {
                            let stride = vertex_buffer_stride.max(12);
                            let len = u64::from(vertex_count).saturating_mul(stride);
                            if let Some(s) = crate::shadow::find(vertex_buffer_va, len) {
                                s.refresh(list, vertex_buffer_va, len);
                            }
                        }
                        if index_buffer != 0 && index_count > 0 {
                            let elem = if crate::resource::pc_format(index_format) == DXGI_FORMAT_R16_UINT.0 as u32 {
                                2u64
                            } else {
                                4u64
                            };
                            let len = u64::from(index_count).saturating_mul(elem);
                            if let Some(s) = crate::shadow::find(index_buffer, len) {
                                s.refresh(list, index_buffer, len);
                            }
                        }
                        if transform3x4 != 0
                            && let Some(s) = crate::shadow::find(transform3x4, 48)
                        {
                            s.refresh(list, transform3x4, 48);
                        }
                    }
                    GeometryKind::Aabbs {
                        aabb_count,
                        aabbs_va,
                        aabbs_stride,
                    } => {
                        if aabbs_va != 0 && aabb_count > 0 {
                            let stride = aabbs_stride.max(24);
                            let len = aabb_count.saturating_mul(stride);
                            if let Some(s) = crate::shadow::find(aabbs_va, len) {
                                s.refresh(list, aabbs_va, len);
                            }
                        }
                    }
                }
            }
            if suppress_builder() {
                for pb in postbuild {
                    if pb.dest_buffer != 0
                        && let Some(s) = crate::shadow::find(pb.dest_buffer, 16)
                    {
                        s.refresh(list, pb.dest_buffer, 16);
                        s.gpu_writes(list, pb.dest_buffer, 16);
                    }
                }
            }
        }
        DxrOp::BuildTlas {
            num_instances,
            instance_descs_va,
            postbuild,
            ..
        } => {
            if *num_instances > 0 && *instance_descs_va != 0 {
                let len = u64::from(*num_instances).saturating_mul(64);
                if let Some(s) = crate::shadow::find(*instance_descs_va, len) {
                    s.refresh(list, *instance_descs_va, len);
                }
            }
            if suppress_builder() {
                for pb in postbuild {
                    if pb.dest_buffer != 0
                        && let Some(s) = crate::shadow::find(pb.dest_buffer, 16)
                    {
                        s.refresh(list, pb.dest_buffer, 16);
                        s.gpu_writes(list, pb.dest_buffer, 16);
                    }
                }
            }
        }
        DxrOp::CopyAs { .. } | DxrOp::ExtractStaticBlas { .. } => {}
    }
}

// ============================================================================
// Submission-time replay on `victory-submit`
// ============================================================================

/// Replays queued [`DxrOp`]s for `runs` onto pooled host `ID3D12GraphicsCommandList4`
/// lists and splices them into `lists` (updating `firsts`) in exact console submission
/// order. Called from `Host::submit` after `card_before` and before `ExecuteCommandLists`.
pub(crate) fn record_submission(
    host: &Host,
    runs: &[&HostList],
    lists: &mut Vec<Option<ID3D12CommandList>>,
    firsts: &mut Vec<usize>,
) -> Option<DxrSubmission> {
    if !enabled() {
        return None;
    }
    let has_ops = runs.iter().any(|l| l.has_dxr_ops());
    if !has_ops && !has_pending_work() {
        return None;
    }
    let Ok(device5) = host.device.cast::<ID3D12Device5>() else {
        return None;
    };

    let mut state_guard = lock_state();
    let state = state_guard.as_mut()?;
    let mut sub = state.take_submission();
    let mut new_lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(lists.len() + runs.len() + 2);
    let mut new_firsts: Vec<usize> = Vec::with_capacity(firsts.len());

    // Clone queued ops once per run so we don't duplicate allocations between
    // readback pre-sizing and command recording.
    let run_ops: Vec<Vec<DxrOp>> = if has_ops {
        runs.iter()
            .map(|r| if r.has_dxr_ops() { r.dxr_ops() } else { Vec::new() })
            .collect()
    } else {
        Vec::new()
    };

    // Pre-size this submission's node readback buffer upfront (capped at 8 MiB) so
    // `record_extract_static_blas` never reallocates the readback buffer mid-submission.
    let mut need_readback = 0u64;
    for ops in &run_ops {
        for op in ops {
            if let DxrOp::ExtractStaticBlas { dest_va, size_bytes } = *op {
                let key = dest_va & !0xFF;
                if key != 0
                    && let Some((_, max_nodes)) = probe_bvh_bounds(key, size_bytes)
                {
                    let dump = (16u64 + max_nodes.saturating_mul(64)).next_multiple_of(64);
                    if need_readback.saturating_add(dump) <= MAX_READBACK_BYTES_PER_SUB {
                        need_readback = need_readback.saturating_add(dump);
                    }
                }
            }
        }
    }
    sub.prepare_node_readback(host, need_readback);

    // Always initialize every newly pre-allocated TLAS slot to a valid 0-instance
    // empty TLAS at `i == 0` before any list in this submission executes (even if
    // a later list in the same submission also rebuilds it).
    let pending_init = std::mem::take(&mut state.pending_empty_tlas);
    let mut built_tlas = false;

    for (i, run) in runs.iter().enumerate() {
        let old_start = firsts[i];
        let old_end = firsts.get(i + 1).copied().unwrap_or(lists.len());
        new_firsts.push(new_lists.len());

        if i == 0
            && !pending_init.is_empty()
            && let Some(init_list) = record_empty_tlas_init(host, &device5, state, &mut sub, &pending_init)
        {
            new_lists.push(Some(init_list));
        }

        if i == 0
            && !state.pending_static_upgrades.is_empty()
            && let Some(upg_list) = record_static_upgrades(host, &device5, state, &mut sub)
        {
            new_lists.push(Some(upg_list));
        }

        if i == 0
            && !state.pending_compactions.is_empty()
            && let Some(cmp_list) = record_static_compactions(host, state, &mut sub)
        {
            new_lists.push(Some(cmp_list));
        }

        let ops = run_ops.get(i).map(Vec::as_slice).unwrap_or(&[]);
        if ops.is_empty() || old_start >= old_end {
            new_lists.extend(lists[old_start..old_end].iter().cloned());
            continue;
        }

        // Run all DXR ops after `run.list` so skinning, buffer copies, and
        // `XboxBvhBuilder.dll` rehydration dispatches in `run.list` have
        // completed before host DXR reads vertex/index/instance/BVH4 buffers.
        new_lists.extend(lists[old_start..old_end].iter().cloned());
        if let Some(cmd) = record_ops(host, &device5, state, &mut sub, run, ops, &mut built_tlas) {
            new_lists.push(Some(cmd));
        }
    }

    if sub.lists.is_empty() {
        sub.readbacks.clear();
        sub.readback_used = 0;
        sub.readback_cap = 0;
        sub.compact_queries.clear();
        sub.kept.clear();
        sub.kept_shadows.clear();
        sub.sub_seq = 0;
        sub.tlas_epoch = 0;
        if state.free_submissions.len() < MAX_FREE_SUBS {
            state.free_submissions.push(sub);
        }
        return None;
    }

    state.sub_seq = state.sub_seq.saturating_add(1);
    sub.sub_seq = state.sub_seq;
    sub.tlas_epoch = if built_tlas { state.tlas_build_epoch } else { 0 };

    *lists = new_lists;
    *firsts = new_firsts;
    Some(sub)
}

struct DecodeJob {
    readbacks: Vec<StaticReadbackRecord>,
    bytes: Vec<u8>,
}

/// Returns the bounded single-worker channel that decodes static BVH4 readbacks off
/// `victory-relay` / `victory-submit` without blocking GPU completion.
fn decode_worker() -> &'static std::sync::mpsc::SyncSender<DecodeJob> {
    static TX: OnceLock<std::sync::mpsc::SyncSender<DecodeJob>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<DecodeJob>(MAX_DECODE_QUEUE);
        let _ = std::thread::Builder::new()
            .name("victory-dxr-decode".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let live: Vec<StaticReadbackRecord> = {
                        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
                        job.readbacks
                            .into_iter()
                            .filter(|rec| {
                                map.get(&rec.guest_key)
                                    .is_some_and(|e| e.generation == rec.generation)
                            })
                            .collect()
                    };
                    let mut decoded = Vec::new();
                    for rec in live {
                        let start = rec.offset as usize;
                        let end = start.saturating_add(rec.len as usize);
                        if end > job.bytes.len() {
                            continue;
                        }
                        if let Some(upg) =
                            decode_static_bvh4_nodes(rec.guest_key, rec.generation, &job.bytes[start..end])
                        {
                            decoded.push(upg);
                        }
                    }
                    if !decoded.is_empty() {
                        let mut guard = DXR_STATE.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(state) = guard.as_mut() {
                            for upg in decoded {
                                if state.pending_static_upgrades.len() < MAX_PENDING_UPGRADES {
                                    state.pending_static_upgrades.push_back(upg);
                                }
                            }
                        }
                    }
                }
            });
        tx
    })
}

/// Returns pooled DXR command allocators and scratch resources once the
/// submission's GPU fence has completed, reclaims eligible retired AS slab slices,
/// and offloads any completed static BVH4 node readbacks to the background decoder.
pub(crate) fn finish_submission(mut sub: DxrSubmission) {
    sub.kept.clear();
    sub.kept_shadows.clear();
    let readbacks = std::mem::take(&mut sub.readbacks);
    let readback_used = sub.readback_used;
    sub.readback_used = 0;
    sub.readback_cap = 0;
    let compact_queries = std::mem::take(&mut sub.compact_queries);
    let completed_sub_seq = sub.sub_seq;
    let completed_tlas_epoch = sub.tlas_epoch;
    sub.sub_seq = 0;
    sub.tlas_epoch = 0;

    let mut new_compactions: Vec<(u64, u64, u64)> = Vec::new();
    if !compact_queries.is_empty()
        && let Some((rb_buf, rb_size)) = sub.compact_info_rb.as_ref()
    {
        let max_slots = compact_queries
            .iter()
            .map(|q| u64::from(q.slot) + 1)
            .max()
            .unwrap_or(0);
        let map_end = (max_slots.saturating_mul(8).min(*rb_size)) as usize;
        if map_end >= 8 {
            let mut ptr = std::ptr::null_mut();
            let read_range = D3D12_RANGE { Begin: 0, End: map_end };
            // SAFETY: submission GPU fence has completed; readback buffer is idle.
            let mapped = unsafe { rb_buf.Map(0, Some(&read_range), Some(&mut ptr)) };
            if mapped.is_ok() && !ptr.is_null() {
                // SAFETY: `ptr` is valid for `map_end` bytes until `Unmap`.
                let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), map_end) };
                let sizes: Vec<u64> = compact_queries
                    .iter()
                    .map(|q| {
                        let off = (q.slot as usize).saturating_mul(8);
                        if off + 8 <= bytes.len() {
                            u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
                        } else {
                            0
                        }
                    })
                    .collect();
                let written_none = D3D12_RANGE { Begin: 0, End: 0 };
                // SAFETY: matches successful `Map(0, ...)`.
                unsafe { rb_buf.Unmap(0, Some(&written_none)) };

                const ALLOW_COMPACTION: u32 =
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0 as u32;
                let map = as_map().read().unwrap_or_else(|e| e.into_inner());
                for (q, compacted_size) in compact_queries.into_iter().zip(sizes) {
                    if !should_compact(q.uncompacted_size, compacted_size) {
                        continue;
                    }
                    let live = map.get(&q.guest_key).is_some_and(|e| {
                        e.kind == AsKind::Blas
                            && e.built
                            && e.generation == q.generation
                            && (e.flags & ALLOW_COMPACTION) != 0
                            && should_compact(e.slice.size, compacted_size)
                    });
                    if live {
                        new_compactions.push((q.guest_key, q.generation, compacted_size));
                    }
                }
            }
        }
    }

    if !readbacks.is_empty()
        && let Some((rb_buf, rb_size)) = sub.node_readback.as_ref()
    {
        let map_end = (readback_used.min(*rb_size)) as usize;
        if map_end >= 16 {
            let mut ptr = std::ptr::null_mut();
            let read_range = D3D12_RANGE { Begin: 0, End: map_end };
            // SAFETY: submission GPU fence has completed; readback buffer is idle.
            let mapped = unsafe { rb_buf.Map(0, Some(&read_range), Some(&mut ptr)) };
            if mapped.is_ok() && !ptr.is_null() {
                // SAFETY: `ptr` is valid for `map_end` bytes until `Unmap`.
                let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), map_end) }.to_vec();
                let written_none = D3D12_RANGE { Begin: 0, End: 0 };
                // SAFETY: matches successful `Map(0, ...)`.
                unsafe { rb_buf.Unmap(0, Some(&written_none)) };
                // Non-blocking enqueue so `victory-relay` never stalls if decode queue is saturated.
                let _ = decode_worker().try_send(DecodeJob { readbacks, bytes });
            }
        }
    }

    // Release unusually large temporary buffers once the GPU fence is done so
    // pooled submissions stay small in steady state.
    if sub.scratch.as_ref().is_some_and(|(_, sz)| *sz > MAX_POOLED_SCRATCH_BYTES) {
        sub.scratch = None;
    }
    if sub
        .extract_verts
        .as_ref()
        .is_some_and(|(_, sz)| *sz > MAX_POOLED_EXTRACT_VERTS_BYTES)
    {
        sub.extract_verts = None;
    }
    if sub
        .host_instances
        .as_ref()
        .is_some_and(|(_, sz)| *sz > MAX_POOLED_HOST_INSTANCES_BYTES)
    {
        sub.host_instances = None;
    }
    if sub.node_staging.as_ref().is_some_and(|(_, sz)| *sz > MAX_POOLED_AUX_BYTES) {
        sub.node_staging = None;
    }
    if sub.node_readback.as_ref().is_some_and(|(_, sz)| *sz > MAX_POOLED_AUX_BYTES) {
        sub.node_readback = None;
    }
    if sub.upgrade_upload.as_ref().is_some_and(|(_, _, sz)| *sz > MAX_POOLED_AUX_BYTES) {
        sub.upgrade_upload = None;
    }
    if sub.upgrade_verts.as_ref().is_some_and(|(_, sz)| *sz > MAX_POOLED_AUX_BYTES) {
        sub.upgrade_verts = None;
    }

    let mut guard = DXR_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(state) = guard.as_mut() else {
        return;
    };
    state.completed_sub_seq = state.completed_sub_seq.max(completed_sub_seq);
    state.completed_tlas_epoch = state.completed_tlas_epoch.max(completed_tlas_epoch);
    state
        .slabs
        .drain_retired(state.completed_tlas_epoch, state.completed_sub_seq);
    for item in new_compactions {
        if state.pending_compactions.len() < MAX_PENDING_COMPACTIONS {
            state.pending_compactions.push_back(item);
        }
    }
    for prelude in sub.lists.drain(..) {
        if state.free_lists.len() < MAX_FREE_LISTS {
            state.free_lists.push(prelude);
        }
    }
    if state.free_submissions.len() < MAX_FREE_SUBS {
        state.free_submissions.push(sub);
    }
}

/// Drains up to `MAX_UPGRADES_PER_SUB` decoded multi-geometry static BLAS upgrades
/// and builds them into host AS slabs, updating `AS_MAP` and the GPU BLAS hash table.
fn record_static_upgrades(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
) -> Option<ID3D12CommandList> {
    let mut batch: Vec<(PendingStaticUpgrade, u64)> = Vec::new();
    let mut batch_bytes = 0u64;
    {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        while batch.len() < MAX_UPGRADES_PER_SUB {
            let Some(front) = state.pending_static_upgrades.front() else {
                break;
            };
            let vert_bytes = (front.vertices.len() as u64).saturating_mul(12);
            let aligned_off = batch_bytes.next_multiple_of(16);
            let next_bytes = aligned_off.saturating_add(vert_bytes);
            if !batch.is_empty() && next_bytes > MAX_UPGRADE_BATCH_BYTES {
                break;
            }
            let Some(upg) = state.pending_static_upgrades.pop_front() else {
                break;
            };
            let still_current = map
                .get(&upg.guest_key)
                .is_some_and(|e| e.generation == upg.generation);
            if !still_current || upg.geometries.is_empty() || upg.vertices.is_empty() {
                continue;
            }
            batch.push((upg, aligned_off));
            batch_bytes = next_bytes;
        }
    }
    if batch.is_empty() || batch_bytes == 0 {
        return None;
    }

    let (upload_buf, upload_ptr) = sub.ensure_upgrade_upload(host, batch_bytes)?;
    let (verts_buf, verts_gpu) = sub.ensure_upgrade_verts(host, batch_bytes)?;

    let build_flags = static_blas_build_flags();
    struct PreparedUpgrade {
        guest_key: u64,
        host_geoms: Vec<D3D12_RAYTRACING_GEOMETRY_DESC>,
        result_bytes: u64,
    }
    let mut prepared = Vec::with_capacity(batch.len());
    let mut max_scratch = 256u64;
    for (upg, batch_off) in &batch {
        let byte_len = upg.vertices.len() * 12;
        // SAFETY: `upload_ptr` is mapped for at least `batch_bytes >= batch_off + byte_len`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                upg.vertices.as_ptr().cast::<u8>(),
                upload_ptr.add(*batch_off as usize),
                byte_len,
            );
        }
        let base_gpu = verts_gpu + *batch_off;
        let mut host_geoms = Vec::with_capacity(upg.geometries.len());
        for g in &upg.geometries {
            let flags = static_geom_flags(g.opaque);
            let (start_addr, stride) = if g.vert_count > 0 {
                (base_gpu + u64::from(g.vert_offset) * 12, 12u64)
            } else {
                (0, 0)
            };
            host_geoms.push(D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: flags,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_UNKNOWN,
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 0,
                        VertexCount: g.vert_count,
                        IndexBuffer: 0,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: start_addr,
                            StrideInBytes: stride,
                        },
                    },
                },
            });
        }
        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: build_flags,
            NumDescs: host_geoms.len() as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: host_geoms.as_ptr(),
            },
        };
        let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: valid device5 and `host_geoms` alive through call.
        unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre) };
        if pre.ResultDataMaxSizeInBytes == 0 {
            continue;
        }
        max_scratch = max_scratch.max(pre.ScratchDataSizeInBytes.max(256));
        prepared.push(PreparedUpgrade {
            guest_key: upg.guest_key,
            host_geoms,
            result_bytes: pre.ResultDataMaxSizeInBytes,
        });
    }
    if prepared.is_empty() {
        return None;
    }

    // Allocate fresh slab slices and swap into `AS_MAP` in one batched write lock,
    // deferring reclamation of replaced Frame-0 slices until 2 TLAS epochs complete.
    let mut built_items: Vec<(usize, u64, u64, u64, u64)> = Vec::with_capacity(prepared.len());
    {
        let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
        for (idx, item) in prepared.iter().enumerate() {
            // Allocate a fresh slab slice so any already-built TLAS tracing the
            // previous frame-0 1-geometry BLAS never sees an in-place overwrite
            // before the next `BuildTlas` switches instances to the upgraded VA.
            let Some((buffer, slice, gpu_va)) = state.slabs.alloc(host, item.result_bytes) else {
                continue;
            };
            let generation = state.bump_gen();
            let slice_size = slice.size;
            let entry = HostAs {
                buffer,
                slice,
                gpu_va,
                kind: AsKind::Blas,
                flags: build_flags.0 as u32,
                num_instances: 0,
                built: true,
                generation,
                topo_hash: 0,
                update_scratch_bytes: 0,
            };
            if let Some(old) = map.insert(item.guest_key, entry) {
                state.retire_as(&old);
            }
            built_items.push((idx, item.guest_key, gpu_va, generation, slice_size));
        }
    }
    if built_items.is_empty() {
        return None;
    }

    let (_, scratch_gpu) = sub.ensure_scratch(host, max_scratch)?;
    let compact_info = if compact_static_blas() && sub.compact_queries.len() < COMPACT_INFO_SLOTS {
        sub.ensure_compact_info(host)
    } else {
        None
    };
    let (allocator, list) = state.take_list(host)?;
    name(&list, || "victory dxr static upgrade".into());
    let list4: ID3D12GraphicsCommandList4 = list.cast().ok()?;

    let mut new_queries: Vec<CompactQueryRecord> = Vec::new();
    let first_slot = sub.compact_queries.len();

    // SAFETY: idle allocator/list; upload, vertex, scratch, compact info, and slab buffers live through submission.
    let recorded: windows::core::Result<()> = unsafe {
        (|| {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            list.ResourceBarrier(&[uav_barrier(None)]);
            list.CopyBufferRegion(&verts_buf, 0, &upload_buf, 0, batch_bytes);
            list.ResourceBarrier(&[transition(
                &verts_buf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
            if let Some((uav_buf, _, _)) = &compact_info {
                list.ResourceBarrier(&[transition(
                    uav_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            for &(idx, guest_key, gpu_va, generation, slice_size) in &built_items {
                let item = &prepared[idx];
                let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: build_flags,
                    NumDescs: item.host_geoms.len() as u32,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: item.host_geoms.as_ptr(),
                    },
                };
                let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: gpu_va,
                    Inputs: inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: scratch_gpu,
                };
                let postbuild = if let Some((_, uav_gpu, _)) = &compact_info
                    && first_slot + new_queries.len() < COMPACT_INFO_SLOTS
                {
                    let slot = (first_slot + new_queries.len()) as u32;
                    new_queries.push(CompactQueryRecord {
                        guest_key,
                        generation,
                        uncompacted_size: slice_size,
                        slot,
                    });
                    Some([D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                        DestBuffer: uav_gpu + u64::from(slot) * 8,
                        InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
                    }])
                } else {
                    None
                };
                list4.BuildRaytracingAccelerationStructure(
                    &desc,
                    postbuild.as_ref().map(|p| p.as_slice()),
                );
                list.ResourceBarrier(&[uav_barrier(None)]);
            }
            if let Some((uav_buf, _, rb_buf)) = &compact_info {
                if !new_queries.is_empty() {
                    list.ResourceBarrier(&[transition(
                        uav_buf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    let off = (first_slot as u64) * 8;
                    let len = (new_queries.len() as u64) * 8;
                    list.CopyBufferRegion(rb_buf, off, uav_buf, off, len);
                    list.ResourceBarrier(&[transition(
                        uav_buf,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                } else {
                    list.ResourceBarrier(&[transition(
                        uav_buf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
            }
            list.ResourceBarrier(&[transition(
                &verts_buf,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
            list.Close()
        })()
    };
    if let Err(e) = recorded {
        tracing::warn!(error = %e, "dxr: static BLAS upgrade list failed");
        state.free_lists.push((allocator, list));
        return None;
    }
    sub.compact_queries.extend(new_queries);
    for &(_, guest_key, host_va, _, _) in &built_items {
        state.insert_blas_hash(guest_key, host_va);
    }
    let upg_geoms: u64 = built_items
        .iter()
        .map(|&(idx, _, _, _, _)| prepared[idx].host_geoms.len() as u64)
        .sum();
    let upg_non_opaque: u64 = built_items
        .iter()
        .flat_map(|&(idx, _, _, _, _)| prepared[idx].host_geoms.iter())
        .filter(|g| (g.Flags.0 & D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE.0) == 0)
        .count() as u64;
    STATS[STAT_UPG_GEOMS].fetch_add(upg_geoms, Ordering::Relaxed);
    STATS[STAT_UPG_NON_OPAQUE].fetch_add(upg_non_opaque, Ordering::Relaxed);
    STATS[STAT_UPGRADED_STATIC].fetch_add(built_items.len() as u64, Ordering::Relaxed);
    let out: ID3D12CommandList = list.cast().ok()?;
    sub.lists.push((allocator, list));
    Some(out)
}

/// Drains up to `MAX_COMPACTIONS_PER_SUB` queued static BLAS compactions, copies them
/// into smaller slab slices via `COPY_MODE_COMPACT`, updates `AS_MAP` and the GPU hash
/// table, and retires the old uncompacted slices under the `tlas_build_epoch + 2` invariant.
fn record_static_compactions(
    host: &Host,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
) -> Option<ID3D12CommandList> {
    const ALLOW_COMPACTION: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0 as u32;
    struct PreparedCompaction {
        guest_key: u64,
        generation: u64,
        src_va: u64,
        dst_buf: ID3D12Resource,
        dst_slice: SlabSlice,
        dst_va: u64,
        flags: u32,
    }
    let mut batch: Vec<PreparedCompaction> = Vec::new();
    {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        while batch.len() < MAX_COMPACTIONS_PER_SUB {
            let Some((guest_key, generation, compacted_size)) = state.pending_compactions.pop_front() else {
                break;
            };
            if batch.iter().any(|b| b.guest_key == guest_key) {
                continue;
            }
            let Some(entry) = map.get(&guest_key) else {
                continue;
            };
            if entry.kind != AsKind::Blas
                || !entry.built
                || entry.generation != generation
                || (entry.flags & ALLOW_COMPACTION) == 0
                || !should_compact(entry.slice.size, compacted_size)
            {
                continue;
            }
            let Some((dst_buf, dst_slice, dst_va)) = state.slabs.alloc(host, compacted_size) else {
                break;
            };
            if dst_slice.size >= entry.slice.size {
                SlabAllocator::free_in_slabs(&mut state.slabs.slabs, dst_slice);
                continue;
            }
            batch.push(PreparedCompaction {
                guest_key,
                generation,
                src_va: entry.gpu_va,
                dst_buf,
                dst_slice,
                dst_va,
                flags: entry.flags & !ALLOW_COMPACTION,
            });
        }
    }
    if batch.is_empty() {
        return None;
    }

    let Some((allocator, list)) = state.take_list(host) else {
        for item in batch {
            SlabAllocator::free_in_slabs(&mut state.slabs.slabs, item.dst_slice);
        }
        return None;
    };
    name(&list, || "victory dxr static compact".into());
    let Ok(list4) = list.cast::<ID3D12GraphicsCommandList4>() else {
        for item in batch {
            SlabAllocator::free_in_slabs(&mut state.slabs.slabs, item.dst_slice);
        }
        state.free_lists.push((allocator, list));
        return None;
    };

    // SAFETY: idle allocator/list; all source and destination AS slabs stay resident and
    // source slices remain unretired until `tlas_build_epoch + 2` and `sub_seq + 2`.
    let recorded: windows::core::Result<()> = unsafe {
        (|| {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            for item in &batch {
                list4.CopyRaytracingAccelerationStructure(
                    item.dst_va,
                    item.src_va,
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
                );
            }
            list.ResourceBarrier(&[uav_barrier(None)]);
            list.Close()
        })()
    };
    if let Err(e) = recorded {
        tracing::warn!(error = %e, "dxr: static BLAS compaction list failed");
        for item in batch {
            SlabAllocator::free_in_slabs(&mut state.slabs.slabs, item.dst_slice);
        }
        state.free_lists.push((allocator, list));
        return None;
    }

    let mut swapped = 0u64;
    {
        let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
        for item in &batch {
            let Some(cur) = map.get(&item.guest_key) else {
                SlabAllocator::free_in_slabs(&mut state.slabs.slabs, item.dst_slice);
                continue;
            };
            if cur.generation != item.generation {
                SlabAllocator::free_in_slabs(&mut state.slabs.slabs, item.dst_slice);
                continue;
            }
            // Preserve `generation` so any in-flight multi-geometry upgrade decoded from
            // Frame-0 readback still replaces a compacted Frame-0 BLAS cleanly.
            let old = cur.clone();
            map.insert(
                item.guest_key,
                HostAs {
                    buffer: item.dst_buf.clone(),
                    slice: item.dst_slice,
                    gpu_va: item.dst_va,
                    kind: AsKind::Blas,
                    flags: item.flags,
                    num_instances: 0,
                    built: true,
                    generation: old.generation,
                    topo_hash: old.topo_hash,
                    update_scratch_bytes: old.update_scratch_bytes,
                },
            );
            state.retire_as(&old);
            state.insert_blas_hash(item.guest_key, item.dst_va);
            swapped += 1;
        }
    }
    if swapped > 0 {
        STATS[STAT_COMPACTED].fetch_add(swapped, Ordering::Relaxed);
    }
    let out: ID3D12CommandList = list.cast().ok()?;
    sub.lists.push((allocator, list));
    Some(out)
}

fn uav_barrier(resource: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: non-owning pointer for the lifetime of the barrier call.
                pResource: ManuallyDrop::new(unsafe {
                    match resource {
                        Some(r) => Some(std::mem::transmute_copy(r)),
                        None => None,
                    }
                }),
            }),
        },
    }
}

fn record_empty_tlas_init(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    tlas_vas: &[u64],
) -> Option<ID3D12CommandList> {
    let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
        NumDescs: 0,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            InstanceDescs: 0,
        },
    };
    let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: valid device5 and zero-instance TLAS inputs.
    unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre) };

    let (allocator, list) = state.take_list(host)?;
    name(&list, || "victory dxr ops".into());
    let list4: ID3D12GraphicsCommandList4 = list.cast().ok()?;
    let (_, scratch_gpu) = sub.ensure_scratch(host, pre.ScratchDataSizeInBytes.max(256))?;

    // SAFETY: allocator/list are idle (returned after GPU fence); live resources.
    let recorded: windows::core::Result<()> = unsafe {
        (|| {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            let map = as_map().read().unwrap_or_else(|e| e.into_inner());
            for &guest_va in tlas_vas {
                let Some(entry) = map.get(&guest_va) else {
                    continue;
                };
                if entry.built || entry.slice.size < pre.ResultDataMaxSizeInBytes {
                    continue;
                }
                let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: entry.gpu_va,
                    Inputs: inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: scratch_gpu,
                };
                list4.BuildRaytracingAccelerationStructure(&desc, None);
                list.ResourceBarrier(&[uav_barrier(None)]);
            }
            list.Close()
        })()
    };
    if let Err(e) = recorded {
        tracing::warn!(error = %e, "dxr: empty TLAS init list failed");
        state.free_lists.push((allocator, list));
        return None;
    }
    {
        let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
        for &guest_va in tlas_vas {
            if let Some(entry) = map.get_mut(&guest_va)
                && entry.slice.size >= pre.ResultDataMaxSizeInBytes
            {
                entry.built = true;
            }
        }
    }
    let out: ID3D12CommandList = list.cast().ok()?;
    sub.lists.push((allocator, list));
    Some(out)
}

fn record_ops(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    owner: &HostList,
    ops: &[DxrOp],
    built_tlas: &mut bool,
) -> Option<ID3D12CommandList> {
    let (allocator, list) = state.take_list(host)?;
    name(&list, || "victory dxr ops".into());
    let list4: ID3D12GraphicsCommandList4 = list.cast().ok()?;

    // Track per-command-list resource state of reusable buffers that start in COMMON.
    let mut hash_in_srv = false;
    let mut host_inst_in_srv = false;
    let mut extract_in_srv = false;
    let compact_start_slot = sub.compact_queries.len();
    let mut compact_uav_active: Option<(ID3D12Resource, ID3D12Resource)> = None;
    let mut recorded_any = false;

    // SAFETY: allocator/list are idle; all buffers referenced are kept in `sub` or slabs.
    let res: windows::core::Result<()> = unsafe {
        (|| {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            // Ensure prior GPU compute/copy work (e.g. skinning or BVH4 rehydration) is visible.
            list.ResourceBarrier(&[uav_barrier(None)]);

            for op in ops {
                match op {
                    DxrOp::BuildBlas {
                        dest_va,
                        source_va,
                        flags,
                        geometries,
                        postbuild,
                    } => {
                        if record_build_blas(
                            host, device5, state, sub, &list, &list4, *dest_va, *source_va, *flags, geometries, postbuild,
                        ) {
                            recorded_any = true;
                        }
                    }
                    DxrOp::CopyAs {
                        dest_va,
                        source_va,
                        mode,
                    } => {
                        if record_copy_as(host, state, &list, &list4, *dest_va, *source_va, *mode) {
                            recorded_any = true;
                        }
                    }
                    DxrOp::ExtractStaticBlas {
                        dest_va,
                        size_bytes,
                    } => {
                        if record_extract_static_blas(
                            host,
                            device5,
                            state,
                            sub,
                            owner,
                            &list,
                            &list4,
                            *dest_va,
                            *size_bytes,
                            &mut extract_in_srv,
                            &mut compact_uav_active,
                        ) {
                            recorded_any = true;
                        }
                    }
                    DxrOp::BuildTlas {
                        dest_va,
                        source_va,
                        flags,
                        num_instances,
                        instance_descs_va,
                        postbuild,
                    } => {
                        if record_build_tlas(
                            host,
                            device5,
                            state,
                            sub,
                            &list,
                            &list4,
                            *dest_va,
                            *source_va,
                            *flags,
                            *num_instances,
                            *instance_descs_va,
                            postbuild,
                            &mut hash_in_srv,
                            &mut host_inst_in_srv,
                            built_tlas,
                        ) {
                            recorded_any = true;
                        }
                    }
                }
            }
            if let Some((uav_buf, rb_buf)) = compact_uav_active.take() {
                let count = sub.compact_queries.len().saturating_sub(compact_start_slot);
                if count > 0 {
                    list.ResourceBarrier(&[transition(
                        &uav_buf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    let off = (compact_start_slot as u64) * 8;
                    let len = (count as u64) * 8;
                    list.CopyBufferRegion(&rb_buf, off, &uav_buf, off, len);
                    list.ResourceBarrier(&[transition(
                        &uav_buf,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                } else {
                    list.ResourceBarrier(&[transition(
                        &uav_buf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
            }
            list.Close()
        })()
    };

    if let Err(e) = res {
        tracing::warn!(error = %e, "dxr: command list recording failed");
        sub.compact_queries.truncate(compact_start_slot);
        state.free_lists.push((allocator, list));
        return None;
    }
    if !recorded_any {
        sub.compact_queries.truncate(compact_start_slot);
        state.free_lists.push((allocator, list));
        return None;
    }
    let out: ID3D12CommandList = list.cast().ok()?;
    sub.lists.push((allocator, list));
    Some(out)
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_build_blas(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    list: &ID3D12GraphicsCommandList,
    list4: &ID3D12GraphicsCommandList4,
    dest_va: u64,
    source_va: u64,
    raw_flags: u32,
    geometries: &[GeometrySpec],
    postbuild: &[PostbuildSpec],
) -> bool {
    let key = dest_va & !0xFF;
    if key == 0 || geometries.is_empty() {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    let mut host_geoms = Vec::with_capacity(geometries.len());
    for g in geometries {
        match g.kind {
            GeometryKind::Triangles {
                transform3x4,
                index_format,
                vertex_format,
                index_count,
                vertex_count,
                index_buffer,
                vertex_buffer_va,
                vertex_buffer_stride,
            } => {
                let pc_vtx_fmt = crate::resource::pc_format(vertex_format);
                let (vtx_align, vtx_min_bytes, vtx_fmt_ok) = match DXGI_FORMAT(pc_vtx_fmt as i32) {
                    DXGI_FORMAT_R32G32B32_FLOAT => (4u64, 12u64, true),
                    DXGI_FORMAT_R32G32_FLOAT => (4, 8, true),
                    DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_SNORM => (2, 8, true),
                    DXGI_FORMAT_R16G16_FLOAT | DXGI_FORMAT_R16G16_SNORM => (2, 4, true),
                    _ => (4, 12, false),
                };
                let (vb_gpu, vc, vb_stride) = if vtx_fmt_ok
                    && vertex_buffer_va != 0
                    && vertex_count > 0
                    && vertex_buffer_stride % vtx_align == 0
                {
                    let stride = vertex_buffer_stride.max(vtx_min_bytes);
                    let len = u64::from(vertex_count)
                        .saturating_sub(1)
                        .saturating_mul(stride)
                        .saturating_add(vtx_min_bytes);
                    match crate::shadow::find(vertex_buffer_va, len) {
                        Some(s) if s.left(vertex_buffer_va) >= len => {
                            let gpu = s.gpu(vertex_buffer_va);
                            if gpu % vtx_align == 0 {
                                sub.kept_shadows.push(s);
                                (gpu, vertex_count, vertex_buffer_stride)
                            } else {
                                (0, 0, 0)
                            }
                        }
                        _ => (0, 0, 0),
                    }
                } else {
                    (0, 0, 0)
                };
                let pc_idx_fmt = crate::resource::pc_format(index_format);
                let (ib_gpu, ic, idx_fmt) = if index_buffer != 0 && index_count >= 3 && vc > 0 {
                    let elem = match DXGI_FORMAT(pc_idx_fmt as i32) {
                        DXGI_FORMAT_R16_UINT => Some(2u64),
                        DXGI_FORMAT_R32_UINT => Some(4u64),
                        _ => None,
                    };
                    let aligned_ic = index_count - (index_count % 3);
                    if let Some(elem) = elem {
                        let len = u64::from(aligned_ic).saturating_mul(elem);
                        match crate::shadow::find(index_buffer, len) {
                            Some(s) if s.left(index_buffer) >= len => {
                                let gpu = s.gpu(index_buffer);
                                if gpu % elem == 0 {
                                    sub.kept_shadows.push(s);
                                    (gpu, aligned_ic, DXGI_FORMAT(pc_idx_fmt as i32))
                                } else {
                                    (0, 0, DXGI_FORMAT_UNKNOWN)
                                }
                            }
                            _ => (0, 0, DXGI_FORMAT_UNKNOWN),
                        }
                    } else {
                        (0, 0, DXGI_FORMAT_UNKNOWN)
                    }
                } else {
                    (0, 0, DXGI_FORMAT_UNKNOWN)
                };
                // If an indexed geometry failed to resolve its index buffer, deactivate it
                // rather than feeding raw vertices with mismatched count/stride. For
                // non-indexed triangles, DXR requires VertexCount to be a multiple of 3.
                let (vb_gpu, vc, vb_stride) = if index_buffer != 0 && index_count > 0 && ib_gpu == 0 {
                    (0, 0, 0)
                } else if ib_gpu == 0 {
                    let aligned_vc = vc - (vc % 3);
                    if aligned_vc == 0 {
                        (0, 0, 0)
                    } else {
                        (vb_gpu, aligned_vc, vb_stride)
                    }
                } else {
                    (vb_gpu, vc, vb_stride)
                };
                let xf_gpu = if transform3x4 != 0 && vc > 0 {
                    match crate::shadow::find(transform3x4, 48) {
                        Some(s) if s.left(transform3x4) >= 48 => {
                            let gpu = s.gpu(transform3x4);
                            if gpu % 16 == 0 {
                                sub.kept_shadows.push(s);
                                gpu
                            } else {
                                0
                            }
                        }
                        _ => 0,
                    }
                } else {
                    0
                };
                let host_vtx_fmt = if vtx_fmt_ok {
                    DXGI_FORMAT(pc_vtx_fmt as i32)
                } else {
                    DXGI_FORMAT_R32G32B32_FLOAT
                };
                host_geoms.push(D3D12_RAYTRACING_GEOMETRY_DESC {
                    Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                    Flags: dynamic_geom_flags(g.flags),
                    Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                        Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                            Transform3x4: xf_gpu,
                            IndexFormat: idx_fmt,
                            VertexFormat: host_vtx_fmt,
                            IndexCount: ic,
                            VertexCount: vc,
                            IndexBuffer: ib_gpu,
                            VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                                StartAddress: vb_gpu,
                                StrideInBytes: vb_stride,
                            },
                        },
                    },
                });
            }
            GeometryKind::Aabbs {
                aabb_count,
                aabbs_va,
                aabbs_stride,
            } => {
                let (ab_gpu, cnt) = if aabbs_va != 0 && aabb_count > 0 && aabbs_stride % 8 == 0 {
                    let stride = aabbs_stride.max(24);
                    let len = aabb_count
                        .saturating_sub(1)
                        .saturating_mul(stride)
                        .saturating_add(24);
                    match crate::shadow::find(aabbs_va, len) {
                        Some(s) if s.left(aabbs_va) >= len => {
                            let gpu = s.gpu(aabbs_va);
                            if gpu % 8 == 0 {
                                sub.kept_shadows.push(s);
                                (gpu, aabb_count)
                            } else {
                                (0, 0)
                            }
                        }
                        _ => (0, 0),
                    }
                } else {
                    (0, 0)
                };
                host_geoms.push(D3D12_RAYTRACING_GEOMETRY_DESC {
                    Type: D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS,
                    Flags: dynamic_geom_flags(g.flags),
                    Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                        AABBs: D3D12_RAYTRACING_GEOMETRY_AABBS_DESC {
                            AABBCount: cnt,
                            AABBs: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                                StartAddress: ab_gpu,
                                StrideInBytes: if cnt > 0 { aabbs_stride } else { 0 },
                            },
                        },
                    },
                });
            }
        }
    }

    const ALLOW_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0 as u32;
    const PERFORM_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0 as u32;

    let topo_hash = blas_topology_hash(&host_geoms);
    let mut fast_update: Option<(u64, u64)> = None;
    let (flags, host_src_va) = if raw_flags & PERFORM_UPDATE != 0 && source_va != 0 {
        let src_key = source_va & !0xFF;
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        if let Some(src) = map.get(&src_key)
            && src.kind == AsKind::Blas
            && src.built
            && (src.flags & ALLOW_UPDATE) != 0
            && src.topo_hash == topo_hash
            && src.topo_hash != 0
        {
            if src_key == key && src.update_scratch_bytes > 0 {
                fast_update = Some((src.gpu_va, src.update_scratch_bytes));
            }
            (raw_flags, src.gpu_va)
        } else {
            (raw_flags & !PERFORM_UPDATE, 0)
        }
    } else {
        (raw_flags & !PERFORM_UPDATE, 0)
    };

    let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(flags as i32),
        NumDescs: host_geoms.len() as u32,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            pGeometryDescs: host_geoms.as_ptr(),
        },
    };

    let (host_dst_va, scratch_need) = if let Some((gpu_va, update_scratch)) = fast_update {
        (gpu_va, update_scratch.max(256))
    } else {
        let (result_bytes, scratch_bytes, update_scratch_bytes) =
            if let Some(&cached) = state.prebuild_cache.get(&(flags, topo_hash)) {
                cached
            } else {
                let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
                // SAFETY: valid device5 and geometry array alive through call.
                unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre) };
                let entry = (
                    pre.ResultDataMaxSizeInBytes,
                    pre.ScratchDataSizeInBytes,
                    pre.UpdateScratchDataSizeInBytes,
                );
                if entry.0 > 0 {
                    if state.prebuild_cache.len() >= MAX_PREBUILD_CACHE {
                        state.prebuild_cache.clear();
                    }
                    state.prebuild_cache.insert((flags, topo_hash), entry);
                }
                entry
            };
        if result_bytes == 0 {
            STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let Some(host_dst) = allocate_or_reuse_as(
            host,
            state,
            key,
            AsKind::Blas,
            result_bytes,
            flags,
            0,
            topo_hash,
            update_scratch_bytes,
        ) else {
            STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let scratch_need = if flags & PERFORM_UPDATE != 0 {
            update_scratch_bytes.max(256)
        } else {
            scratch_bytes.max(256)
        };
        (host_dst.gpu_va, scratch_need)
    };
    let Some((_, scratch_gpu)) = sub.ensure_scratch(host, scratch_need) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };

    let host_postbuild = translate_postbuild(sub, postbuild);
    let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        DestAccelerationStructureData: host_dst_va,
        Inputs: inputs,
        SourceAccelerationStructureData: host_src_va,
        ScratchAccelerationStructureData: scratch_gpu,
    };
    // SAFETY: open list4; dest, src, scratch, geometry buffers and postbuild buffers live.
    unsafe {
        list4.BuildRaytracingAccelerationStructure(
            &desc,
            (!host_postbuild.is_empty()).then_some(host_postbuild.as_slice()),
        );
        list.ResourceBarrier(&[uav_barrier(None)]);
    }
    state.insert_blas_hash(key, host_dst_va);
    STATS[STAT_DYNAMIC_BLAS].fetch_add(1, Ordering::Relaxed);
    true
}

unsafe fn record_copy_as(
    host: &Host,
    state: &mut DxrState,
    list: &ID3D12GraphicsCommandList,
    list4: &ID3D12GraphicsCommandList4,
    dest_va: u64,
    source_va: u64,
    mode: u32,
) -> bool {
    let dst_key = dest_va & !0xFF;
    let src_key = source_va & !0xFF;
    if dst_key == 0 || src_key == 0 {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    let src = {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        map.get(&src_key).cloned()
    };
    let Some(src) = src else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    if !src.built {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    const ALLOW_COMPACTION: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0 as u32;
    let host_mode = match mode {
        0 => D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE,
        1 => {
            if src.flags & ALLOW_COMPACTION != 0 {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT
            } else {
                // Fall back to CLONE if our host build did not set ALLOW_COMPACTION.
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE
            }
        }
        _ => {
            STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
            return false;
        }
    };
    let Some(dst) = allocate_or_reuse_as(
        host,
        state,
        dst_key,
        src.kind,
        src.slice.size,
        src.flags,
        src.num_instances,
        0,
        0,
    ) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    // SAFETY: open list4; both AS buffers are in RAYTRACING_ACCELERATION_STRUCTURE state.
    unsafe {
        list4.CopyRaytracingAccelerationStructure(dst.gpu_va, src.gpu_va, host_mode);
        list.ResourceBarrier(&[uav_barrier(None)]);
    }
    if src.kind == AsKind::Blas {
        state.insert_blas_hash(dst_key, dst.gpu_va);
    }
    STATS[STAT_COPIES].fetch_add(1, Ordering::Relaxed);
    true
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_extract_static_blas(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    owner: &HostList,
    list: &ID3D12GraphicsCommandList,
    list4: &ID3D12GraphicsCommandList4,
    dest_va: u64,
    size_bytes: u64,
    extract_in_srv: &mut bool,
    compact_uav_active: &mut Option<(ID3D12Resource, ID3D12Resource)>,
) -> bool {
    let key = dest_va & !0xFF;
    if key == 0 {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    let Some((size_bytes, max_nodes)) = probe_bvh_bounds(key, size_bytes) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let Some(pass) = extract_bvh4_pass(host) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let mem_binding = crate::descriptor::with_view_heap(|heap| {
        let ready = crate::memory::ready(owner, heap, &[], &[], &[], crate::memory::Use::Reads, None)?;
        Some((heap.heap.clone(), heap.gpu_handle(ready.base), ready.directory, ready.count))
    })
    .flatten();
    let Some((view_heap, mem_table_gpu, directory_gpu, copy_limit)) = mem_binding else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };

    let vert_bytes = max_nodes.saturating_mul(72); // 2 tris × 3 verts × 12B float3
    let prev_buf_raw = sub.extract_verts.as_ref().map(|(b, _)| b.as_raw());
    let Some((vert_buf, vert_gpu)) = sub.ensure_extract_verts(host, vert_bytes) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    if prev_buf_raw != Some(vert_buf.as_raw()) {
        *extract_in_srv = false;
    }

    let geom = D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: static_geom_flags(true),
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: DXGI_FORMAT_UNKNOWN,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: 0,
                VertexCount: (max_nodes * 6) as u32,
                IndexBuffer: 0,
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: vert_gpu,
                    StrideInBytes: 12,
                },
            },
        },
    };
    let build_flags = static_blas_build_flags();
    let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: build_flags,
        NumDescs: 1,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            pGeometryDescs: &geom,
        },
    };
    let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: valid device5 and local geometry desc.
    unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre) };
    if pre.ResultDataMaxSizeInBytes == 0 {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }

    let Some(host_dst) = allocate_or_reuse_as(
        host,
        state,
        key,
        AsKind::Blas,
        pre.ResultDataMaxSizeInBytes,
        build_flags.0 as u32,
        0,
        0,
        0,
    ) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let Some((_, scratch_gpu)) = sub.ensure_scratch(host, pre.ScratchDataSizeInBytes.max(256)) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };

    // Reserve a slice of this submission's readback buffer for async multi-geometry upgrade.
    let dump_bytes = 16u64 + max_nodes.saturating_mul(64);
    let dump_copy = if sub.readback_used.saturating_add(dump_bytes) <= sub.readback_cap
        && let Some((rb_buf, _)) = sub.node_readback.as_ref().map(|(b, sz)| (b.clone(), *sz))
        && let Some((staging_buf, staging_gpu)) = sub.ensure_node_staging(host, dump_bytes)
    {
        let rb_off = sub.readback_used;
        sub.readback_used = rb_off.saturating_add(dump_bytes).next_multiple_of(64);
        sub.readbacks.push(StaticReadbackRecord {
            guest_key: key,
            generation: host_dst.generation,
            offset: rb_off,
            len: dump_bytes,
        });
        Some((staging_buf, staging_gpu, rb_buf, rb_off, dump_bytes))
    } else {
        None
    };

    // SAFETY: open list4; view_heap, directory, vert_buf, staging/readback, AS slab, and scratch are live.
    unsafe {
        if *extract_in_srv {
            list.ResourceBarrier(&[transition(
                &vert_buf,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            *extract_in_srv = false;
        }
        if let Some((staging_buf, _, _, _, _)) = &dump_copy {
            list.ResourceBarrier(&[transition(
                staging_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        list.SetDescriptorHeaps(&[Some(view_heap)]);
        list.SetComputeRootSignature(&pass.root);
        list.SetPipelineState(&pass.pso);
        let (dump_nodes, u1_gpu) = match &dump_copy {
            Some((_, staging_gpu, _, _, _)) => (max_nodes as u32, *staging_gpu),
            None => (0u32, vert_gpu),
        };
        let constants = [
            key as u32,
            (key >> 32) as u32,
            max_nodes as u32,
            size_bytes.min(u64::from(u32::MAX)) as u32,
            dump_nodes,
            copy_limit,
        ];
        list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
        list.SetComputeRootShaderResourceView(1, directory_gpu);
        list.SetComputeRootUnorderedAccessView(2, vert_gpu);
        list.SetComputeRootUnorderedAccessView(3, u1_gpu);
        list.SetComputeRootDescriptorTable(4, mem_table_gpu);
        let groups = (max_nodes as u32).div_ceil(64);
        list.Dispatch(groups, 1, 1);

        if let Some((staging_buf, _, rb_buf, rb_off, dump_len)) = dump_copy {
            list.ResourceBarrier(&[transition(
                &staging_buf,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyBufferRegion(&rb_buf, rb_off, &staging_buf, 0, dump_len);
            list.ResourceBarrier(&[transition(
                &staging_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }

        list.ResourceBarrier(&[transition(
            &vert_buf,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        )]);
        *extract_in_srv = true;

        let compact_postbuild = if compact_static_blas()
            && sub.compact_queries.len() < COMPACT_INFO_SLOTS
            && let Some((uav_buf, uav_gpu, rb_buf)) = sub.ensure_compact_info(host)
        {
            if compact_uav_active.is_none() {
                list.ResourceBarrier(&[transition(
                    &uav_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                *compact_uav_active = Some((uav_buf, rb_buf));
            }
            let slot = sub.compact_queries.len() as u32;
            sub.compact_queries.push(CompactQueryRecord {
                guest_key: key,
                generation: host_dst.generation,
                uncompacted_size: host_dst.slice.size,
                slot,
            });
            Some([D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: uav_gpu + u64::from(slot) * 8,
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
            }])
        } else {
            None
        };

        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: host_dst.gpu_va,
            Inputs: inputs,
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_gpu,
        };
        list4.BuildRaytracingAccelerationStructure(
            &desc,
            compact_postbuild.as_ref().map(|p| p.as_slice()),
        );
        list.ResourceBarrier(&[uav_barrier(None)]);
    }
    state.insert_blas_hash(key, host_dst.gpu_va);
    STATS[STAT_STATIC_BLAS].fetch_add(1, Ordering::Relaxed);
    true
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_build_tlas(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    list: &ID3D12GraphicsCommandList,
    list4: &ID3D12GraphicsCommandList4,
    dest_va: u64,
    _source_va: u64,
    raw_flags: u32,
    num_instances: u32,
    instance_descs_va: u64,
    postbuild: &[PostbuildSpec],
    hash_in_srv: &mut bool,
    host_inst_in_srv: &mut bool,
    built_tlas: &mut bool,
) -> bool {
    let key = dest_va & !0xFF;
    if key == 0 {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }

    // Upload only modified ranges of the GPU BLAS hash table before patching instances.
    let hash_gpu_buf = state.ensure_hash_gpu(host);
    let Some(hash_gpu_buf) = hash_gpu_buf else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    if state.hash_dirty.is_dirty() {
        let Some((upload_buf, upload_ptr)) = sub.ensure_hash_upload(host) else {
            STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let ranges = state.hash_dirty.take_ranges();
        if !ranges.is_empty() {
            // SAFETY: `upload_ptr` maps `HASH_TABLE_BYTES` matching `state.blas_table`.
            unsafe {
                if *hash_in_srv {
                    list.ResourceBarrier(&[transition(
                        &hash_gpu_buf,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                }
                for (start_slot, end_slot) in ranges {
                    let byte_off = (start_slot * 16) as u64;
                    let byte_len = ((end_slot - start_slot + 1) * 16) as u64;
                    std::ptr::copy_nonoverlapping(
                        state.blas_table.as_ptr().add(start_slot).cast::<u8>(),
                        upload_ptr.add(byte_off as usize),
                        byte_len as usize,
                    );
                    list.CopyBufferRegion(&hash_gpu_buf, byte_off, &upload_buf, byte_off, byte_len);
                }
                list.ResourceBarrier(&[transition(
                    &hash_gpu_buf,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
            }
            *hash_in_srv = true;
        }
    }

    let mut effective_instances = num_instances;
    let mut host_inst_gpu = 0u64;

    if effective_instances > 0 && instance_descs_va != 0 {
        let inst_bytes = u64::from(effective_instances).saturating_mul(64);
        let guest_shadow = crate::shadow::find(instance_descs_va, inst_bytes);
        let pass = patch_instances_pass(host);
        let stats_buf = state.ensure_patch_stats(host);
        if let (Some(guest_shadow), Some(pass), Some((_, stats_gpu))) = (guest_shadow, pass, stats_buf)
            && guest_shadow.left(instance_descs_va) >= inst_bytes
        {
            let prev_raw = sub.host_instances.as_ref().map(|(b, _)| b.as_raw());
            let Some((inst_buf, inst_gpu)) = sub.ensure_host_instances(host, inst_bytes) else {
                STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
                return false;
            };
            if prev_raw != Some(inst_buf.as_raw()) {
                *host_inst_in_srv = false;
            }
            let guest_gpu = guest_shadow.gpu(instance_descs_va);
            let aligned_base = guest_gpu & !255;
            let src_offset = (guest_gpu & 255) as u32;
            sub.kept_shadows.push(guest_shadow);
            // SAFETY: open list; all root views and buffers live through submission.
            unsafe {
                if *host_inst_in_srv {
                    list.ResourceBarrier(&[transition(
                        &inst_buf,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                    *host_inst_in_srv = false;
                }
                list.SetComputeRootSignature(&pass.root);
                list.SetPipelineState(&pass.pso);
                let constants = [effective_instances, src_offset, HASH_MASK, 0u32];
                list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.SetComputeRootShaderResourceView(1, aligned_base);
                list.SetComputeRootShaderResourceView(2, hash_gpu_buf.GetGPUVirtualAddress());
                list.SetComputeRootUnorderedAccessView(3, inst_gpu);
                list.SetComputeRootUnorderedAccessView(4, stats_gpu);
                list.Dispatch(effective_instances.div_ceil(64), 1, 1);
                list.ResourceBarrier(&[transition(
                    &inst_buf,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
            }
            *hash_in_srv = true;
            *host_inst_in_srv = true;
            host_inst_gpu = inst_gpu;
        } else {
            effective_instances = 0;
        }
    } else {
        effective_instances = 0;
    }

    // Always rebuild TLAS from scratch with PREFER_FAST_TRACE: D3D12 forbids TLAS
    // `PERFORM_UPDATE` whenever any instance transitions `AccelerationStructure` or
    // `InstanceMask` between 0 and non-0, and a 3.5K-instance fast-trace TLAS build
    // takes <0.1 ms while reducing overlapping AABB candidate callbacks in AnyHit loops.
    const ALLOW_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0 as u32;
    const PERFORM_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0 as u32;
    const PREFER_FAST_BUILD: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD.0 as u32;
    const PREFER_FAST_TRACE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE.0 as u32;
    let flags = (raw_flags & !(ALLOW_UPDATE | PERFORM_UPDATE | PREFER_FAST_BUILD)) | PREFER_FAST_TRACE;
    let host_src_va = 0u64;

    let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(flags as i32),
        NumDescs: effective_instances,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            InstanceDescs: host_inst_gpu,
        },
    };
    let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: valid device5 and TLAS inputs.
    unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre) };
    if pre.ResultDataMaxSizeInBytes == 0 {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }

    let Some(host_dst) = allocate_or_reuse_as(
        host,
        state,
        key,
        AsKind::Tlas,
        pre.ResultDataMaxSizeInBytes,
        flags,
        effective_instances,
        0,
        0,
    ) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let scratch_need = if flags & PERFORM_UPDATE != 0 {
        pre.UpdateScratchDataSizeInBytes.max(256)
    } else {
        pre.ScratchDataSizeInBytes.max(256)
    };
    let Some((_, scratch_gpu)) = sub.ensure_scratch(host, scratch_need) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };

    let host_postbuild = translate_postbuild(sub, postbuild);
    let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        DestAccelerationStructureData: host_dst.gpu_va,
        Inputs: inputs,
        SourceAccelerationStructureData: host_src_va,
        ScratchAccelerationStructureData: scratch_gpu,
    };
    // SAFETY: open list4; dest, src, scratch, host_instances, and postbuild buffers live.
    unsafe {
        list4.BuildRaytracingAccelerationStructure(
            &desc,
            (!host_postbuild.is_empty()).then_some(host_postbuild.as_slice()),
        );
        list.ResourceBarrier(&[uav_barrier(None)]);
    }
    if !*built_tlas {
        state.tlas_build_epoch = state.tlas_build_epoch.saturating_add(1);
        *built_tlas = true;
    }
    STATS[STAT_TLAS].fetch_add(1, Ordering::Relaxed);
    true
}

fn translate_postbuild(
    sub: &mut DxrSubmission,
    postbuild: &[PostbuildSpec],
) -> Vec<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC> {
    if !suppress_builder() || postbuild.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(postbuild.len());
    for pb in postbuild {
        if pb.dest_buffer == 0 {
            continue;
        }
        if let Some(s) = crate::shadow::find(pb.dest_buffer, 16) {
            let gpu = s.gpu(pb.dest_buffer);
            sub.kept_shadows.push(s);
            out.push(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: gpu,
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE(pb.info_type as i32),
            });
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn allocate_or_reuse_as(
    host: &Host,
    state: &mut DxrState,
    guest_key: u64,
    kind: AsKind,
    min_bytes: u64,
    flags: u32,
    num_instances: u32,
    topo_hash: u64,
    update_scratch_bytes: u64,
) -> Option<HostAs> {
    let generation = state.bump_gen();
    let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
    if let Some(existing) = map.get_mut(&guest_key)
        && existing.kind == kind
        && existing.slice.size >= min_bytes
    {
        existing.flags = flags;
        existing.num_instances = num_instances;
        existing.built = true;
        existing.generation = generation;
        existing.topo_hash = topo_hash;
        existing.update_scratch_bytes = update_scratch_bytes;
        return Some(existing.clone());
    }
    let alloc_bytes = if kind == AsKind::Tlas {
        min_bytes.max(DEFAULT_TLAS_RESERVE)
    } else {
        min_bytes
    };
    let (buffer, slice, gpu_va) = state.slabs.alloc(host, alloc_bytes)?;
    let entry = HostAs {
        buffer,
        slice,
        gpu_va,
        kind,
        flags,
        num_instances,
        built: true,
        generation,
        topo_hash,
        update_scratch_bytes,
    };
    if let Some(old) = map.insert(guest_key, entry.clone()) {
        state.retire_as(&old);
    }
    Some(entry)
}

/// FNV-1a topology fingerprint of a BLAS's geometry descriptors so `PERFORM_UPDATE`
/// only runs when geometry count, types, flags, formats, counts, and transform
/// nullness match the source BLAS identically (D3D12 DXR spec requirement).
fn blas_topology_hash(geoms: &[D3D12_RAYTRACING_GEOMETRY_DESC]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    mix(geoms.len() as u64);
    for g in geoms {
        mix(g.Type.0 as u64);
        mix(g.Flags.0 as u64);
        if g.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES {
            // SAFETY: `Type == TRIANGLES`.
            let t = unsafe { g.Anonymous.Triangles };
            mix(t.IndexFormat.0 as u64);
            mix(t.VertexFormat.0 as u64);
            mix(u64::from(t.IndexCount));
            mix(u64::from(t.VertexCount));
            mix(u64::from(t.Transform3x4 != 0));
            mix(u64::from(t.IndexBuffer != 0));
            mix(u64::from(t.VertexBuffer.StartAddress != 0));
        } else {
            // SAFETY: `Type == PROCEDURAL_PRIMITIVE_AABBS`.
            let a = unsafe { g.Anonymous.AABBs };
            mix(a.AABBCount);
            mix(u64::from(a.AABBs.StartAddress != 0));
        }
    }
    if h == 0 { 1 } else { h }
}

// ============================================================================
// Pure-Rust unpacked BVH4 TriangleNode decoder (CPU, zero D3D12 dependency)
// ============================================================================

/// Extracts one triangle's 3 vertices from a 64-byte TriangleNode's 4 vertices
/// using `TRIANGLE_VERTICES` (`tri0: [0,1,2]`, `tri1: [1,3,2]`) and the winding
/// permutation byte from `dw[15]`. Returns `None` for invalid/inactive triangles.
fn decode_node_triangle(verts: &[[f32; 3]; 4], map: [usize; 3], id_byte: u32) -> Option<[[f32; 3]; 3]> {
    let i = (id_byte & 3) as usize;
    let j = ((id_byte >> 2) & 3) as usize;
    if i >= 3 || j >= 3 || i == j {
        return None;
    }
    let first = (3 - i - j) & 3;
    let tri = [verts[map[first]], verts[map[i]], verts[map[j]]];
    if tri.iter().any(|v| v.iter().any(|c| c.is_nan())) {
        return None;
    }
    Some(tri)
}

/// Decodes a readback buffer containing `[num_elements, offset_to_elements, dump_nodes, size_bytes]`
/// followed by `dump_nodes` unpacked 64-byte console BVH4 TriangleNodes into a multi-geometry
/// [`PendingStaticUpgrade`] preserving exact `GeometryIndex`, `PrimitiveIndex`, and `OPAQUE` flags.
///
/// Returns `None` if the dump is invalid, empty, exceeds hard OOM bounds, or is already
/// bit-exact with Frame 0's single-geometry opaque sequential extraction.
fn decode_static_bvh4_nodes(guest_key: u64, generation: u64, data: &[u8]) -> Option<PendingStaticUpgrade> {
    if data.len() < 16 {
        return None;
    }
    let read_u32 = |off: usize| u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let num_elements = read_u32(0) as usize;
    let offset_to_elements = read_u32(4);
    let dump_nodes = read_u32(8) as usize;
    let size_bytes = read_u32(12);
    if offset_to_elements < 64 || num_elements == 0 || dump_nodes == 0 || size_bytes < offset_to_elements {
        return None;
    }
    let avail_nodes = (data.len() - 16) / 64;
    let max_by_size = ((size_bytes - offset_to_elements) / 64) as usize;
    let node_count = num_elements.min(dump_nodes).min(avail_nodes).min(max_by_size);
    if node_count == 0 {
        return None;
    }

    #[derive(Clone, Copy)]
    struct GeomMeta {
        has_tris: bool,
        opaque: bool,
        max_prim: u32,
    }
    let mut geoms: Vec<GeomMeta> = Vec::new();
    let mut needs_upgrade = false;
    let mut any_tri = false;

    let parse_node = |node_idx: usize| -> ([[f32; 3]; 4], u32, u32, u32, u32) {
        let base = 16 + node_idx * 64;
        let dw = |k: usize| read_u32(base + k * 4);
        let verts = [
            [f32::from_bits(dw(0)), f32::from_bits(dw(1)), f32::from_bits(dw(2))],
            [f32::from_bits(dw(3)), f32::from_bits(dw(4)), f32::from_bits(dw(5))],
            [f32::from_bits(dw(6)), f32::from_bits(dw(7)), f32::from_bits(dw(8))],
            [f32::from_bits(dw(9)), f32::from_bits(dw(10)), f32::from_bits(dw(11))],
        ];
        (verts, dw(12), dw(13), dw(14), dw(15))
    };

    // Pass 1: discover per-geometry primitive bounds, opacity, and upgrade necessity.
    for node_idx in 0..node_count {
        let (verts, dw12, prim0, prim1, dw15) = parse_node(node_idx);
        let geom_idx = (dw12 >> 8) as usize;
        let is_opaque = (dw12 & 1) != 0;
        let has_tri1 = (dw12 & 0x80) != 0;

        let t0 = decode_node_triangle(&verts, [0, 1, 2], dw15 & 0xFF);
        let t1 = if has_tri1 {
            decode_node_triangle(&verts, [1, 3, 2], (dw15 >> 8) & 0xFF)
        } else {
            None
        };
        if t0.is_none() && t1.is_none() {
            continue;
        }
        if geom_idx >= MAX_UPGRADE_GEOMS {
            return None;
        }
        if geoms.len() <= geom_idx {
            geoms.resize(
                geom_idx + 1,
                GeomMeta {
                    has_tris: false,
                    opaque: true,
                    max_prim: 0,
                },
            );
        }
        let meta = &mut geoms[geom_idx];
        for (sub_idx, tri_opt, prim_idx) in [(0usize, t0, prim0), (1usize, t1, prim1)] {
            if tri_opt.is_none() {
                continue;
            }
            if (prim_idx as usize) >= MAX_UPGRADE_TRIS {
                return None;
            }
            any_tri = true;
            if !meta.has_tris || prim_idx > meta.max_prim {
                meta.max_prim = prim_idx;
            }
            meta.has_tris = true;
            meta.opaque &= is_opaque;
            let frame0_prim = (node_idx * 2 + sub_idx) as u32;
            if geom_idx != 0 || !is_opaque || prim_idx != frame0_prim {
                needs_upgrade = true;
            }
        }
    }

    if !any_tri || !needs_upgrade {
        return None;
    }

    let mut total_tris = 0usize;
    let mut geom_tri_offsets = Vec::with_capacity(geoms.len());
    let mut upgraded_geoms = Vec::with_capacity(geoms.len());
    for g in &geoms {
        let prim_count = if g.has_tris { (g.max_prim as usize) + 1 } else { 0 };
        total_tris = total_tris.checked_add(prim_count)?;
        if total_tris > MAX_UPGRADE_TRIS {
            return None;
        }
        let offset_tris = total_tris - prim_count;
        geom_tri_offsets.push(offset_tris);
        upgraded_geoms.push(UpgradedGeom {
            vert_offset: (offset_tris * 3) as u32,
            vert_count: (prim_count * 3) as u32,
            opaque: g.opaque,
        });
    }

    // Pass 2: scatter valid triangles into exact `PrimitiveIndex` slots; holes stay NaN.
    let nan = f32::from_bits(0x7FC0_0000);
    let mut vertices = vec![[nan; 3]; total_tris * 3];
    for node_idx in 0..node_count {
        let (verts, dw12, prim0, prim1, dw15) = parse_node(node_idx);
        let geom_idx = (dw12 >> 8) as usize;
        let has_tri1 = (dw12 & 0x80) != 0;

        if let Some(tri0) = decode_node_triangle(&verts, [0, 1, 2], dw15 & 0xFF) {
            let slot = (geom_tri_offsets[geom_idx] + prim0 as usize) * 3;
            vertices[slot] = tri0[0];
            vertices[slot + 1] = tri0[1];
            vertices[slot + 2] = tri0[2];
        }
        if has_tri1 && let Some(tri1) = decode_node_triangle(&verts, [1, 3, 2], (dw15 >> 8) & 0xFF) {
            let slot = (geom_tri_offsets[geom_idx] + prim1 as usize) * 3;
            vertices[slot] = tri1[0];
            vertices[slot + 1] = tri1[1];
            vertices[slot + 2] = tri1[2];
        }
    }

    Some(PendingStaticUpgrade {
        guest_key,
        generation,
        geometries: upgraded_geoms,
        vertices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_dump(num_elements: u32, offset_to_elements: u32, dump_nodes: u32, nodes: &[[u32; 16]]) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + nodes.len() * 64);
        let size_bytes = offset_to_elements.saturating_add((nodes.len() as u32) * 64);
        for w in [num_elements, offset_to_elements, dump_nodes, size_bytes] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for n in nodes {
            for w in n {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    fn make_node(verts: [[f32; 3]; 4], dw12: u32, prim0: u32, prim1: u32, dw15: u32) -> [u32; 16] {
        let mut dw = [0u32; 16];
        for (vi, v) in verts.iter().enumerate() {
            dw[vi * 3] = v[0].to_bits();
            dw[vi * 3 + 1] = v[1].to_bits();
            dw[vi * 3 + 2] = v[2].to_bits();
        }
        dw[12] = dw12;
        dw[13] = prim0;
        dw[14] = prim1;
        dw[15] = dw15;
        dw
    }

    #[test]
    fn decode_multi_geom_opacity_winding_and_sparse_prims() {
        let v = [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        // id_byte: canonical [0,1,2] is i=1, j=2 -> (2<<2)|1 = 0x09; rotated [1,2,0] is i=2, j=0 -> (0<<2)|2 = 0x02.
        let n0 = make_node(v, (0 << 8) | 0x80 | 1, 0, 1, 0x09 | (0x09 << 8));
        // Geometry 2 (non-opaque, leaving geometry 1 empty), sparse prim0=5 (rotated), prim1=2 (canonical).
        let n1 = make_node(v, (2 << 8) | 0x80, 5, 2, 0x02 | (0x09 << 8));
        let dump = pack_dump(2, 128, 2, &[n0, n1]);

        let upg = decode_static_bvh4_nodes(0x1000, 7, &dump).expect("upgrade expected");
        assert_eq!(upg.guest_key, 0x1000);
        assert_eq!(upg.generation, 7);
        assert_eq!(
            upg.geometries,
            vec![
                UpgradedGeom { vert_offset: 0, vert_count: 6, opaque: true },
                UpgradedGeom { vert_offset: 6, vert_count: 0, opaque: true },
                UpgradedGeom { vert_offset: 6, vert_count: 18, opaque: false },
            ]
        );
        assert_eq!(upg.vertices.len(), 24);
        // Geom 0 prim 0: [v0, v1, v2]
        assert_eq!(&upg.vertices[0..3], &[v[0], v[1], v[2]]);
        // Geom 0 prim 1 (tri1 map [1,3,2]): [v1, v3, v2]
        assert_eq!(&upg.vertices[3..6], &[v[1], v[3], v[2]]);
        // Geom 2 prim 0, 1, 3, 4 are inactive NaN holes.
        for hole_prim in [0usize, 1, 3, 4] {
            let s = 6 + hole_prim * 3;
            assert!(upg.vertices[s][0].is_nan(), "expected NaN at hole prim {hole_prim}");
        }
        // Geom 2 prim 2 (tri1 canonical): [v1, v3, v2]
        assert_eq!(&upg.vertices[6 + 2 * 3..6 + 3 * 3], &[v[1], v[3], v[2]]);
        // Geom 2 prim 5 (tri0 rotated i=2,j=0 => first=1 => [v1, v2, v0])
        assert_eq!(&upg.vertices[6 + 5 * 3..6 + 6 * 3], &[v[1], v[2], v[0]]);
    }

    #[test]
    fn decode_mixed_opacity_in_same_geom_is_non_opaque() {
        let v = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        let n0 = make_node(v, (0 << 8) | 1, 0, 0, 0x09);
        let n1 = make_node(v, 0 << 8, 1, 0, 0x09);
        let dump = pack_dump(2, 64, 2, &[n0, n1]);

        let upg = decode_static_bvh4_nodes(0x2000, 1, &dump).expect("non-opaque must upgrade");
        assert_eq!(upg.geometries.len(), 1);
        assert!(!upg.geometries[0].opaque);
        assert_eq!(upg.geometries[0].vert_count, 6);
    }

    #[test]
    fn decode_skips_when_frame0_already_exact_and_upgrades_on_reorder() {
        let v = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        let exact = make_node(v, (0 << 8) | 0x80 | 1, 0, 1, 0x09 | (0x09 << 8));
        assert!(decode_static_bvh4_nodes(0x3000, 1, &pack_dump(1, 64, 1, &[exact])).is_none());

        // Same single opaque geometry, but PrimitiveIndex swapped -> must upgrade.
        let swapped = make_node(v, (0 << 8) | 0x80 | 1, 1, 0, 0x09 | (0x09 << 8));
        let upg = decode_static_bvh4_nodes(0x3000, 1, &pack_dump(1, 64, 1, &[swapped]))
            .expect("reordered prims must upgrade");
        assert_eq!(&upg.vertices[0..3], &[v[1], v[3], v[2]]);
        assert_eq!(&upg.vertices[3..6], &[v[0], v[1], v[2]]);
    }

    #[test]
    fn decode_rejects_corrupt_bounds_and_ignores_unmapped_nodes() {
        let v = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        // Geom index out of bounds
        let bad_geom = make_node(v, ((MAX_UPGRADE_GEOMS as u32) << 8) | 1, 0, 0, 0x09);
        assert!(decode_static_bvh4_nodes(0x4000, 1, &pack_dump(1, 64, 1, &[bad_geom])).is_none());
        // Prim index out of bounds
        let bad_prim = make_node(v, 1, MAX_UPGRADE_TRIS as u32, 0, 0x09);
        assert!(decode_static_bvh4_nodes(0x4000, 1, &pack_dump(1, 64, 1, &[bad_prim])).is_none());
        // Corrupt header offset < 64
        assert!(decode_static_bvh4_nodes(0x4000, 1, &pack_dump(1, 32, 1, &[make_node(v, 0, 0, 0, 0x09)])).is_none());
        // Unmapped sentinel (dw15 = 0xFFFF_FFFF) alongside valid non-opaque node is skipped cleanly.
        let unmapped = make_node([[0.0; 3]; 4], 0xFFFF_FF00, 0xDEAD_BEEF, 0xDEAD_BEEF, 0xFFFF_FFFF);
        let valid = make_node(v, 0, 3, 0, 0x09);
        let upg = decode_static_bvh4_nodes(0x4000, 1, &pack_dump(2, 64, 2, &[unmapped, valid]))
            .expect("valid node should survive unmapped neighbor");
        assert_eq!(upg.geometries.len(), 1);
        assert_eq!(upg.geometries[0].vert_count, 12);
    }

    #[test]
    fn extract_bvh4_hlsl_compiles() {
        let code = compile(EXTRACT_BVH4_HLSL, s!("main"), s!("cs_5_1")).expect("EXTRACT_BVH4_HLSL must compile");
        assert!(!code.is_empty());
    }

    #[test]
    fn slab_allocator_deferred_retire_coalesce_and_best_fit_reuse() {
        let mut alloc = SlabAllocator::<usize>::default();
        let make = |slab_size: u64, slab_idx: usize| -> Option<(usize, u64)> {
            let _ = slab_size;
            Some((slab_idx, (slab_idx as u64 + 1) << 32))
        };

        // Allocate four adjacent 256B-aligned slices inside slab 0.
        let (_, a, va_a) = alloc.alloc_with(200, make).unwrap();
        let (_, b, _) = alloc.alloc_with(512, make).unwrap();
        let (_, c, _) = alloc.alloc_with(256, make).unwrap();
        let (_, d, _) = alloc.alloc_with(1024, make).unwrap();
        assert_eq!(a, SlabSlice { slab_idx: 0, offset: 0, size: 256 });
        assert_eq!(b, SlabSlice { slab_idx: 0, offset: 256, size: 512 });
        assert_eq!(c, SlabSlice { slab_idx: 0, offset: 768, size: 256 });
        assert_eq!(d, SlabSlice { slab_idx: 0, offset: 1024, size: 1024 });
        assert_eq!(va_a, 1u64 << 32);

        // Defer-retire `b` (BLAS: requires tlas_epoch >= 3 && sub_seq >= 3) and `d` (non-TLAS: requires sub_seq >= 2).
        alloc.retire(b, 3, 3);
        alloc.retire(d, 0, 2);

        // Neither is released at (tlas=1, sub=1).
        assert_eq!(alloc.drain_retired(1, 1), 0);
        assert_eq!(alloc.slabs[0].free, Vec::<(u64, u64)>::new());
        assert_eq!(alloc.slabs[0].used, 2048);

        // At (tlas=1, sub=2), `d` (tail) is released without being blocked by `b` and folds back into `used`.
        assert_eq!(alloc.drain_retired(1, 2), 1);
        assert_eq!(alloc.slabs[0].used, 1024);
        assert!(alloc.slabs[0].free.is_empty());

        // At (tlas=2, sub=3), `b` is still retained because tlas_epoch < 3.
        assert_eq!(alloc.drain_retired(2, 3), 0);
        // At (tlas=3, sub=3), `b` is freed into the free list.
        assert_eq!(alloc.drain_retired(3, 3), 1);
        assert_eq!(alloc.slabs[0].free, vec![(256, 512)]);

        // Free `a` (adjacent left) and `c` (adjacent right + touches tail): all three coalesce and reset `used` to 0!
        alloc.free(a);
        assert_eq!(alloc.slabs[0].free, vec![(0, 768)]);
        alloc.free(c);
        assert_eq!(alloc.slabs[0].used, 0);
        assert!(alloc.slabs[0].free.is_empty());

        // Best-fit reuse: create two holes (512B and 256B), allocate 256B -> picks exact 256B hole.
        let (_, x0, _) = alloc.alloc_with(512, make).unwrap();
        let (_, _keep0, _) = alloc.alloc_with(256, make).unwrap();
        let (_, x1, _) = alloc.alloc_with(256, make).unwrap();
        let (_, _keep1, _) = alloc.alloc_with(256, make).unwrap();
        alloc.free(x0);
        alloc.free(x1);
        assert_eq!(alloc.slabs[0].free, vec![(0, 512), (768, 256)]);

        let (_, reused, va_reused) = alloc.alloc_with(256, make).unwrap();
        assert_eq!(reused, x1);
        assert_eq!(va_reused, (1u64 << 32) + 768);
        assert_eq!(alloc.slabs[0].free, vec![(0, 512)]);

        let (total, free) = alloc.stats();
        assert_eq!(total, AS_SLAB_SIZE);
        assert_eq!(free, (AS_SLAB_SIZE - 1280) + 512);
    }

    #[test]
    fn hash_dirty_tracker_coalesces_slots_and_overflows_to_bounding_range() {
        let mut t = HashDirtyTracker::full();
        assert!(t.is_dirty());
        assert_eq!(t.take_ranges(), vec![(0, HASH_CAPACITY - 1)]);
        assert!(!t.is_dirty());
        assert!(t.take_ranges().is_empty());

        // Nearby slots merge; distant slots stay separate.
        t.mark_slot(100);
        t.mark_slot(100 + HASH_DIRTY_MERGE_GAP);
        t.mark_slot(50_000);
        t.mark_slot(100); // duplicate
        assert_eq!(
            t.take_ranges(),
            vec![(100, 100 + HASH_DIRTY_MERGE_GAP), (50_000, 50_000)]
        );
        assert!(!t.is_dirty());

        // Exceeding MAX_DIRTY_HASH_SLOTS collapses into a single [min..=max] range.
        for i in 0..=MAX_DIRTY_HASH_SLOTS {
            t.mark_slot(1000 + i * 500);
        }
        assert_eq!(t.take_ranges(), vec![(1000, 1000 + MAX_DIRTY_HASH_SLOTS * 500)]);
        assert!(!t.is_dirty());
    }

    #[test]
    fn geometry_flags_set_no_duplicate_anyhit_on_non_opaque() {
        let opaque = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        let no_dup = D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION;

        assert_eq!(dynamic_geom_flags(0), no_dup);
        assert_eq!(dynamic_geom_flags(opaque.0 as u32), opaque);
        assert_eq!(dynamic_geom_flags((opaque | no_dup).0 as u32), opaque | no_dup);
        assert_eq!(static_geom_flags(false), no_dup);
    }

    #[test]
    fn should_compact_filters_trivial_and_invalid_savings() {
        // Zero, equal, or larger compacted sizes are rejected.
        assert!(!should_compact(65_536, 0));
        assert!(!should_compact(65_536, 65_536));
        assert!(!should_compact(65_536, 70_000));
        // Saving <= 4 KiB (even after 256B alignment) is rejected.
        assert!(!should_compact(16_384, 12_288));
        assert!(!should_compact(16_384, 12_033)); // aligns to 12_288 -> saves 4096
        // Small BLAS saving > 4 KiB and >= 10% compacts.
        assert!(should_compact(16_384, 8_192));
        assert!(should_compact(65_536, 32_768));
        // Large BLAS saving only 8 KiB (< 10% and < 64 KiB) is skipped.
        assert!(!should_compact(1 << 20, (1 << 20) - 8192));
        // Large BLAS saving >= 64 KiB compacts even if under 10%.
        assert!(should_compact(4 << 20, (4 << 20) - 128 * 1024));
    }

    #[test]
    fn compaction_retires_old_blas_slice_until_tlas_epoch_plus_two() {
        let mut alloc = SlabAllocator::<usize>::default();
        let make = |_slab_size: u64, slab_idx: usize| -> Option<(usize, u64)> {
            Some((slab_idx, (slab_idx as u64 + 1) << 32))
        };

        // Simulate uncompacted static BLAS (256 KiB) followed by async compaction to 128 KiB.
        let (_, uncompacted, _) = alloc.alloc_with(256 * 1024, make).unwrap();
        assert!(should_compact(uncompacted.size, 128 * 1024));
        let (_, compacted, _) = alloc.alloc_with(128 * 1024, make).unwrap();
        assert_eq!(compacted.offset, 256 * 1024);
        assert_eq!(compacted.size, 128 * 1024);

        // Old uncompacted slice retired at tlas_epoch=5, sub_seq=10 -> releases at (7, 12).
        alloc.retire(uncompacted, 7, 12);
        assert_eq!(alloc.drain_retired(6, 12), 0);
        assert_eq!(alloc.drain_retired(7, 11), 0);
        assert_eq!(alloc.drain_retired(7, 12), 1);

        // Reclaimed 256 KiB hole is immediately reused by the next static BLAS allocation.
        let (_, reused, _) = alloc.alloc_with(256 * 1024, make).unwrap();
        assert_eq!(reused, uncompacted);
    }
}
