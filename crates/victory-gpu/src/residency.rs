//! Host-memory admission for speculative texture work and reusable upload storage.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
static PRESSURED: AtomicBool = AtomicBool::new(false);
static HEADROOM: AtomicU64 = AtomicU64::new(u64::MAX);

fn memory_status() -> Option<MEMORYSTATUSEX> {
    // SAFETY: the API initializes the fields of a correctly sized structure.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

// The texture sweep may run for seconds inside a single submitted frame.
static NEXT_COMMIT_SAMPLE: AtomicU64 = AtomicU64::new(0);
static AVAILABLE_COMMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static RETRY_UPLOAD_AT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn allocation_error(error: &windows::core::Error) {
    if error.code() == windows::Win32::Foundation::E_OUTOFMEMORY {
        RETRY_UPLOAD_AT.fetch_max(crate::host::coarse_ms().saturating_add(500), Ordering::Relaxed);
    }
}

thread_local! {
    static STAGING_DEFERRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn note_staging_deferred() {
    STAGING_DEFERRED.set(true);
}

pub(crate) fn take_staging_deferred() -> bool {
    STAGING_DEFERRED.replace(false)
}

pub(crate) fn upload_refused() -> bool {
    crate::host::coarse_ms() < RETRY_UPLOAD_AT.load(Ordering::Relaxed)
}

fn sample_commit(now: u64) -> u64 {
    let next = NEXT_COMMIT_SAMPLE.load(Ordering::Relaxed);
    if now >= next
        && NEXT_COMMIT_SAMPLE
            .compare_exchange(next, now.saturating_add(32), Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        && let Some(status) = memory_status()
    {
        AVAILABLE_COMMIT.store(status.ullAvailPageFile, Ordering::Relaxed);
    }
    AVAILABLE_COMMIT.load(Ordering::Relaxed)
}

fn has_upload_reserve(commit: u64, bake: bool) -> bool {
    // Keep a protective floor against Rust/game OOM aborts without starving
    // bindless world textures or one-shot ped decoration bakes when steady-state
    // commit headroom sits in the 1-4 GiB range on 32 GiB hosts.
    commit >= if bake { 256 * MIB } else { 512 * MIB }
}

fn has_growth_reserve(commit: u64, bake: bool) -> bool {
    has_growth_reserve_for(commit, bake, low_end_ram())
}

fn has_growth_reserve_for(commit: u64, bake: bool, low_end: bool) -> bool {
    // Growing a fresh 64 MiB upload block for speculative bindless prefetch
    // requires more headroom than reusing slices already committed inside the
    // staging pool, and even more on <= 16 GiB RAM systems.
    let floor = if low_end {
        if bake { 384 * MIB } else { 1536 * MIB }
    } else if bake {
        256 * MIB
    } else {
        1024 * MIB
    };
    commit >= floor
}

/// Leave commit for game allocations and submissions already in flight while
/// still letting bindless textures and one-shot material bakes upload.
pub(crate) fn admit_texture_pass(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_upload_reserve(sample_commit(now), bake)
}

/// Whether the staging allocator may commit a brand-new OS upload buffer
/// (reusing existing free ranges in already-committed pool blocks stays open
/// whenever `admit_texture_pass` is true).
pub(crate) fn allow_staging_growth(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_growth_reserve(sample_commit(now), bake)
}

/// Optional manual override for low-end RAM/VRAM/CPU scaling (`VICTORY_LOW_END`).
/// Accepts `1`/`true`/`yes`/`on` (`Some(true)`) or `0`/`false`/`no`/`off` (`Some(false)`).
pub(crate) fn low_end_override() -> Option<bool> {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        let raw = std::env::var("VICTORY_LOW_END").ok()?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

/// True on hosts with `<= 16 GiB` of physical RAM (or when `VICTORY_LOW_END=1`).
pub(crate) fn low_end_ram() -> bool {
    if let Some(v) = low_end_override() {
        return v;
    }
    static LOW: OnceLock<bool> = OnceLock::new();
    *LOW.get_or_init(|| memory_status().is_some_and(|s| s.ullTotalPhys <= 16 * GIB))
}

/// True on adapters with `<= 12 GiB` DXGI local VRAM budget (or when `VICTORY_LOW_END=1`).
pub(crate) fn low_end_vram() -> bool {
    if let Some(v) = low_end_override() {
        return v;
    }
    static CACHED: AtomicU8 = AtomicU8::new(0);
    match CACHED.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    if let Some([budget, ..]) = crate::host::host().and_then(|h| h.video_memory()) {
        let low = budget <= 12 * GIB;
        CACHED.store(if low { 2 } else { 1 }, Ordering::Relaxed);
        return low;
    }
    false
}

/// True on hosts with `<= 8` logical CPUs (or when `VICTORY_LOW_END=1`).
pub(crate) fn low_end_cpu() -> bool {
    if let Some(v) = low_end_override() {
        return v;
    }
    static LOW: OnceLock<bool> = OnceLock::new();
    *LOW.get_or_init(|| std::thread::available_parallelism().is_ok_and(|n| n.get() <= 8))
}

pub(crate) fn limited_host_ram() -> bool {
    if low_end_override() == Some(true) {
        return true;
    }
    static LIMITED: OnceLock<bool> = OnceLock::new();
    *LIMITED.get_or_init(|| memory_status().is_some_and(|s| s.ullTotalPhys <= 32 * GIB))
}

fn next_pressure(was: bool, physical: u64, commit: u64) -> bool {
    next_pressure_for(was, physical, commit, low_end_ram())
}

fn next_pressure_for(was: bool, physical: u64, commit: u64, low_end: bool) -> bool {
    if low_end {
        if was {
            physical < 6 * GIB || commit < 7 * GIB
        } else {
            physical <= 4 * GIB || commit <= 5 * GIB
        }
    } else if was {
        physical < 5 * GIB || commit < 6 * GIB
    } else {
        physical <= 3 * GIB || commit <= 4 * GIB
    }
}

/// Called by the once-per-second submission sweep. A failed sample keeps the last good state.
pub(crate) fn refresh() {
    let Some(status) = memory_status() else { return };
    let (physical, commit) = (status.ullAvailPhys, status.ullAvailPageFile);
    HEADROOM.store(physical.min(commit), Ordering::Relaxed);
    let was = pressured();
    let pressure = next_pressure(was, physical, commit);
    if pressure != was {
        PRESSURED.store(pressure, Ordering::Relaxed);
        if pressure {
            tracing::warn!(
                pid = std::process::id(),
                frame = crate::frame::current(),
                physical_free_mib = physical >> 20,
                available_commit_mib = commit >> 20,
                buffer_mirror_span_mib = crate::shadow::live_span_bytes() >> 20,
                texture_staging_mib = crate::texture::live_staging_bytes() >> 20,
                spare_scratch_mib = crate::host::ALIVE[4].load(Ordering::Relaxed) >> 20,
                game_filled_gpu_texture_mib = crate::resource::KINDS[1].load(Ordering::Relaxed) >> 20,
                "host memory pressure: limiting texture prefetch and idle upload storage"
            );
        } else {
            tracing::info!(physical_free_mib = physical >> 20, available_commit_mib = commit >> 20, "host memory pressure cleared");
        }
    }
}

pub(crate) fn headroom() -> Option<u64> {
    let bytes = HEADROOM.load(Ordering::Relaxed);
    (bytes != u64::MAX).then_some(bytes)
}

pub(crate) fn pressured() -> bool {
    PRESSURED.load(Ordering::Relaxed)
}

/// True when OS commit headroom is critically low (< 1.5 GiB) or an allocation
/// just failed with `E_OUTOFMEMORY`, requiring emergency throttling.
pub(crate) fn critical_commit() -> bool {
    let now = crate::host::coarse_ms();
    now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) || sample_commit(now) < 1536 * MIB
}

/// A soft per-frame admission limit. A single texture can cross it. The
/// queue retains unserved slots so they can be retried on later frames.
pub(crate) fn prefetch_limit(bytes: u64, pressure: bool, critical: bool, limited: bool, staging: u64) -> u64 {
    prefetch_limit_for(bytes, pressure, critical, limited, low_end_ram(), staging)
}

fn prefetch_limit_for(bytes: u64, pressure: bool, critical: bool, limited: bool, low_end: bool, staging: u64) -> u64 {
    if critical {
        bytes.min(if low_end { 8 << 20 } else { 16 << 20 })
    } else if low_end {
        if pressure || staging >= 256 * MIB {
            bytes.min(32 << 20)
        } else {
            bytes.min(64 << 20)
        }
    } else if pressure || (limited && staging >= GIB) {
        bytes.min(64 << 20)
    } else {
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bake_and_lookahead_leave_room_for_live_game_allocations() {
        for (commit, bake, lookahead, growth) in [
            (128 * MIB, false, false, false),
            (256 * MIB, true, false, false),
            (512 * MIB, true, true, false),
            (768 * MIB, true, true, false),
            (1024 * MIB, true, true, true),
        ] {
            assert_eq!(has_upload_reserve(commit, true), bake);
            assert_eq!(has_upload_reserve(commit, false), lookahead);
            assert_eq!(has_growth_reserve_for(commit, false, false), growth);
        }
        assert!(!has_growth_reserve_for(1024 * MIB, false, true));
        assert!(has_growth_reserve_for(1536 * MIB, false, true));
        assert!(!has_growth_reserve_for(256 * MIB, true, true));
        assert!(has_growth_reserve_for(384 * MIB, true, true));
    }

    #[test]
    fn either_constraint_enters_pressure_and_both_must_recover() {
        assert!(next_pressure_for(false, 12 * GIB, 4 * GIB, false));
        assert!(next_pressure_for(false, 3 * GIB, 16 * GIB, false));
        assert!(next_pressure_for(true, 4 * GIB, 7 * GIB, false));
        assert!(next_pressure_for(true, 6 * GIB, 5 * GIB, false));
        assert!(!next_pressure_for(true, 5 * GIB, 6 * GIB, false));
        assert!(!next_pressure_for(false, 4 * GIB, 5 * GIB, false));
        // Low-end RAM enters pressure earlier and requires higher recovery headroom.
        assert!(next_pressure_for(false, 4 * GIB, 8 * GIB, true));
        assert!(next_pressure_for(false, 8 * GIB, 5 * GIB, true));
        assert!(next_pressure_for(true, 5 * GIB, 7 * GIB, true));
        assert!(!next_pressure_for(true, 6 * GIB, 7 * GIB, true));
    }

    #[test]
    fn prefetch_respects_existing_budgets_and_resumes_after_recovery() {
        assert_eq!(prefetch_limit_for(512 << 20, false, false, true, false, GIB), 64 << 20);
        assert_eq!(prefetch_limit_for(512 << 20, false, false, true, false, GIB - 1), 512 << 20);
        assert_eq!(prefetch_limit_for(512 << 20, false, false, false, false, 2 * GIB), 512 << 20);
        assert_eq!(prefetch_limit_for(0, true, true, false, false, 0), 0);
        assert_eq!(prefetch_limit_for(8 << 20, true, true, false, false, 0), 8 << 20);
        assert_eq!(prefetch_limit_for(512 << 20, true, false, false, false, 0), 64 << 20);
        assert_eq!(prefetch_limit_for(512 << 20, true, true, false, false, 0), 16 << 20);
        assert_eq!(prefetch_limit_for(192 << 20, false, false, false, false, 0), 192 << 20);
        // Low-end RAM tightens per-frame prefetch limits.
        assert_eq!(prefetch_limit_for(512 << 20, false, false, true, true, 0), 64 << 20);
        assert_eq!(prefetch_limit_for(512 << 20, false, false, true, true, 256 * MIB), 32 << 20);
        assert_eq!(prefetch_limit_for(512 << 20, true, false, true, true, 0), 32 << 20);
        assert_eq!(prefetch_limit_for(512 << 20, true, true, true, true, 0), 8 << 20);
    }
}
