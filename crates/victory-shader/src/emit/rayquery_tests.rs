//! DXIL SM 6.6 `RayQuery` validation and hardware GPU self-tests.
//!
//! Covers both bound `RaytracingAccelerationStructure` root SRVs (`t0, space1`)
//! and SM 6.6 `ResourceDescriptorHeap` SRVs, opaque hit/miss state queries, and
//! non-opaque `Proceed()` / `CommitNonOpaqueTriangleHit()` traversal loops.

use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::bitcode::{Attr, Bin, Cast, Term, V, pred};
use super::host::Host;
use super::shader::{ROOT_CBV_SPACE, Shader, rq_op};
use super::validator::Validator;
use crate::dxbc::{Container, FourCC};

const D3D_SHADER_FEATURE_RAYTRACING_TIER_1_1: u64 = 0x10_0000;
const D3D_SHADER_FEATURE_RESOURCE_DESCRIPTOR_HEAP_INDEXING: u64 = 0x200_0000;

/// Dwords written per lane by the opaque hit/miss shader (22 used, padded to 24).
const OPAQUE_LANE_STRIDE_DWORDS: u32 = 24;
/// Dwords written per lane by the non-opaque proceed/commit shader.
const PROCEED_LANE_STRIDE_DWORDS: u32 = 4;
/// Dwords written per lane by the two-pass tMax pre-shrink shader.
const TWO_PASS_LANE_STRIDE_DWORDS: u32 = 6;

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
}

fn sfi0_flags(dxil: &[u8]) -> u64 {
    let c = Container::parse(dxil).expect("valid DXBC container");
    let part = c.part(FourCC::SFI0).expect("SFI0 part");
    u64::from_le_bytes(part.data[..8].try_into().expect("8-byte SFI0"))
}

fn validate_dxil(label: &str, dxil: &[u8]) {
    let mut validator = Validator::load(Validator::default_path().as_deref())
        .unwrap_or_else(|e| panic!("{label}: failed to load dxil.dll validator: {e}"));
    let mut copy = dxil.to_vec();
    validator
        .validate(&mut copy)
        .unwrap_or_else(|e| panic!("{label}: DXIL validation failed: {e}"));
}

fn store_i32(s: &mut Shader, b: usize, uav: V, byte_off: V, val: V) {
    let undef = s.m.undef(s.t.i32);
    let mask = s.m.i8(1);
    let align = s.m.i32(4);
    let void = s.t.void;
    s.op(
        b,
        "rawBufferStore.i32",
        Attr::NoUnwind,
        void,
        140,
        &[uav, byte_off, undef, val, undef, undef, undef, mask, align],
    );
}

fn store_f32(s: &mut Shader, b: usize, uav: V, byte_off: V, val: V) {
    let bits = s.m.cast(b, Cast::Bitcast, val, s.t.i32);
    store_i32(s, b, uav, byte_off, bits);
}

fn byte_offset(s: &mut Shader, b: usize, base: V, dword_index: u32) -> V {
    if dword_index == 0 {
        return base;
    }
    let off = s.m.i32(dword_index * 4);
    s.m.bin(b, Bin::Add, base, off)
}

/// `[numthreads(2, 1, 1)]` compute shader tracing lane 0 (hitting ray) and
/// lane 1 (missing ray) against either a bound root SRV (`t0, space1`) or
/// `ResourceDescriptorHeap[0]`, storing all committed hit fields to `u0, space1`.
fn emit_opaque_hit_miss_cs(use_heap: bool) -> Vec<u8> {
    let mut s = Shader::new(false);
    s.threads = Some([2, 1, 1]);
    let entry = s.entry;
    let loop_head = s.m.new_block();
    let loop_body = s.m.new_block();
    let done = s.m.new_block();

    let uav = s.root_uav_handle(0);
    let tlas = if use_heap {
        let idx = s.m.i32(0);
        s.heap_handle(entry, idx, false, false, (16, 0))
    } else {
        s.rtas_root_srv_handle(entry, 0)
    };

    let i32t = s.t.i32;
    let tid = s.op(
        entry,
        "flattenedThreadIdInGroup.i32",
        Attr::ReadNone,
        i32t,
        96,
        &[],
    );
    let zero_i = s.m.i32(0);
    let is_lane0 = s.m.cmp(entry, pred::EQ, tid, zero_i);

    let hit_ox = s.m.f32(11.0f32.to_bits());
    let hit_oy = s.m.f32(22.0f32.to_bits());
    let miss_o = s.m.f32(100.0f32.to_bits());
    let ox = s.m.select(entry, is_lane0, hit_ox, miss_o);
    let oy = s.m.select(entry, is_lane0, hit_oy, miss_o);
    let oz = s.m.f32(1.0f32.to_bits());
    let t_min = s.m.f32(0.001f32.to_bits());
    let dx = s.m.f32(0.0f32.to_bits());
    let dy = s.m.f32(0.0f32.to_bits());
    let dz = s.m.f32(1.0f32.to_bits());
    let t_max = s.m.f32(100.0f32.to_bits());

    let rq = s.allocate_ray_query(entry, 0);
    let mask = s.m.i32(0xFF);
    s.ray_query_trace_inline(
        entry,
        rq,
        tlas,
        zero_i,
        mask,
        [ox, oy, oz],
        t_min,
        [dx, dy, dz],
        t_max,
    );
    s.m.terminate(entry, Term::Br(loop_head));

    let more = s.ray_query_proceed(loop_head, rq);
    s.m.terminate(loop_head, Term::CondBr(more, loop_body, done));

    // Opaque-only scene: abort if any non-opaque candidate unexpectedly appears.
    s.ray_query_abort(loop_body, rq);
    s.m.terminate(loop_body, Term::Br(loop_head));

    let stride = s.m.i32(OPAQUE_LANE_STRIDE_DWORDS * 4);
    let base = s.m.bin(done, Bin::Mul, tid, stride);

    let status = s.ray_query_state_i32(done, rq_op::COMMITTED_STATUS, rq);
    let ray_t = s.ray_query_state_f32(done, rq_op::COMMITTED_RAY_T, rq);
    let bary_u = s.ray_query_state_vec_f32(done, rq_op::COMMITTED_TRIANGLE_BARYCENTRICS, rq, 0);
    let bary_v = s.ray_query_state_vec_f32(done, rq_op::COMMITTED_TRIANGLE_BARYCENTRICS, rq, 1);
    let inst_id = s.ray_query_state_i32(done, rq_op::COMMITTED_INSTANCE_ID, rq);
    let inst_idx = s.ray_query_state_i32(done, rq_op::COMMITTED_INSTANCE_INDEX, rq);
    let geom_idx = s.ray_query_state_i32(done, rq_op::COMMITTED_GEOMETRY_INDEX, rq);
    let prim_idx = s.ray_query_state_i32(done, rq_op::COMMITTED_PRIMITIVE_INDEX, rq);
    let contrib = s.ray_query_state_i32(done, rq_op::COMMITTED_INSTANCE_CONTRIBUTION, rq);
    let front_i1 = s.ray_query_state_i1(done, rq_op::COMMITTED_TRIANGLE_FRONT_FACE, rq);
    let front = s.m.cast(done, Cast::ZExt, front_i1, i32t);

    let mut mat = [zero_i; 12];
    for row in 0..3u32 {
        for col in 0..4u8 {
            mat[(row * 4 + u32::from(col)) as usize] = s.ray_query_state_mat_f32(
                done,
                rq_op::COMMITTED_OBJECT_TO_WORLD_3X4,
                rq,
                row,
                col,
            );
        }
    }

    let o0 = byte_offset(&mut s, done, base, 0);
    store_i32(&mut s, done, uav, o0, status);
    let o1 = byte_offset(&mut s, done, base, 1);
    store_f32(&mut s, done, uav, o1, ray_t);
    let o2 = byte_offset(&mut s, done, base, 2);
    store_f32(&mut s, done, uav, o2, bary_u);
    let o3 = byte_offset(&mut s, done, base, 3);
    store_f32(&mut s, done, uav, o3, bary_v);
    let o4 = byte_offset(&mut s, done, base, 4);
    store_i32(&mut s, done, uav, o4, inst_id);
    let o5 = byte_offset(&mut s, done, base, 5);
    store_i32(&mut s, done, uav, o5, inst_idx);
    let o6 = byte_offset(&mut s, done, base, 6);
    store_i32(&mut s, done, uav, o6, geom_idx);
    let o7 = byte_offset(&mut s, done, base, 7);
    store_i32(&mut s, done, uav, o7, prim_idx);
    let o8 = byte_offset(&mut s, done, base, 8);
    store_i32(&mut s, done, uav, o8, contrib);
    let o9 = byte_offset(&mut s, done, base, 9);
    store_i32(&mut s, done, uav, o9, front);
    for (k, &m) in mat.iter().enumerate() {
        let ok = byte_offset(&mut s, done, base, 10 + k as u32);
        store_f32(&mut s, done, uav, ok, m);
    }

    s.m.terminate(done, Term::Ret);
    s.finish()
}

/// `[numthreads(2, 1, 1)]` compute shader running a `Proceed()` loop:
/// lane 0 rejects non-opaque triangle candidates; lane 1 commits them via
/// `CommitNonOpaqueTriangleHit`.
fn emit_non_opaque_proceed_cs() -> Vec<u8> {
    let mut s = Shader::new(false);
    s.threads = Some([2, 1, 1]);
    let entry = s.entry;
    let loop_head = s.m.new_block();
    let loop_body = s.m.new_block();
    let do_commit = s.m.new_block();
    let loop_latch = s.m.new_block();
    let done = s.m.new_block();

    let uav = s.root_uav_handle(0);
    let tlas = s.rtas_root_srv_handle(entry, 0);

    let i32t = s.t.i32;
    let tid = s.op(
        entry,
        "flattenedThreadIdInGroup.i32",
        Attr::ReadNone,
        i32t,
        96,
        &[],
    );
    let zero_i = s.m.i32(0);
    let is_lane1 = s.m.cmp(entry, pred::NE, tid, zero_i);

    let zero_f = s.m.f32(0.0f32.to_bits());
    let one_f = s.m.f32(1.0f32.to_bits());
    let t_min = s.m.f32(0.001f32.to_bits());
    let t_max = s.m.f32(100.0f32.to_bits());

    let rq = s.allocate_ray_query(entry, 0);
    let mask = s.m.i32(0xFF);
    s.ray_query_trace_inline(
        entry,
        rq,
        tlas,
        zero_i,
        mask,
        [zero_f, zero_f, zero_f],
        t_min,
        [zero_f, zero_f, one_f],
        t_max,
    );
    s.m.terminate(entry, Term::Br(loop_head));

    let more = s.ray_query_proceed(loop_head, rq);
    s.m.terminate(loop_head, Term::CondBr(more, loop_body, done));

    let ctype = s.ray_query_state_i32(loop_body, rq_op::CANDIDATE_TYPE, rq);
    let is_tri = s.m.cmp(loop_body, pred::EQ, ctype, zero_i);
    let should_commit = s.m.bin(loop_body, Bin::And, is_tri, is_lane1);
    s.m.terminate(loop_body, Term::CondBr(should_commit, do_commit, loop_latch));

    s.ray_query_commit_non_opaque_triangle_hit(do_commit, rq);
    s.m.terminate(do_commit, Term::Br(loop_latch));

    s.m.terminate(loop_latch, Term::Br(loop_head));

    let stride = s.m.i32(PROCEED_LANE_STRIDE_DWORDS * 4);
    let base = s.m.bin(done, Bin::Mul, tid, stride);

    let status = s.ray_query_state_i32(done, rq_op::COMMITTED_STATUS, rq);
    let ray_t = s.ray_query_state_f32(done, rq_op::COMMITTED_RAY_T, rq);
    let geom_idx = s.ray_query_state_i32(done, rq_op::COMMITTED_GEOMETRY_INDEX, rq);
    let prim_idx = s.ray_query_state_i32(done, rq_op::COMMITTED_PRIMITIVE_INDEX, rq);

    let o0 = byte_offset(&mut s, done, base, 0);
    store_i32(&mut s, done, uav, o0, status);
    let o1 = byte_offset(&mut s, done, base, 1);
    store_f32(&mut s, done, uav, o1, ray_t);
    let o2 = byte_offset(&mut s, done, base, 2);
    store_i32(&mut s, done, uav, o2, geom_idx);
    let o3 = byte_offset(&mut s, done, base, 3);
    store_i32(&mut s, done, uav, o3, prim_idx);

    s.m.terminate(done, Term::Ret);
    s.finish()
}

/// `[numthreads(3, 1, 1)]` compute shader exercising two-pass `tMax` pre-shrink:
/// Pass 1 (`CULL_NON_OPAQUE | SKIP_PROCEDURAL_PRIMITIVES`) finds the closest opaque
/// hit and tightens `tMax`, then Pass 2 (`CULL_OPAQUE | SKIP_PROCEDURAL_PRIMITIVES`)
/// runs `AnyHit` (accepting only on lane 1, rejecting on lane 2) and merges
/// committed hit metadata from Pass 2 (when hit) else Pass 1.
fn emit_two_pass_anyhit_cs() -> Vec<u8> {
    let mut s = Shader::new(false);
    s.threads = Some([3, 1, 1]);
    let entry = s.entry;
    let p1_proceed = s.m.new_block();
    let p2_start = s.m.new_block();
    let p2_proceed = s.m.new_block();
    let anyhit_body = s.m.new_block();
    let do_commit = s.m.new_block();
    let latch = s.m.new_block();
    let after = s.m.new_block();
    let from_q2 = s.m.new_block();
    let check_q1 = s.m.new_block();
    let from_q1 = s.m.new_block();
    let miss_blk = s.m.new_block();
    let done = s.m.new_block();

    let uav = s.root_uav_handle(0);
    let tlas = s.rtas_root_srv_handle(entry, 0);

    let i32t = s.t.i32;
    let tid = s.op(
        entry,
        "flattenedThreadIdInGroup.i32",
        Attr::ReadNone,
        i32t,
        96,
        &[],
    );
    let zero_i = s.m.i32(0);
    let one_i = s.m.i32(1);
    let is_lane0 = s.m.cmp(entry, pred::EQ, tid, zero_i);
    let is_lane1 = s.m.cmp(entry, pred::EQ, tid, one_i);

    let ox0 = s.m.f32(0.0f32.to_bits());
    let ox12 = s.m.f32(10.0f32.to_bits());
    let ox = s.m.select(entry, is_lane0, ox0, ox12);
    let zero_f = s.m.f32(0.0f32.to_bits());
    let one_f = s.m.f32(1.0f32.to_bits());
    let t_min = s.m.f32(0.001f32.to_bits());
    let t_max = s.m.f32(100.0f32.to_bits());

    // Pass 1: CULL_NON_OPAQUE (0x80) | SKIP_PROCEDURAL_PRIMITIVES (0x200). Never FORCE_OPAQUE!
    let q1 = s.allocate_ray_query(entry, 0x280);
    // Pass 2: CULL_OPAQUE (0x40) | SKIP_PROCEDURAL_PRIMITIVES (0x200).
    let q2 = s.allocate_ray_query(entry, 0x240);

    let mask = s.m.i32(0xFF);
    s.ray_query_trace_inline(
        entry,
        q1,
        tlas,
        zero_i,
        mask,
        [ox, zero_f, zero_f],
        t_min,
        [zero_f, zero_f, one_f],
        t_max,
    );
    s.m.terminate(entry, Term::Br(p1_proceed));

    let p1_more = s.ray_query_proceed(p1_proceed, q1);
    s.m.terminate(p1_proceed, Term::CondBr(p1_more, p1_proceed, p2_start));

    let p1_status = s.ray_query_state_i32(p2_start, rq_op::COMMITTED_STATUS, q1);
    let q1_hit = s.m.cmp(p2_start, pred::EQ, p1_status, one_i);
    let p1_t = s.ray_query_state_f32(p2_start, rq_op::COMMITTED_RAY_T, q1);
    let t_clamp = s.m.select(p2_start, q1_hit, p1_t, t_max);

    s.ray_query_trace_inline(
        p2_start,
        q2,
        tlas,
        zero_i,
        mask,
        [ox, zero_f, zero_f],
        t_min,
        [zero_f, zero_f, one_f],
        t_clamp,
    );
    s.m.terminate(p2_start, Term::Br(p2_proceed));

    let ah_count = s.m.phi(p2_proceed, i32t);
    let p2_more = s.ray_query_proceed(p2_proceed, q2);
    s.m.terminate(p2_proceed, Term::CondBr(p2_more, anyhit_body, after));

    let next_count = s.m.bin(anyhit_body, Bin::Add, ah_count, one_i);
    s.m.terminate(anyhit_body, Term::CondBr(is_lane1, do_commit, latch));

    s.ray_query_commit_non_opaque_triangle_hit(do_commit, q2);
    s.m.terminate(do_commit, Term::Br(latch));

    s.m.terminate(latch, Term::Br(p2_proceed));
    s.m.set_phi(ah_count, vec![(zero_i, p2_start), (next_count, latch)]);

    let p2_status = s.ray_query_state_i32(after, rq_op::COMMITTED_STATUS, q2);
    let q2_hit = s.m.cmp(after, pred::EQ, p2_status, one_i);
    s.m.terminate(after, Term::CondBr(q2_hit, from_q2, check_q1));
    s.m.terminate(check_q1, Term::CondBr(q1_hit, from_q1, miss_blk));

    let q2_t = s.ray_query_state_f32(from_q2, rq_op::COMMITTED_RAY_T, q2);
    let q2_t_i = s.m.cast(from_q2, Cast::Bitcast, q2_t, i32t);
    let q2_geom = s.ray_query_state_i32(from_q2, rq_op::COMMITTED_GEOMETRY_INDEX, q2);
    let q2_prim = s.ray_query_state_i32(from_q2, rq_op::COMMITTED_PRIMITIVE_INDEX, q2);
    let q2_id = s.ray_query_state_i32(from_q2, rq_op::COMMITTED_INSTANCE_ID, q2);
    s.m.terminate(from_q2, Term::Br(done));

    let q1_t_i = s.m.cast(from_q1, Cast::Bitcast, p1_t, i32t);
    let q1_geom = s.ray_query_state_i32(from_q1, rq_op::COMMITTED_GEOMETRY_INDEX, q1);
    let q1_prim = s.ray_query_state_i32(from_q1, rq_op::COMMITTED_PRIMITIVE_INDEX, q1);
    let q1_id = s.ray_query_state_i32(from_q1, rq_op::COMMITTED_INSTANCE_ID, q1);
    s.m.terminate(from_q1, Term::Br(done));

    let tmax_i = s.m.i32(100.0f32.to_bits());
    s.m.terminate(miss_blk, Term::Br(done));

    let status = s.m.phi(done, i32t);
    s.m.set_phi(status, vec![(one_i, from_q2), (one_i, from_q1), (zero_i, miss_blk)]);
    let ray_t_i = s.m.phi(done, i32t);
    s.m.set_phi(ray_t_i, vec![(q2_t_i, from_q2), (q1_t_i, from_q1), (tmax_i, miss_blk)]);
    let geom_idx = s.m.phi(done, i32t);
    s.m.set_phi(geom_idx, vec![(q2_geom, from_q2), (q1_geom, from_q1), (zero_i, miss_blk)]);
    let prim_idx = s.m.phi(done, i32t);
    s.m.set_phi(prim_idx, vec![(q2_prim, from_q2), (q1_prim, from_q1), (zero_i, miss_blk)]);
    let inst_id = s.m.phi(done, i32t);
    s.m.set_phi(inst_id, vec![(q2_id, from_q2), (q1_id, from_q1), (zero_i, miss_blk)]);

    let stride = s.m.i32(TWO_PASS_LANE_STRIDE_DWORDS * 4);
    let base = s.m.bin(done, Bin::Mul, tid, stride);
    for (k, val) in [status, ray_t_i, geom_idx, prim_idx, inst_id, ah_count]
        .into_iter()
        .enumerate()
    {
        let off = byte_offset(&mut s, done, base, k as u32);
        store_i32(&mut s, done, uav, off, val);
    }

    s.m.terminate(done, Term::Ret);
    s.finish()
}

struct TriangleGeom<'a> {
    vertices: &'a [[f32; 3]],
    flags: D3D12_RAYTRACING_GEOMETRY_FLAGS,
}

struct InstanceSpec {
    transform: [f32; 12],
    instance_id: u32,
    instance_mask: u8,
    contrib: u32,
    flags: u8,
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

fn transition_barrier(
    r: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn has_raytracing_tier_1_1(host: &Host) -> bool {
    let mut opts5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    // SAFETY: CheckFeatureSupport writes into opts5 of the given byte size.
    unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5,
            (&raw mut opts5).cast(),
            size_of_val(&opts5) as u32,
        )
    }
    .is_ok()
        && opts5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0
}

/// Builds BLAS + 1-instance TLAS on `host`, dispatches `cs_dxil` (`(1, 1, 1)`),
/// and reads back `out_dwords` from `u0, space1`.
fn run_rayquery_gpu(
    host: &Host,
    cs_dxil: &[u8],
    use_heap: bool,
    geoms: &[TriangleGeom<'_>],
    inst: &InstanceSpec,
    out_dwords: u32,
) -> Result<Vec<u32>, String> {
    let dev = &host.device;
    let dev5: ID3D12Device5 = dev.cast().map_err(err)?;

    let mut params = Vec::new();
    if !use_heap {
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: 0,
                    RegisterSpace: ROOT_CBV_SPACE,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }
    params.push(D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: 0,
                RegisterSpace: ROOT_CBV_SPACE,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    let rs_flags = if use_heap {
        D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    };
    let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: rs_flags,
            },
        },
    };

    // SAFETY: D3D12 calls with live descriptors; all GPU resources stay in
    // `keep` until the fence confirms GPU completion.
    unsafe {
        let mut blob = None;
        let mut error = None;
        D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, Some(&mut error)).map_err(err)?;
        let blob: ID3DBlob = blob.ok_or("no root signature blob")?;
        let rs_bytes = std::slice::from_raw_parts(
            blob.GetBufferPointer().cast::<u8>(),
            blob.GetBufferSize(),
        );
        let root_sig: ID3D12RootSignature = dev.CreateRootSignature(0, rs_bytes).map_err(err)?;
        let pso = host.compute_pipeline_with(&root_sig, cs_dxil)?;

        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(err)?;
        let alloc: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(err)?;
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
            .map_err(err)?;
        let list4: ID3D12GraphicsCommandList4 = list.cast().map_err(err)?;

        let buf_desc = |size: u64, flags: D3D12_RESOURCE_FLAGS| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let make = |heap: D3D12_HEAP_TYPE,
                    desc: &D3D12_RESOURCE_DESC,
                    state: D3D12_RESOURCE_STATES| {
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                desc,
                state,
                None,
                &mut r,
            )
            .map_err(err)?;
            r.ok_or_else(|| "no resource".to_string())
        };
        let upload_bytes = |bytes: &[u8]| -> Result<ID3D12Resource, String> {
            let size = (bytes.len() as u64).max(256).next_multiple_of(256);
            let b = make(
                D3D12_HEAP_TYPE_UPLOAD,
                &buf_desc(size, D3D12_RESOURCE_FLAG_NONE),
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut p = std::ptr::null_mut();
            b.Map(0, None, Some(&mut p)).map_err(err)?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.cast::<u8>(), bytes.len());
            b.Unmap(0, None);
            Ok(b)
        };
        let as_align = u64::from(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);

        let mut keep: Vec<ID3D12Resource> = Vec::new();
        let mut geom_descs = Vec::with_capacity(geoms.len());
        for g in geoms {
            let bytes = std::slice::from_raw_parts(
                g.vertices.as_ptr().cast::<u8>(),
                size_of_val(g.vertices),
            );
            let vb = upload_bytes(bytes)?;
            geom_descs.push(D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: g.flags,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_UNKNOWN,
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 0,
                        VertexCount: g.vertices.len() as u32,
                        IndexBuffer: 0,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: vb.GetGPUVirtualAddress(),
                            StrideInBytes: 12,
                        },
                    },
                },
            });
            keep.push(vb);
        }

        let blas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: geom_descs.len() as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: geom_descs.as_ptr(),
            },
        };
        let mut blas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev5.GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &mut blas_info);

        let blas_size = blas_info.ResultDataMaxSizeInBytes.max(as_align).next_multiple_of(as_align);
        let blas_scratch_size = blas_info.ScratchDataSizeInBytes.max(as_align).next_multiple_of(as_align);
        let blas = make(
            D3D12_HEAP_TYPE_DEFAULT,
            &buf_desc(blas_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        )?;
        let blas_scratch = make(
            D3D12_HEAP_TYPE_DEFAULT,
            &buf_desc(blas_scratch_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;

        let instance_desc = D3D12_RAYTRACING_INSTANCE_DESC {
            Transform: inst.transform,
            _bitfield1: (inst.instance_id & 0x00FF_FFFF) | (u32::from(inst.instance_mask) << 24),
            _bitfield2: (inst.contrib & 0x00FF_FFFF) | (u32::from(inst.flags) << 24),
            AccelerationStructure: blas.GetGPUVirtualAddress(),
        };
        let inst_bytes = std::slice::from_raw_parts(
            (&raw const instance_desc).cast::<u8>(),
            size_of::<D3D12_RAYTRACING_INSTANCE_DESC>(),
        );
        let inst_buf = upload_bytes(inst_bytes)?;

        let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: inst_buf.GetGPUVirtualAddress(),
            },
        };
        let mut tlas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut tlas_info);

        let tlas_size = tlas_info.ResultDataMaxSizeInBytes.max(as_align).next_multiple_of(as_align);
        let tlas_scratch_size = tlas_info.ScratchDataSizeInBytes.max(as_align).next_multiple_of(as_align);
        let tlas = make(
            D3D12_HEAP_TYPE_DEFAULT,
            &buf_desc(tlas_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        )?;
        let tlas_scratch = make(
            D3D12_HEAP_TYPE_DEFAULT,
            &buf_desc(tlas_scratch_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;

        let blas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: blas.GetGPUVirtualAddress(),
            Inputs: blas_inputs,
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: blas_scratch.GetGPUVirtualAddress(),
        };
        list4.BuildRaytracingAccelerationStructure(&blas_desc, None);
        list.ResourceBarrier(&[uav_barrier()]);

        let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: tlas.GetGPUVirtualAddress(),
            Inputs: tlas_inputs,
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: tlas_scratch.GetGPUVirtualAddress(),
        };
        list4.BuildRaytracingAccelerationStructure(&tlas_desc, None);
        list.ResourceBarrier(&[uav_barrier()]);

        let out_bytes = u64::from(out_dwords) * 4;
        let out_buf = make(
            D3D12_HEAP_TYPE_DEFAULT,
            &buf_desc(out_bytes.max(256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        let readback = make(
            D3D12_HEAP_TYPE_READBACK,
            &buf_desc(out_bytes.max(256), D3D12_RESOURCE_FLAG_NONE),
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        let srv_heap = if use_heap {
            let heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(err)?;
            let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV {
                        Location: tlas.GetGPUVirtualAddress(),
                    },
                },
            };
            dev.CreateShaderResourceView(
                None::<&ID3D12Resource>,
                Some(&srv_desc),
                heap.GetCPUDescriptorHandleForHeapStart(),
            );
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            Some(heap)
        } else {
            None
        };

        list.SetComputeRootSignature(&root_sig);
        if use_heap {
            list.SetComputeRootUnorderedAccessView(0, out_buf.GetGPUVirtualAddress());
        } else {
            list.SetComputeRootShaderResourceView(0, tlas.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(1, out_buf.GetGPUVirtualAddress());
        }
        list.Dispatch(1, 1, 1);

        list.ResourceBarrier(&[transition_barrier(
            &out_buf,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyBufferRegion(&readback, 0, &out_buf, 0, out_bytes);
        list.Close().map_err(err)?;

        queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        queue.Signal(&fence, 1).map_err(err)?;
        while fence.GetCompletedValue() < 1 {
            if let Err(e) = dev.GetDeviceRemovedReason() {
                return Err(format!("device removed: {}", err(e)));
            }
            std::thread::yield_now();
        }
        if let Err(e) = dev.GetDeviceRemovedReason() {
            return Err(format!("device removed: {}", err(e)));
        }

        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).map_err(err)?;
        let words = std::slice::from_raw_parts(p.cast::<u32>(), out_dwords as usize).to_vec();
        readback.Unmap(0, None);

        drop(srv_heap);
        drop(keep);
        drop((blas, blas_scratch, inst_buf, tlas, tlas_scratch, out_buf));
        Ok(words)
    }
}

#[test]
fn dxil_validates_rayquery_bound_and_heap() {
    let bound = emit_opaque_hit_miss_cs(false);
    let heap = emit_opaque_hit_miss_cs(true);
    let proceed = emit_non_opaque_proceed_cs();
    let two_pass = emit_two_pass_anyhit_cs();

    assert_ne!(
        sfi0_flags(&bound) & D3D_SHADER_FEATURE_RAYTRACING_TIER_1_1,
        0,
        "bound RayQuery shader must set SFI0 RAYTRACING_TIER_1_1"
    );
    assert_eq!(
        sfi0_flags(&bound) & D3D_SHADER_FEATURE_RESOURCE_DESCRIPTOR_HEAP_INDEXING,
        0,
        "bound-only RayQuery shader must not set RESOURCE_HEAP"
    );
    assert_eq!(
        sfi0_flags(&heap)
            & (D3D_SHADER_FEATURE_RAYTRACING_TIER_1_1
                | D3D_SHADER_FEATURE_RESOURCE_DESCRIPTOR_HEAP_INDEXING),
        D3D_SHADER_FEATURE_RAYTRACING_TIER_1_1
            | D3D_SHADER_FEATURE_RESOURCE_DESCRIPTOR_HEAP_INDEXING,
        "heap RayQuery shader must set both RAYTRACING_TIER_1_1 and RESOURCE_HEAP"
    );

    validate_dxil("bound RayQuery CS", &bound);
    validate_dxil("heap RayQuery CS", &heap);
    validate_dxil("non-opaque Proceed RayQuery CS", &proceed);
    validate_dxil("two-pass AnyHit RayQuery CS", &two_pass);
}

#[test]
fn gpu_rayquery_opaque_hit_and_miss() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    if !has_raytracing_tier_1_1(&host) {
        eprintln!("hardware device lacks DXR Tier 1.1; skipped");
        return;
    }

    let tri = [[0.0f32, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
    let geoms = [TriangleGeom {
        vertices: &tri,
        flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
    }];
    let transform = [
        1.0f32, 0.0, 0.0, 10.0,
        0.0, 1.0, 0.0, 20.0,
        0.0, 0.0, 1.0, 5.0,
    ];
    let inst = InstanceSpec {
        transform,
        instance_id: 0x12345,
        instance_mask: 0xFF,
        contrib: 7,
        flags: 0,
    };

    for (label, use_heap) in [("bound", false), ("heap", true)] {
        let dxil = emit_opaque_hit_miss_cs(use_heap);
        let out = run_rayquery_gpu(
            &host,
            &dxil,
            use_heap,
            &geoms,
            &inst,
            2 * OPAQUE_LANE_STRIDE_DWORDS,
        )
        .unwrap_or_else(|e| panic!("{label}: run_rayquery_gpu failed: {e}"));

        // Lane 0: hitting ray (11, 22, 1) -> +Z hits z=5 at t=4, object point (1, 2, 0),
        // barycentrics (u=0.25, v=0.50).
        let hit = &out[..OPAQUE_LANE_STRIDE_DWORDS as usize];
        assert_eq!(hit[0], 1, "{label}: CommittedStatus");
        let ray_t = f32::from_bits(hit[1]);
        let bary_u = f32::from_bits(hit[2]);
        let bary_v = f32::from_bits(hit[3]);
        assert!(
            (ray_t - 4.0).abs() < 1e-4,
            "{label}: CommittedRayT got {ray_t}, want 4.0"
        );
        assert!(
            (bary_u - 0.25).abs() < 1e-4 && (bary_v - 0.50).abs() < 1e-4,
            "{label}: CommittedTriangleBarycentrics got ({bary_u}, {bary_v}), want (0.25, 0.50)"
        );
        assert_eq!(hit[4], 0x12345, "{label}: CommittedInstanceID");
        assert_eq!(hit[5], 0, "{label}: CommittedInstanceIndex");
        assert_eq!(hit[6], 0, "{label}: CommittedGeometryIndex");
        assert_eq!(hit[7], 0, "{label}: CommittedPrimitiveIndex");
        assert_eq!(hit[8], 7, "{label}: CommittedInstanceContributionToHitGroupIndex");
        // Triangle [(0,0,0),(4,0,0),(0,4,0)] viewed from -Z toward +Z is CCW (back-facing in LH DXR).
        assert_eq!(hit[9], 0, "{label}: CommittedTriangleFrontFace");
        for (k, &want) in transform.iter().enumerate() {
            let got = f32::from_bits(hit[10 + k]);
            assert!(
                (got - want).abs() < 1e-5,
                "{label}: CommittedObjectToWorld3x4[{k}] got {got}, want {want}"
            );
        }

        // Lane 1: missing ray (100, 100, 1) -> +Z.
        let miss = &out[OPAQUE_LANE_STRIDE_DWORDS as usize..];
        assert_eq!(miss[0], 0, "{label}: miss CommittedStatus");
        let miss_t = f32::from_bits(miss[1]);
        assert!(
            (miss_t - 100.0).abs() < 1e-4,
            "{label}: miss CommittedRayT got {miss_t}, want 100.0"
        );
    }
}

#[test]
fn gpu_rayquery_non_opaque_proceed_and_commit() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    if !has_raytracing_tier_1_1(&host) {
        eprintln!("hardware device lacks DXR Tier 1.1; skipped");
        return;
    }

    let near_tri = [[-2.0f32, -2.0, 2.0], [2.0, -2.0, 2.0], [0.0, 2.0, 2.0]];
    let far_tri = [[-2.0f32, -2.0, 5.0], [2.0, -2.0, 5.0], [0.0, 2.0, 5.0]];
    let geoms = [
        TriangleGeom {
            vertices: &near_tri,
            flags: D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,
        },
        TriangleGeom {
            vertices: &far_tri,
            flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        },
    ];
    let inst = InstanceSpec {
        transform: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
        instance_id: 0,
        instance_mask: 0xFF,
        contrib: 0,
        flags: 0,
    };

    let dxil = emit_non_opaque_proceed_cs();
    let out = run_rayquery_gpu(
        &host,
        &dxil,
        false,
        &geoms,
        &inst,
        2 * PROCEED_LANE_STRIDE_DWORDS,
    )
    .unwrap_or_else(|e| panic!("non-opaque proceed run_rayquery_gpu failed: {e}"));

    // Lane 0 rejected non-opaque geom 0 at z=2 -> hit opaque geom 1 at t=5.
    let lane0 = &out[..PROCEED_LANE_STRIDE_DWORDS as usize];
    assert_eq!(lane0[0], 1, "lane 0 CommittedStatus");
    let t0 = f32::from_bits(lane0[1]);
    assert!((t0 - 5.0).abs() < 1e-4, "lane 0 CommittedRayT got {t0}, want 5.0");
    assert_eq!(lane0[2], 1, "lane 0 CommittedGeometryIndex");
    assert_eq!(lane0[3], 0, "lane 0 CommittedPrimitiveIndex");

    // Lane 1 committed non-opaque geom 0 at z=2 -> hit geom 0 at t=2.
    let lane1 = &out[PROCEED_LANE_STRIDE_DWORDS as usize..];
    assert_eq!(lane1[0], 1, "lane 1 CommittedStatus");
    let t1 = f32::from_bits(lane1[1]);
    assert!((t1 - 2.0).abs() < 1e-4, "lane 1 CommittedRayT got {t1}, want 2.0");
    assert_eq!(lane1[2], 0, "lane 1 CommittedGeometryIndex");
    assert_eq!(lane1[3], 0, "lane 1 CommittedPrimitiveIndex");
}

#[test]
fn gpu_rayquery_two_pass_tmax_preshrink() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    if !has_raytracing_tier_1_1(&host) {
        eprintln!("hardware device lacks DXR Tier 1.1; skipped");
        return;
    }

    // Lane 0 at x=0: closer opaque geom 0 (z=2) before farther non-opaque geom 1 (z=5).
    let lane0_near_opaque = [[-2.0f32, -2.0, 2.0], [2.0, -2.0, 2.0], [0.0, 2.0, 2.0]];
    let lane0_far_nonopaque = [[-2.0f32, -2.0, 5.0], [2.0, -2.0, 5.0], [0.0, 2.0, 5.0]];
    // Lanes 1 & 2 at x=10: closer non-opaque geom 2 (z=2) before farther opaque geom 3 (z=5).
    let lane12_near_nonopaque = [[8.0f32, -2.0, 2.0], [12.0, -2.0, 2.0], [10.0, 2.0, 2.0]];
    let lane12_far_opaque = [[8.0f32, -2.0, 5.0], [12.0, -2.0, 5.0], [10.0, 2.0, 5.0]];
    let geoms = [
        TriangleGeom {
            vertices: &lane0_near_opaque,
            flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        },
        TriangleGeom {
            vertices: &lane0_far_nonopaque,
            flags: D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,
        },
        TriangleGeom {
            vertices: &lane12_near_nonopaque,
            flags: D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,
        },
        TriangleGeom {
            vertices: &lane12_far_opaque,
            flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        },
    ];
    let inst = InstanceSpec {
        transform: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
        instance_id: 0x5A5A,
        instance_mask: 0xFF,
        contrib: 3,
        flags: 0,
    };

    let dxil = emit_two_pass_anyhit_cs();
    let out = run_rayquery_gpu(
        &host,
        &dxil,
        false,
        &geoms,
        &inst,
        3 * TWO_PASS_LANE_STRIDE_DWORDS,
    )
    .unwrap_or_else(|e| panic!("two-pass anyhit run_rayquery_gpu failed: {e}"));

    let stride = TWO_PASS_LANE_STRIDE_DWORDS as usize;

    // (a) Lane 0: closer opaque hit (z=2) beats farther non-opaque (z=5); AnyHit skipped.
    let l0 = &out[0..stride];
    assert_eq!(l0[0], 1, "lane 0 CommittedStatus");
    let t0 = f32::from_bits(l0[1]);
    assert!((t0 - 2.0).abs() < 1e-4, "lane 0 CommittedRayT got {t0}, want 2.0");
    assert_eq!(l0[2], 0, "lane 0 CommittedGeometryIndex (Pass 1 opaque)");
    assert_eq!(l0[3], 0, "lane 0 CommittedPrimitiveIndex");
    assert_eq!(l0[4], 0x5A5A, "lane 0 CommittedInstanceID (Pass 1 metadata)");
    assert_eq!(l0[5], 0, "lane 0 AnyHit must be skipped when tMax pre-shrinks past non-opaque");

    // (b) Lane 1: closer accepted non-opaque hit (z=2) beats farther opaque (z=5).
    let l1 = &out[stride..2 * stride];
    assert_eq!(l1[0], 1, "lane 1 CommittedStatus");
    let t1 = f32::from_bits(l1[1]);
    assert!((t1 - 2.0).abs() < 1e-4, "lane 1 CommittedRayT got {t1}, want 2.0");
    assert_eq!(l1[2], 2, "lane 1 CommittedGeometryIndex (Pass 2 non-opaque)");
    assert_eq!(l1[3], 0, "lane 1 CommittedPrimitiveIndex");
    assert_eq!(l1[4], 0x5A5A, "lane 1 CommittedInstanceID (Pass 2 metadata)");
    assert_eq!(l1[5], 1, "lane 1 AnyHit invoked once");

    // (c) Lane 2: closer rejected non-opaque hit (z=2) falls back to farther opaque (z=5).
    let l2 = &out[2 * stride..3 * stride];
    assert_eq!(l2[0], 1, "lane 2 CommittedStatus");
    let t2 = f32::from_bits(l2[1]);
    assert!((t2 - 5.0).abs() < 1e-4, "lane 2 CommittedRayT got {t2}, want 5.0");
    assert_eq!(l2[2], 3, "lane 2 CommittedGeometryIndex (Pass 1 fallback)");
    assert_eq!(l2[3], 0, "lane 2 CommittedPrimitiveIndex");
    assert_eq!(l2[4], 0x5A5A, "lane 2 CommittedInstanceID (Pass 1 metadata)");
    assert_eq!(l2[5], 1, "lane 2 AnyHit invoked once before fallback");
}
