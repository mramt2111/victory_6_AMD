//! Every pipeline the title has ever created, kept on disk
//! (`VICTORY_PIPELINE_BOOK=1`), and their host pipelines made ahead of a run
//! (`victory warm`), so that the driver's own cache on disk has them before
//! a draw asks for one.
//!
//! The title creates 84,000 pipelines while it loads and draws with some
//! 2,000 of them; which ones is known from earlier runs (`library`), and
//! those are made during loading. One met for the first time (a place never
//! seen, or everything after a change of the translator or of the driver) is
//! compiled when a draw wants it. Making all of them in the running title
//! (`VICTORY_EARLY_ALL=1`) ended it inside the driver's compiler; made here,
//! a compiler that crashes or hangs ends a child process, costs that pipeline
//! alone and is named in `warm-bad.txt`.
//!
//! Two files beside the library, both only ever added to (a torn end is cut
//! off): `known-shaders.bin` (stage, hash, length, container) and
//! `known-pipelines.bin` (length, then the key, the shaders' stages and
//! hashes, and the graphics state). They hold the title's own code: they stay
//! in the cache folder.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::CString;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

mod writer;

const SHADERS: &str = "known-shaders.bin";
const PIPELINES: &str = "known-pipelines.bin";
/// Stages by their number in the files.
const STAGES: [&str; 8] =
    ["vertex", "pixel", "domain", "hull", "geometry", "compute", "amplification", "mesh"];

/// `VICTORY_PIPELINE_BOOK=1`: the pipelines the title creates are kept.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_BOOK").is_some_and(|v| v == "1"))
}

struct Book {
    dir: PathBuf,
    keys: HashSet<[u8; 16]>,
    shaders: HashSet<[u8; 32]>,
    /// Not yet written: shaders, then pipelines.
    new: [Vec<u8>; 2],
    added: usize,
}

struct Recorder {
    book: Mutex<Book>,
    writer: writer::Writer,
}

static BOOK: OnceLock<Option<Recorder>> = OnceLock::new();

fn book() -> Option<&'static Recorder> {
    BOOK.get_or_init(|| {
        let dir = super::library::folder().filter(|_| on())?;
        std::fs::create_dir_all(&dir).ok()?;
        let mut keys = HashSet::new();
        let mut shaders = HashSet::new();
        let ends = [
            walk_shaders(&dir, |_, hash, _, _| {
                let _ = shaders.insert(hash);
            }),
            walk_pipelines(&dir, |_, bytes| {
                keys.extend(bytes.first_chunk::<16>().copied());
            }),
        ];
        for (file, end) in [SHADERS, PIPELINES].iter().zip(ends) {
            let path = dir.join(file);
            if std::fs::metadata(&path).is_ok_and(|m| m.len() > end)
                && let Ok(f) = std::fs::OpenOptions::new().write(true).open(&path)
            {
                let _ = f.set_len(end);
            }
        }
        tracing::info!(pipelines = keys.len(), shaders = shaders.len(), "title pipelines known from earlier runs");
        Some(Recorder {
            writer: writer::Writer::start(flush).ok()?,
            book: Mutex::new(Book { dir, keys, shaders, new: Default::default(), added: 0 }),
        })
    })
    .as_ref()
}

/// The title created a pipeline.
pub(super) fn note(key: &str, shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) {
    let Some(recorder) = book() else { return };
    let mut id = [0u8; 16];
    for (at, b) in id.iter_mut().enumerate() {
        *b = key.get(2 * at..2 * at + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).unwrap_or(0);
    }
    let mut book = recorder.book.lock().unwrap_or_else(|e| e.into_inner());
    if shaders.is_empty() || !book.keys.insert(id) {
        return;
    }
    let mut entry = id.to_vec();
    entry.push(shaders.len() as u8);
    for s in shaders {
        let hash = *blake3::hash(&s.container).as_bytes();
        let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u8;
        entry.push(stage);
        entry.extend(hash);
        if book.shaders.insert(hash) {
            let new = &mut book.new[0];
            new.push(stage);
            new.extend(hash);
            new.extend((s.container.len() as u32).to_le_bytes());
            new.extend(&s.container);
        }
    }
    entry.push(u8::from(graphics.is_some()));
    if let Some(g) = graphics {
        g.put(&mut entry);
    }
    book.new[1].extend((entry.len() as u32).to_le_bytes());
    book.new[1].extend(entry);
    book.added += 1;
}

/// Appends pipelines noted since the last successful write, if the book is active.
pub fn flush() {
    let Some(Some(recorder)) = BOOK.get() else { return };
    flush_book(&recorder.book);
}

/// Gives the existing writer a short chance to save after the fatal GPU report.
pub fn flush_on_gpu_failure() {
    if let Some(Some(recorder)) = BOOK.get() {
        recorder.writer.request_flush(std::time::Duration::from_millis(250));
    }
}

fn flush_book(book: &Mutex<Book>) {
    // The timer and shutdown may both flush. Keep their writes in order while
    // allowing the title to note more pipelines during disk I/O.
    static WRITER: Mutex<()> = Mutex::new(());
    let _writer = WRITER.lock().unwrap_or_else(|e| e.into_inner());
    let (dir, mut new, added) = {
        let mut book = book.lock().unwrap_or_else(|e| e.into_inner());
        if book.added == 0 {
            return;
        }
        (book.dir.clone(), std::mem::take(&mut book.new), std::mem::take(&mut book.added))
    };
    for (file, bytes) in [SHADERS, PIPELINES].into_iter().zip(&mut new) {
        if bytes.is_empty() {
            continue;
        }
        let written = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(file))
            .and_then(|f| append_bytes(f, bytes));
        if let Err(e) = written {
            let mut book = book.lock().unwrap_or_else(|e| e.into_inner());
            for (mut older, newer) in new.into_iter().zip(&mut book.new) {
                if !older.is_empty() {
                    older.append(newer);
                    *newer = older;
                }
            }
            book.added += added;
            drop(book);
            tracing::warn!(error = %e, file, "keeping unwritten title pipelines for retry");
            return;
        }
    }
    tracing::info!(pipelines = added, "title pipelines written down");
}

/// A failed append may already have written a prefix of a record.
fn append_bytes(mut out: impl Write, bytes: &mut Vec<u8>) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    let mut written = 0;
    let result = loop {
        if written == bytes.len() {
            break Ok(());
        }
        match out.write(&bytes[written..]) {
            Ok(0) => break Err(Error::from(ErrorKind::WriteZero)),
            Ok(n) => written += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => break Err(e),
        }
    };
    bytes.drain(..written);
    result
}

/// Calls `f` with each container's stage, hash, place and length; where the
/// last whole one ends.
fn walk_shaders(dir: &Path, mut f: impl FnMut(u8, [u8; 32], u64, u32)) -> u64 {
    let Ok(file) = std::fs::File::open(dir.join(SHADERS)) else { return 0 };
    let len = file.metadata().map_or(0, |m| m.len());
    let mut file = std::io::BufReader::new(file);
    let mut at = 0;
    let mut head = [0u8; 37];
    while file.read_exact(&mut head).is_ok() {
        let size = u32::from_le_bytes([head[33], head[34], head[35], head[36]]);
        if at + 37 + u64::from(size) > len || file.seek_relative(i64::from(size)).is_err() {
            break;
        }
        let mut hash = [0; 32];
        hash.copy_from_slice(&head[1..33]);
        f(head[0], hash, at + 37, size);
        at += 37 + u64::from(size);
    }
    at
}

/// Calls `f` with each pipeline's number and entry; where the last whole one
/// ends.
fn walk_pipelines(dir: &Path, mut f: impl FnMut(usize, &[u8])) -> u64 {
    let bytes = std::fs::read(dir.join(PIPELINES)).unwrap_or_default();
    let (mut at, mut n) = (0, 0);
    while let Some(head) = bytes.get(at..at + 4) {
        let size = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize;
        let Some(entry) = bytes.get(at + 4..at + 4 + size) else { break };
        f(n, entry);
        (at, n) = (at + 4 + size, n + 1);
    }
    at as u64
}

/// How many pipelines are known in `dir`.
pub fn count(dir: &Path) -> usize {
    let mut n = 0;
    walk_pipelines(dir, |_, _| n += 1);
    n
}

/// The key of each pipeline known in `dir`, as the early list has it.
pub fn keys(dir: &Path) -> Vec<String> {
    let mut all = Vec::new();
    walk_pipelines(dir, |_, e| {
        all.push(e.get(..16).unwrap_or_default().iter().map(|b| format!("{b:02x}")).collect());
    });
    all
}

/// What a driver's cached pipelines depend on besides the title's: the
/// translator with its settings, and the driver.
pub fn fingerprint() -> String {
    let mut h = blake3::Hasher::new();
    h.update(&crate::shader_cache::translation());
    if let Some(host) = crate::host::host() {
        h.update(&host.driver().to_le_bytes());
    }
    h.finalize().to_hex()[..16].to_owned()
}

/// The translations of earlier runs are in `dir` (the pipeline library is
/// not used: what is made here is the driver's to keep).
pub fn use_translations(dir: PathBuf) {
    crate::shader_cache::use_shader_cache(dir);
}

/// Makes the host pipelines of those of `which` (numbers in the file) and
/// lets each go again: the driver's cache on disk keeps them. `each` is told
/// the number before, and what came of it after: milliseconds and whether
/// there is one.
pub fn make(dir: &Path, which: &[usize], mut each: impl FnMut(usize, Option<(f64, bool)>)) {
    let mut places = HashMap::new();
    walk_shaders(dir, |stage, hash, at, len| {
        let _ = places.insert(hash, (stage, at, len));
    });
    let Ok(mut file) = std::fs::File::open(dir.join(SHADERS)) else { return };
    let wanted: HashSet<usize> = which.iter().copied().collect();
    let mut entries = HashMap::new();
    walk_pipelines(dir, |n, bytes| {
        if wanted.contains(&n) {
            entries.insert(n, bytes.to_vec());
        }
    });
    let mut read = |hash: &[u8; 32]| {
        let &(_, at, len) = places.get(hash)?;
        let mut bytes = vec![0; len as usize];
        file.seek(SeekFrom::Start(at)).ok()?;
        file.read_exact(&mut bytes).ok()?;
        Some(bytes)
    };
    for n in which {
        let Some(entry) = entries.get(n) else { continue };
        each(*n, None);
        let began = std::time::Instant::now();
        let made = parse(entry, &mut read)
            .and_then(|(shaders, g)| HostPipeline::new(&shaders, g.as_deref()))
            .is_some();
        each(*n, Some((began.elapsed().as_secs_f64() * 1e3, made)));
    }
}

#[allow(clippy::type_complexity)]
fn parse(
    entry: &[u8],
    read: &mut impl FnMut(&[u8; 32]) -> Option<Vec<u8>>,
) -> Option<(Vec<Arc<Shader>>, Option<Box<GraphicsState>>)> {
    let mut rest = entry.get(16..)?;
    let (&count, after) = rest.split_first()?;
    rest = after;
    let mut shaders = Vec::new();
    for _ in 0..count {
        let (head, after) = rest.split_first_chunk::<33>()?;
        rest = after;
        let mut hash = [0; 32];
        hash.copy_from_slice(&head[1..]);
        shaders.push(super::shader(STAGES.get(usize::from(head[0]))?, &read(&hash)?));
    }
    let (&graphics, rest) = rest.split_first()?;
    let g = match graphics {
        0 => None,
        _ => Some(Box::new(GraphicsState::get(rest)?)),
    };
    Some((shaders, g))
}

/// Calls `$f` with each number of a graphics state but its input elements',
/// as a place of 32 bits, in one order.
macro_rules! numbers {
    ($g:expr, $f:ident) => {{
        $f!($g.blend.AlphaToCoverageEnable.0);
        $f!($g.blend.IndependentBlendEnable.0);
        for t in &mut $g.blend.RenderTarget {
            $f!(t.BlendEnable.0);
            $f!(t.LogicOpEnable.0);
            $f!(t.SrcBlend.0);
            $f!(t.DestBlend.0);
            $f!(t.BlendOp.0);
            $f!(t.SrcBlendAlpha.0);
            $f!(t.DestBlendAlpha.0);
            $f!(t.BlendOpAlpha.0);
            $f!(t.LogicOp.0);
            $f!(t.RenderTargetWriteMask);
        }
        $f!($g.sample_mask);
        $f!($g.rasterizer.FillMode.0);
        $f!($g.rasterizer.CullMode.0);
        $f!($g.rasterizer.FrontCounterClockwise.0);
        $f!($g.rasterizer.DepthBias);
        $f!($g.rasterizer.DepthClipEnable.0);
        $f!($g.rasterizer.MultisampleEnable.0);
        $f!($g.rasterizer.AntialiasedLineEnable.0);
        $f!($g.rasterizer.ForcedSampleCount);
        $f!($g.rasterizer.ConservativeRaster.0);
        $f!($g.depth.DepthEnable.0);
        $f!($g.depth.DepthWriteMask.0);
        $f!($g.depth.DepthFunc.0);
        $f!($g.depth.StencilEnable.0);
        $f!($g.depth.StencilReadMask);
        $f!($g.depth.StencilWriteMask);
        for face in [&mut $g.depth.FrontFace, &mut $g.depth.BackFace] {
            $f!(face.StencilFailOp.0);
            $f!(face.StencilDepthFailOp.0);
            $f!(face.StencilPassOp.0);
            $f!(face.StencilFunc.0);
        }
        $f!($g.strip_cut.0);
        $f!($g.topology.0);
        $f!($g.targets.NumRenderTargets);
        for format in &mut $g.targets.RTFormats {
            $f!(format.0);
        }
        $f!($g.depth_format.0);
        $f!($g.samples.Count);
        $f!($g.samples.Quality);
    }};
}

impl GraphicsState {
    /// The state as bytes.
    fn put(&self, out: &mut Vec<u8>) {
        out.push(u8::from(self.depth_bounds));
        for f in [self.rasterizer.DepthBiasClamp, self.rasterizer.SlopeScaledDepthBias] {
            out.extend(f.to_le_bytes());
        }
        let mut plain = GraphicsState {
            blend: self.blend,
            sample_mask: self.sample_mask,
            rasterizer: self.rasterizer,
            depth: self.depth,
            strip_cut: self.strip_cut,
            topology: self.topology,
            targets: self.targets,
            depth_format: self.depth_format,
            samples: self.samples,
            ..Default::default()
        };
        macro_rules! write {
            ($x:expr) => {
                out.extend(($x as u32).to_le_bytes())
            };
        }
        numbers!(plain, write);
        out.push(self.inputs.len() as u8);
        for (e, name) in self.inputs.iter().zip(&self.names) {
            let name = name.as_bytes();
            out.push(name.len() as u8);
            out.extend(name);
            for v in [
                e.SemanticIndex,
                e.Format.0 as u32,
                e.InputSlot,
                e.AlignedByteOffset,
                e.InputSlotClass.0 as u32,
                e.InstanceDataStepRate,
            ] {
                out.extend(v.to_le_bytes());
            }
        }
    }

    fn get(bytes: &[u8]) -> Option<GraphicsState> {
        let mut g = GraphicsState::default();
        let (&bounds, mut rest) = bytes.split_first()?;
        g.depth_bounds = bounds != 0;
        let mut next = || {
            let (head, after) = rest.split_first_chunk::<4>()?;
            rest = after;
            Some(u32::from_le_bytes(*head))
        };
        g.rasterizer.DepthBiasClamp = f32::from_bits(next()?);
        g.rasterizer.SlopeScaledDepthBias = f32::from_bits(next()?);
        macro_rules! read {
            ($x:expr) => {
                $x = next()? as _
            };
        }
        numbers!(g, read);
        let (&count, mut rest) = rest.split_first()?;
        let mut elements = Vec::new();
        for _ in 0..count {
            let (&len, after) = rest.split_first()?;
            let (name, after) = after.split_at_checked(usize::from(len))?;
            let (numbers, after) = after.split_first_chunk::<24>()?;
            rest = after;
            let mut v = [0u32; 6];
            for (v, b) in v.iter_mut().zip(numbers.chunks_exact(4)) {
                *v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            elements.push((CString::new(name).ok()?, v));
        }
        g.names = elements.iter().map(|e| e.0.clone()).collect();
        g.inputs = g
            .names
            .iter()
            .zip(&elements)
            .map(|(name, (_, v))| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: v[0],
                Format: DXGI_FORMAT(v[1] as i32),
                InputSlot: v[2],
                AlignedByteOffset: v[3],
                InputSlotClass: D3D12_INPUT_CLASSIFICATION(v[4] as i32),
                InstanceDataStepRate: v[5],
            })
            .collect();
        Some(g)
    }
}

/// Maximum time between two PSO draws to count them as co-occurring (tau = 0.5 s).
const CO_OCCURRENCE_WINDOW: Duration = Duration::from_millis(500);
/// Maximum distinct PSO keys tracked in memory (for both edges and compile-time EWMA).
const CO_OCCURRENCE_MAX_KEYS: usize = 4_096;
/// Maximum co-occurring neighbors retained per PSO key.
const CO_OCCURRENCE_MAX_NEIGHBORS: usize = 8;
/// Maximum recent draws kept in the active sliding window.
const CO_OCCURRENCE_RECENT_CAP: usize = 16;
/// Exponential decay half-life for co-occurrence weights (T_1/2 = 120 s).
const CO_OCCURRENCE_HALF_LIFE_SECS: f32 = 120.0;
/// Exponential decay rate lambda = ln(2) / T_1/2 (s^-1).
const CO_OCCURRENCE_LAMBDA: f32 = std::f32::consts::LN_2 / CO_OCCURRENCE_HALF_LIFE_SECS;
/// Intra-window temporal kernel rate alpha = 2.0 s^-1 for exp(-alpha * dt).
const CO_OCCURRENCE_ALPHA: f32 = 2.0;
/// Default prior compile duration c_hat = 8.0 ms for unseen PSOs.
const COMPILE_MS_PRIOR: f32 = 8.0;
/// Frame wait-budget cap (ms) in Smith's-rule utility numerator min(c_hat, 32.0).
const COMPILE_MS_BUDGET_CAP: f32 = 32.0;
/// Fixed overhead term (ms) in Smith's-rule utility denominator (c_hat + 0.5).
const COMPILE_MS_OVERHEAD: f32 = 0.5;
/// EWMA smoothing factor for per-PSO measured compile durations.
const COMPILE_MS_EWMA_ALPHA: f32 = 0.25;

#[derive(Clone, Copy)]
struct DecayedEdge {
    key: u64,
    weight: f32,
    last_updated: Instant,
}

impl DecayedEdge {
    #[inline]
    fn decayed_weight(&self, now: Instant) -> f32 {
        let elapsed = now.saturating_duration_since(self.last_updated).as_secs_f32();
        self.weight * (-CO_OCCURRENCE_LAMBDA * elapsed).exp()
    }
}

struct CoOccurrenceTracker {
    recent: VecDeque<(u64, Instant)>,
    graph: HashMap<u64, Vec<DecayedEdge>>,
    compile_ms: HashMap<u64, f32>,
    last_now: Option<Instant>,
    window: Duration,
    max_keys: usize,
    max_neighbors: usize,
}

impl CoOccurrenceTracker {
    fn new() -> Self {
        Self::with_limits(
            CO_OCCURRENCE_WINDOW,
            CO_OCCURRENCE_MAX_KEYS,
            CO_OCCURRENCE_MAX_NEIGHBORS,
        )
    }

    fn with_limits(window: Duration, max_keys: usize, max_neighbors: usize) -> Self {
        Self {
            recent: VecDeque::with_capacity(CO_OCCURRENCE_RECENT_CAP),
            graph: HashMap::new(),
            compile_ms: HashMap::new(),
            last_now: None,
            window,
            max_keys,
            max_neighbors,
        }
    }

    fn record_at(&mut self, key: u64, now: Instant) {
        self.last_now = Some(self.last_now.map_or(now, |prev| prev.max(now)));
        while let Some(&(_, ts)) = self.recent.front() {
            if now.saturating_duration_since(ts) > self.window {
                let _ = self.recent.pop_front();
            } else {
                break;
            }
        }
        for i in 0..self.recent.len() {
            let (prev, prev_ts) = self.recent[i];
            if prev != key {
                let dt = now.saturating_duration_since(prev_ts);
                self.add_edge(prev, key, now, dt);
                self.add_edge(key, prev, now, dt);
            }
        }
        if let Some(pos) = self.recent.iter().position(|&(k, _)| k == key) {
            let _ = self.recent.remove(pos);
        } else if self.recent.len() >= CO_OCCURRENCE_RECENT_CAP {
            let _ = self.recent.pop_front();
        }
        self.recent.push_back((key, now));
    }

    fn add_edge(&mut self, from: u64, to: u64, now: Instant, dt: Duration) {
        if self.max_keys == 0 || self.max_neighbors == 0 {
            return;
        }
        let delta_w = if dt <= self.window {
            (-CO_OCCURRENCE_ALPHA * dt.as_secs_f32()).exp()
        } else {
            0.0
        };
        if delta_w <= 0.0 {
            return;
        }
        if !self.graph.contains_key(&from) && self.graph.len() >= self.max_keys {
            return;
        }
        let neighbors = self.graph.entry(from).or_default();
        if let Some(edge) = neighbors.iter_mut().find(|e| e.key == to) {
            edge.weight = edge.decayed_weight(now) + delta_w;
            edge.last_updated = now;
            return;
        }
        if neighbors.len() < self.max_neighbors {
            neighbors.push(DecayedEdge {
                key: to,
                weight: delta_w,
                last_updated: now,
            });
        } else if let Some((min_idx, min_weight)) = neighbors
            .iter()
            .enumerate()
            .map(|(idx, e)| (idx, e.decayed_weight(now)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            && min_weight <= delta_w
        {
            neighbors[min_idx] = DecayedEdge {
                key: to,
                weight: delta_w,
                last_updated: now,
            };
        }
    }

    fn record_compile_ms(&mut self, key: u64, ms: f32) {
        if !ms.is_finite() || ms < 0.0 || self.max_keys == 0 {
            return;
        }
        if let Some(ewma) = self.compile_ms.get_mut(&key) {
            *ewma = *ewma * (1.0 - COMPILE_MS_EWMA_ALPHA) + ms * COMPILE_MS_EWMA_ALPHA;
            return;
        }
        if self.compile_ms.len() >= self.max_keys {
            return;
        }
        let _ = self.compile_ms.insert(key, ms);
    }

    fn estimated_compile_ms(&self, key: u64) -> f32 {
        self.compile_ms.get(&key).copied().unwrap_or(COMPILE_MS_PRIOR)
    }

    fn predicted_neighbors(&self, key: u64, limit: usize) -> Vec<u64> {
        let now = self.last_now.unwrap_or_else(Instant::now);
        self.predicted_neighbors_at(key, limit, now)
    }

    fn predicted_neighbors_at(&self, key: u64, limit: usize, now: Instant) -> Vec<u64> {
        if limit == 0 {
            return Vec::new();
        }
        let Some(neighbors) = self.graph.get(&key) else {
            return Vec::new();
        };
        let mut weights: Vec<(u64, f32)> = Vec::with_capacity(neighbors.len());
        let mut total_weight = 0.0f32;
        for edge in neighbors {
            let w = edge.decayed_weight(now);
            if w > 0.0 && w.is_finite() {
                total_weight += w;
                weights.push((edge.key, w));
            }
        }
        if total_weight <= 0.0 || !total_weight.is_finite() {
            return Vec::new();
        }
        let mut ranked: Vec<(u64, f32)> = weights
            .into_iter()
            .map(|(neighbor, w)| {
                let p_ij = w / total_weight;
                let c_hat = self.estimated_compile_ms(neighbor);
                let utility =
                    (p_ij * c_hat.min(COMPILE_MS_BUDGET_CAP)) / (c_hat + COMPILE_MS_OVERHEAD);
                (neighbor, utility)
            })
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(limit);
        ranked.into_iter().map(|(k, _)| k).collect()
    }
}

static CO_OCCURRENCE: Mutex<Option<CoOccurrenceTracker>> = Mutex::new(None);

/// Records a PSO key drawn/bound by the title so co-occurring PSOs drawn
/// within 500 ms can be speculatively pre-warmed.
pub(crate) fn record_draw_key(key: u64) {
    let now = Instant::now();
    let mut tracker = CO_OCCURRENCE.lock().unwrap_or_else(|e| e.into_inner());
    tracker
        .get_or_insert_with(CoOccurrenceTracker::new)
        .record_at(key, now);
}

/// Records a measured PSO compile duration (in milliseconds) in the per-PSO
/// EWMA estimator used by Smith's-rule utility ranking.
pub(crate) fn record_compile_ms(key: u64, ms: f32) {
    if !ms.is_finite() || ms < 0.0 {
        return;
    }
    let mut tracker = CO_OCCURRENCE.lock().unwrap_or_else(|e| e.into_inner());
    tracker
        .get_or_insert_with(CoOccurrenceTracker::new)
        .record_compile_ms(key, ms);
}

/// Returns up to `limit` neighbor PSO keys ranked by exponentially decayed
/// Smith's-rule utility.
pub(crate) fn predicted_neighbors(key: u64, limit: usize) -> Vec<u64> {
    if limit == 0 {
        return Vec::new();
    }
    let now = Instant::now();
    let tracker = CO_OCCURRENCE.lock().unwrap_or_else(|e| e.into_inner());
    tracker
        .as_ref()
        .map(|t| t.predicted_neighbors_at(key, limit, now))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_open_retries_before_later_records() {
        for blocked in [SHADERS, PIPELINES] {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path();
            let book = Mutex::new(Book {
                dir: dir.into(),
                keys: HashSet::new(),
                shaders: HashSet::new(),
                new: [vec![1], vec![11]],
                added: 1,
            });
            std::fs::create_dir(dir.join(blocked)).unwrap();
            flush_book(&book);
            std::fs::remove_dir(dir.join(blocked)).unwrap();
            {
                let mut b = book.lock().unwrap();
                assert_eq!(b.added, 1);
                b.new[0].push(2);
                b.new[1].push(12);
                b.added += 1;
            }
            flush_book(&book);
            flush_book(&book);
            assert_eq!(std::fs::read(dir.join(SHADERS)).unwrap(), [1, 2]);
            assert_eq!(std::fs::read(dir.join(PIPELINES)).unwrap(), [11, 12]);
            assert_eq!(book.lock().unwrap().added, 0);
        }
    }

    #[test]
    fn a_short_write_keeps_only_the_unwritten_suffix() {
        let mut pending = vec![1, 2, 3, 4, 5];
        let mut first = [0; 2];
        // A fixed-size output accepts the prefix, then returns Ok(0).
        let err = append_bytes(&mut first[..], &mut pending).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::WriteZero);
        assert_eq!(first, [1, 2]);
        assert_eq!(pending, [3, 4, 5]);
        let mut rest = Vec::new();
        append_bytes(&mut rest, &mut pending).unwrap();
        assert!(pending.is_empty());
        assert_eq!(rest, [3, 4, 5]);
    }

    #[test]
    fn a_state_comes_back_as_it_was() {
        let mut g = GraphicsState { sample_mask: 7, depth_bounds: true, ..Default::default() };
        g.rasterizer.DepthBias = -3;
        g.rasterizer.SlopeScaledDepthBias = 1.25;
        g.blend.RenderTarget[3].RenderTargetWriteMask = 5;
        g.depth.StencilReadMask = 9;
        g.targets.NumRenderTargets = 2;
        g.targets.RTFormats[1] = DXGI_FORMAT_R8_UNORM;
        g.depth_format = DXGI_FORMAT_D32_FLOAT;
        g.names = vec![CString::new("POSITION").unwrap(), CString::new("T").unwrap()];
        g.inputs = g
            .names
            .iter()
            .enumerate()
            .map(|(n, name)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: n as u32,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 2,
                AlignedByteOffset: D3D12_APPEND_ALIGNED_ELEMENT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                InstanceDataStepRate: 4,
            })
            .collect();
        let mut bytes = Vec::new();
        g.put(&mut bytes);
        let back = GraphicsState::get(&bytes).expect("whole");
        assert_eq!(back.describe(), g.describe());
        assert!(GraphicsState::get(&bytes[..bytes.len() - 1]).is_none());
        assert!(GraphicsState::get(&bytes[..9]).is_none());
    }

    #[test]
    fn a_torn_end_is_not_read() {
        let dir = std::env::temp_dir().join(format!("victory-known-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes = Vec::new();
        for n in 0..3u8 {
            bytes.extend(20u32.to_le_bytes());
            bytes.extend([n; 20]);
        }
        std::fs::write(dir.join(PIPELINES), &bytes[..bytes.len() - 5]).unwrap();
        assert_eq!(count(&dir), 2);
        assert_eq!(keys(&dir)[1], "01".repeat(16));
        assert_eq!(walk_pipelines(&dir, |_, _| ()), 48);
        let mut shader = vec![5u8];
        shader.extend([9; 32]);
        shader.extend(4u32.to_le_bytes());
        shader.extend([1, 2, 3, 4]);
        let one = shader.len() as u64;
        shader.extend_from_slice(&shader.clone()[..20]);
        std::fs::write(dir.join(SHADERS), &shader).unwrap();
        let mut seen = Vec::new();
        assert_eq!(walk_shaders(&dir, |stage, hash, at, len| seen.push((stage, hash[0], at, len))), one);
        assert_eq!(seen, [(5, 9, 37, 4)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn co_occurrence_recording_ranking_and_bounds() {
        let base = Instant::now();
        let mut tracker = CoOccurrenceTracker::new();
        assert_eq!(tracker.window, Duration::from_millis(500));
        assert_eq!(tracker.max_keys, 4_096);
        assert_eq!(tracker.max_neighbors, 8);

        // Draw key 10 with 20 three times, with 30 twice, and with 40 once within 500 ms.
        tracker.record_at(10, base);
        tracker.record_at(20, base + Duration::from_millis(50));
        tracker.record_at(30, base + Duration::from_millis(100));
        tracker.record_at(40, base + Duration::from_millis(150));

        tracker.record_at(10, base + Duration::from_millis(200));
        tracker.record_at(20, base + Duration::from_millis(220));
        tracker.record_at(30, base + Duration::from_millis(240));

        tracker.record_at(10, base + Duration::from_millis(300));
        tracker.record_at(20, base + Duration::from_millis(320));

        // Key 50 arrives > 500 ms after the previous draws: must not pair with 10/20/30/40.
        tracker.record_at(50, base + Duration::from_millis(1_000));

        assert_eq!(tracker.predicted_neighbors(10, 4), vec![20, 30, 40]);
        assert_eq!(tracker.predicted_neighbors(10, 2), vec![20, 30]);
        assert_eq!(tracker.predicted_neighbors(10, 0), Vec::<u64>::new());
        assert_eq!(tracker.predicted_neighbors(50, 4), Vec::<u64>::new());

        // Verify key and neighbor capacity bounds.
        let mut bounded = CoOccurrenceTracker::with_limits(Duration::from_millis(500), 4, 3);
        let t0 = Instant::now();
        for k in 1..=12u64 {
            bounded.record_at(k, t0 + Duration::from_millis(k * 10));
        }
        assert!(bounded.graph.len() <= 4);
        for neighbors in bounded.graph.values() {
            assert!(neighbors.len() <= 3);
        }
        for k in 1..=12u64 {
            bounded.record_compile_ms(k, 5.0);
        }
        assert!(bounded.compile_ms.len() <= 4);
    }

    #[test]
    fn exponential_decay_forgets_stale_levels() {
        let base = Instant::now();
        let mut tracker = CoOccurrenceTracker::new();

        // Level 1: draw key 10 with stale neighbor 20 six times (1 s apart, dt = 20 ms each).
        for step in 0..6u64 {
            let t = base + Duration::from_secs(step);
            tracker.record_at(10, t);
            tracker.record_at(20, t + Duration::from_millis(20));
        }
        assert_eq!(
            tracker.predicted_neighbors_at(10, 2, base + Duration::from_secs(6)),
            vec![20]
        );

        // Verify exact half-life decay on a stored edge after 120 s.
        let edge_20 = tracker
            .graph
            .get(&10)
            .and_then(|edges| edges.iter().find(|e| e.key == 20))
            .copied()
            .expect("edge 10 -> 20 present");
        let after_half_life = edge_20.decayed_weight(edge_20.last_updated + Duration::from_secs(120));
        assert!((after_half_life - edge_20.weight * 0.5).abs() < 1e-4);

        // Level 2 (10 minutes = 5 half-lives later): stale weight decays by 2^-5 = 1/32,
        // so a single fresh co-occurrence with neighbor 30 outranks stale neighbor 20.
        let level2 = base + Duration::from_secs(600);
        tracker.record_at(10, level2);
        tracker.record_at(30, level2 + Duration::from_millis(20));
        assert_eq!(
            tracker.predicted_neighbors_at(10, 2, level2 + Duration::from_millis(20)),
            vec![30, 20]
        );
    }

    #[test]
    fn smiths_rule_utility_ranks_fast_psos_above_slow_compute() {
        let base = Instant::now();
        let mut tracker = CoOccurrenceTracker::new();

        // Pair key 100 with slow compute PSO 200 three times, and fast graphics PSO 300 once.
        for step in 0..3u64 {
            let t = base + Duration::from_secs(step);
            tracker.record_at(100, t);
            tracker.record_at(200, t + Duration::from_millis(20));
        }
        let t_fast = base + Duration::from_secs(3);
        tracker.record_at(100, t_fast);
        tracker.record_at(300, t_fast + Duration::from_millis(20));

        // With equal default priors (8.0 ms), higher-probability 200 outranks 300.
        assert_eq!(tracker.estimated_compile_ms(200), COMPILE_MS_PRIOR);
        assert_eq!(tracker.estimated_compile_ms(300), COMPILE_MS_PRIOR);
        assert_eq!(tracker.predicted_neighbors(100, 2), vec![200, 300]);

        // Record measured compile times: 200 takes 2,000 ms, 300 takes 4 ms.
        // U(300|100) ~= (0.25 * 4.0) / 4.5 ~= 0.222 >> U(200|100) ~= (0.75 * 32.0) / 2000.5 ~= 0.012.
        tracker.record_compile_ms(200, 2_000.0);
        tracker.record_compile_ms(300, 4.0);
        assert_eq!(tracker.predicted_neighbors(100, 2), vec![300, 200]);

        // Verify EWMA smoothing and non-finite / negative guards.
        tracker.record_compile_ms(300, 12.0);
        assert!((tracker.estimated_compile_ms(300) - 6.0).abs() < 1e-5);
        tracker.record_compile_ms(300, f32::NAN);
        tracker.record_compile_ms(300, f32::INFINITY);
        tracker.record_compile_ms(300, -5.0);
        assert!((tracker.estimated_compile_ms(300) - 6.0).abs() < 1e-5);
    }
}
