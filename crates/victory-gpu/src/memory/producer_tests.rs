use super::*;
use crate::descriptor::host::FIRST_RAW_PAGE_SLOTS;
use windows::core::s;
use windows_sys::Win32::System::Memory::*;

#[test]
fn recreated_heap_reserves_its_own_raw_page_views() {
    let Some(mut heap) = HostHeap::new(false, 4) else { return };
    assert_eq!(heap.raw_page_views(&[]), Some((4, 0)));
    assert_eq!(heap.raw_page_views(&[]), Some((4, 0)));
    assert_eq!(heap.dynamic(1), Some(4 + FIRST_RAW_PAGE_SLOTS));
    heap = HostHeap::new(false, 4).unwrap();
    assert_eq!(heap.dynamic(7), Some(4));
    assert_eq!(heap.raw_page_views(&[]), Some((11, 0)));
    assert_eq!(heap.dynamic(1), Some(11 + FIRST_RAW_PAGE_SLOTS));
}

// Exercise dynamic growth of the raw-memory descriptor run past the initial
// reservation while an earlier recorded command list still targets the old base.
#[test]
fn raw_page_views_grow_past_initial_run_while_older_lists_live() {
    let Some(host) = host() else {
        assert!(std::env::var_os("VICTORY_D3D12_WARP").is_none());
        return;
    };
    const CELL: u64 = 4 << 20;
    let page = (0x1500..0x1600)
        .map(|i| i * 4 * CELL)
        .find(|&at| unsafe {
            !VirtualAlloc(
                at as _,
                PAGE as usize,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
            .is_null()
        })
        .expect("reserve a guest page");
    let field = page + 0x180;
    let first_marker = 0x5123_abcd_u32;
    let second_marker = 0x1220_1260_u32;
    let mut memory = Memory::new().unwrap();
    let heap = HostHeap::new(false, 4).unwrap();
    let source = r#"
        ByteAddressBuffer directory : register(t0);
        RWByteAddressBuffer pages[8192] : register(u0);
        cbuffer Args : register(b0) { uint page; uint offset; uint marker; };
        [RootSignature("SRV(t0), DescriptorTable(UAV(u0,numDescriptors=8192)), RootConstants(num32BitConstants=3,b0)")]
        [numthreads(1,1,1)] void main() {
            uint2 entry = directory.Load2(page * 8);
            uint copy = entry.x & 0x7fffffff;
            if (copy) pages[copy].Store(entry.y + offset, marker);
        }"#;
    let code = crate::host::compile(source, s!("main"), s!("cs_5_1")).unwrap();
    let root: ID3D12RootSignature = unsafe { host.device.CreateRootSignature(0, &code) }.unwrap();
    let pso: ID3D12PipelineState = unsafe {
        host.device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
    }
    .unwrap();
    let record = |memory: &mut Memory, base: u32, dir_page: u64, marker: u32| {
        let list = host.new_list().unwrap();
        let (dir_start, dir_len) = memory.directory.1.span();
        memory.directory.1.refresh(&list, dir_start, dir_len);
        let dir_gpu = memory.directory.1.gpu(dir_start);
        let args = [dir_page as u32, (field - page) as u32, marker];
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRootShaderResourceView(0, dir_gpu);
            list.list
                .SetComputeRootDescriptorTable(1, heap.gpu_handle(base));
            list.list
                .SetComputeRoot32BitConstants(2, 3, args.as_ptr().cast(), 0);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        list
    };
    let backing_a = crate::shadow::by_address(page, PAGE).unwrap();
    let idx_a = memory.number(backing_a.clone()).unwrap();
    assert_eq!(idx_a, 0);
    let page_a = page / PAGE;
    fill(memory.entry(page_a), idx_a, 0);
    crate::watch::wrote(memory.entry(page_a) as usize, 8);
    let (base1, count1) = heap.raw_page_views(&memory.copies).unwrap();
    assert_eq!(count1, 1);
    let first = record(&mut memory, base1, page_a, first_marker);

    // Fill the initial reservation to capacity; numbering another backing must
    // grow the heap's raw-page run without invalidating `base1`.
    let cap = (FIRST_RAW_PAGE_SLOTS - 1) as usize;
    memory.copies.resize(cap, backing_a.clone());
    let second_page = page + PAGE;
    let second_mem = unsafe {
        VirtualAlloc(
            second_page as _,
            PAGE as usize,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    assert!(!second_mem.is_null());
    let backing_b = crate::shadow::by_address(second_page, PAGE).unwrap();
    let idx_b = memory.number(backing_b.clone()).unwrap();
    assert_eq!(idx_b, cap);
    let page_b = second_page / PAGE;
    fill(memory.entry(page_b), idx_b, 0);
    crate::watch::wrote(memory.entry(page_b) as usize, 8);
    let (base2, count2) = heap.raw_page_views(&memory.copies).unwrap();
    assert_eq!(count2 as usize, cap + 1);
    assert_ne!(base2, base1);
    let second = record(&mut memory, base2, page_b, second_marker);

    // Submit both only after the run grew so `first` exercises its older base.
    host.execute(&[&first]);
    host.execute(&[&second]);

    let readback = host.read_back_buffer(8).unwrap();
    let copy = host.new_list().unwrap();
    unsafe {
        copy.list
            .CopyBufferRegion(&readback, 0, &backing_a.buffer, field - page, 4);
        copy.list
            .CopyBufferRegion(&readback, 4, &backing_b.buffer, field - page, 4);
    }
    copy.close();
    host.execute(&[&copy]);
    let mut bytes = std::ptr::null_mut();
    let words = unsafe {
        readback
            .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 8 }), Some(&mut bytes))
            .unwrap();
        let words = bytes.cast::<[u32; 2]>().read();
        readback.Unmap(0, Some(&D3D12_RANGE::default()));
        words
    };
    assert_ne!(unsafe { VirtualFree(page as _, 0, MEM_RELEASE) }, 0);
    assert_ne!(unsafe { VirtualFree(second_page as _, 0, MEM_RELEASE) }, 0);
    assert_eq!(words, [first_marker, second_marker]);
}
