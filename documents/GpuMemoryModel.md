# GPU Memory Modeling Scope

This document states what VeriTile means when it talks about "GPU memory".
It is a functional-correctness model for the supported Triton subset, not a
microarchitectural model of CUDA hardware.

## Modeled Layers

| GPU concept | VeriTile model | Notes |
| --- | --- | --- |
| Global memory / HBM | `BlockState.mem : RegionName -> Nat -> MemCell` | Target of `tl.load` and `tl.store`. `RegionName` separates named buffers; `Nat` is a cell offset, not a byte address. Proof-facing Real contracts use `readMem` / `writeMem`. |
| Block-local variables | `BlockState.regs : RegFile` | Logical Triton SSA/register values created by assignments. This is not the physical CUDA register file. |
| Addressing metadata | `TensorView`, `Offset.strided`, `InputAt` | Connects logical tensors to global-memory offsets for theorem statements and proof internals. |

The main theorem-facing surface is:

```lean
TensorView.loaded s view tensor
TensorView.observe sf view idx
```

`TensorView` is metadata: it records a region, base offset, and per-axis
strides. It does not store data. Data lives in `BlockState.mem`, while existing
Real-valued theorem statements observe it through `BlockState.readMem`.

## Addressing Model

VeriTile currently supports strided-affine tensor views:

```text
addr = base + sum_i idx_i * stride_i
```

This covers the layouts used by the current FA-1 4D views and the usual
contiguous/strided examples. Lower-level proofs can still use `InputAt` with
an arbitrary offset function when a helper has not been packaged as a
`TensorView`.

Partially modeled:

- First-class pointer values for `RegionName × Nat`.
- Triton-style block pointer values for `tl.make_block_ptr` / `tl.advance` and
  checked block-pointer load/store with zero padding / store skip.

Not modeled yet:

- Pointer casts, pointer comparison, or pointer alias analysis beyond
  `RegionName` equality.
- Hardware/TMA block-pointer behavior beyond the sequential lane semantics.
- Full paged-KV proof infrastructure. The storage model now supports the
  core indirect-addressing pattern — typed index loads feeding pointer
  arithmetic — and `IndirectView` packages the read-only view layer. Bounds,
  alias/page-ownership, and paged FA-1 equivalence proofs are still future
  consumer work; see issue #42.
- Rich signed/unsigned integer dtype lattice. The current typed HBM model
  supports `.nat` (`tl.uint8/uint16/uint32/uint64`) and mathematical `.int`
  (`tl.int8/int16/int32/int64`), enough for index/block-table cells, but not
  bit-width-specific integer semantics.

## Sequential Consistency

Within one symbolic program instance, memory is sequentially consistent:

- a `tl.store` updates `BlockState.mem`;
- a later `tl.load` from the same `RegionName` and offset observes that update;
- masked stores leave masked-off lanes unchanged;
- masked loads with `other=None` use `BlockState.undef` for masked-off lanes.

This is the right abstraction for current per-program proofs. Whole-grid
execution is only modeled as a theorem surface: `GridIndex` instantiates
`BlockState.pids`, and `Kernel.ForAllPrograms` / `ForAllProgramsSome` quantify
over every program instance in an ND grid. VeriTile does not yet model a
sequential or concurrent launch executor, global memory merge, overlapping
writes, races, atomics, or scheduling.

Layer-2a frame reasoning is modeled as a predicate-level proof contract:
`WriteFootprint := (RegionName × Nat) → Prop` and `BlockState.WriteWithin`
state that a single-program execution changed only the cells inside a supplied
footprint. Layer-2b adds `Kernel.mergeFrames`: an extensional, disjoint
whole-grid merge over explicit per-program `Kernel.ExecFrame`s. This is still
not a concurrent/interleaved executor; overlapping writes, atomics, scheduling,
barriers, async, and shared memory remain outside the model.

## Not Modeled

The following hardware layers and performance effects are intentionally outside
the current semantic contract:

- Shared memory / SMEM allocation and bank conflicts.
- L1/L2 caches, cache modifiers, eviction policy, volatile memory behavior, and
  coalescing.
- Physical register allocation, register pressure, spilling, occupancy, warps,
  lanes, or scheduling.
- Tensor Core / WGMMA instruction behavior and mixed-precision accumulation.
- Async copy, TMA, barriers, fences, or inter-program synchronization.
- Atomics and cross-block memory races.

These omissions mean VeriTile proves real-valued functional correctness for a
single symbolic Triton program instance. It does not prove performance
properties or CUDA memory-system fidelity.

## Extension Path

The current model is intentionally small. The landed memory-proof layers are:

- **Memory safety / bounds (#48):** active-lane region-bounds contracts on top
  of the current typed storage layer.
- **Write footprints / frame (#60):** predicate-level `WriteFootprint` and
  `BlockState.WriteWithin` contracts for single-program frame reasoning.
- **Disjoint whole-grid composition (#49):** `Kernel.mergeFrames` merges
  explicit per-program `Kernel.ExecFrame`s when their write footprints are
  pairwise disjoint.

The next proof-ergonomics layer is:

- **Structured footprint extraction (#61):** `WriteFootprint.tileImage`,
  `activeTileImage`, and address-image helpers derive predicate footprints from
  common direct, masked, checked block-pointer, and regular store patterns.
- **Unrelated-frame helpers (#62):** convenience lemmas prove that cells or
  whole regions outside a single-program or grid footprint are preserved.

Longer-term extension points remain:

- **Paged KV / indirect addressing (#42):** extend the current `IndirectView`
  smoke/proof surface into paged-attention-specific logical views and
  consumer-side equivalence theorems.
- **Async and concurrency (#12):** `ConcurrencySemantics.md` defines the
  boundary for shared-memory state, barriers, atomics, async/TMA, and explicit
  scheduling or trace models.
- **Floating-point fidelity (#11):** replace or refine the `R` abstraction with
  IEEE / mixed-precision semantics where needed.
