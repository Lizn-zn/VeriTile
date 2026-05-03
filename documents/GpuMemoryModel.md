# GPU Memory Modeling Scope

This document states what VeriTile means when it talks about "GPU memory".
It is a functional-correctness model for the supported Triton subset, not a
microarchitectural model of CUDA hardware.

## Modeled Layers

| GPU concept | VeriTile model | Notes |
| --- | --- | --- |
| Global memory / HBM | `BlockState.mem : RegionName -> Nat -> ℝ` | Target of `tl.load` and `tl.store`. `RegionName` separates named buffers; `Nat` is a cell offset, not a byte address. |
| Block-local variables | `BlockState.regs : RegFile` | Logical Triton SSA/register values created by assignments. This is not the physical CUDA register file. |
| Addressing metadata | `TensorView`, `Offset.strided`, `InputAt` | Connects logical tensors to global-memory offsets for theorem statements and proof internals. |

The main theorem-facing surface is:

```lean
TensorView.loaded s view tensor
TensorView.observe sf view idx
```

`TensorView` is metadata: it records a region, base offset, and per-axis
strides. It does not store data. Data lives in `BlockState.mem`.

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
- Paged-KV or gather-style data-dependent indirection. That needs a sibling
  view model on top of the same storage layer; see issue #42.
- Typed non-real memory. The storage model is currently real-valued; see
  issue #20.

## Sequential Consistency

Within one symbolic program instance, memory is sequentially consistent:

- a `tl.store` updates `BlockState.mem`;
- a later `tl.load` from the same `RegionName` and offset observes that update;
- masked stores leave masked-off lanes unchanged;
- masked loads with `other=None` use `BlockState.undef` for masked-off lanes.

This is the right abstraction for current single-program proofs. Whole-grid
execution is not modeled: theorem statements quantify over symbolic program
IDs through `BlockState.pids`, and grid coverage is proved manually when
needed.

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

The current model is intentionally small. The likely extension points are:

- **Typed memory (#20):** generalize storage beyond `RegionName -> Nat -> ℝ` so
  integer/Nat tensors can be loaded and stored.
- **Paged KV / indirect addressing (#42):** add a gathered or paged view layer
  for data-dependent address maps.
- **Async and concurrency (#12):** introduce shared-memory state, barriers,
  atomics, and an explicit scheduling or trace model.
- **Floating-point fidelity (#11):** replace or refine the `R` abstraction with
  IEEE / mixed-precision semantics where needed.
