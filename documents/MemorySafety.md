# Layer-1 Memory Bounds Safety

`VeriTile.Triton.MemoryBounds` defines the lightweight bounds-safety layer for
Triton memory operations:

```lean
abbrev RegionBounds := RegionName -> Nat

def Op.MemorySafe (bounds : RegionBounds) : Op dtype shape -> Prop
def Stmt.MemorySafe (bounds : RegionBounds) : Stmt -> Prop
def Kernel.MemorySafe (bounds : RegionBounds) (k : Kernel) : Prop
def ComputeKernel.MemorySafe (bounds : RegionBounds) (ck : ComputeKernel) : Prop
```

The public predicate is state-independent, but memory operations internally
quantify over all `BlockState`s. This is necessary because masks and
first-class pointers are ordinary expressions whose active lanes and dynamic
addresses are known only after evaluation.

## Active Lanes

Loads and stores are checked only on lanes that can perform a memory access.

- `MaskOpt.none`: every lane is active.
- `MaskOpt.mask m`: only lanes where `m` evaluates to `true` are active.
- `MaskOpt.maskOther m other`: address safety follows the same active lanes as
  `mask`; inactive lanes take `other` and do not read memory.
- Block-pointer `boundary_check`: lanes whose `BlockPtr.inBounds` check is
  false are safe vacuously. Checked loads return zero/default padding and
  checked stores skip the memory write.

This matches the sequential operational semantics in `EvalOp.lean` and
`Step.lean`: inactive lanes do not issue a memory read/write that needs a region
bound.

## Addressing Forms

The predicate covers the three memory-addressing forms in the AST:

- Direct region offsets, `MemAccess.region region off`.
- First-class pointer values, `MemAccess.ptr ptr`, using an operational
  `Op.PointerAddressesSafeOn` predicate over evaluated `(RegionName, Nat)`
  pointer lanes.
- Block pointers, `MemAccess.blockPtr ptr boundaryCheck`, where checked
  out-of-bounds lanes are excluded from the bound obligation.

Pointer-register provenance is intentionally not solved here. The optional
checker in `VeriTile.Triton.MemoryTyping` can later provide sufficient
conditions for dynamic pointer-address safety; the bounds layer stays semantic
and composable.

## Non-Goals

This layer does not model or prove:

- Race freedom or cross-program memory composition.
- Frame theorems or ownership/permission accounting.
- Aliasing or page ownership.
- Atomics, barriers, async copy, shared memory, or scheduling semantics.
- Hardware-specific memory hierarchy behavior.

Those are later layers in the memory-model roadmap. `Kernel.MemorySafe` is only
the per-program active-lane region-bounds contract.

Layer 2a is implemented separately in `VeriTile.Triton.MemoryFrame`. It adds
predicate-level write footprints and `BlockState.WriteWithin` frame contracts
for single-program executions. Layer 2b lives in
`VeriTile.Triton.Launch.Composition`: it merges explicit per-program
`Kernel.ExecFrame`s when their write footprints are pairwise disjoint.
Structured footprint extraction and proof automation are tracked by #61.

## Examples

`VeriTile/Examples/MemorySafety.lean` contains representative proofs:

- `straightLineCopy_memorySafe`: direct region+offset load/store.
- `maskedTailAdd_memorySafe`: masked inactive lanes are safe vacuously.
- `blockPtrBoundary_memorySafe`: `boundary_check` excludes checked
  block-pointer out-of-bounds lanes.
