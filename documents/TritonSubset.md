# Supported Triton Subset and Semantic Gaps

This document records the Triton-like surface syntax currently embedded in
Lean by VeriTile, the semantic model behind it, and the gaps that are not yet
modeled. It is an artifact-facing contract: if a kernel uses syntax outside
this subset, either the DSL rejects it or the proof is not claiming Triton
semantic fidelity for that feature.

## Scope Summary

VeriTile models a typed Triton-style kernel language, not Python execution and
not full Triton IR. Kernels are written inside Lean using `triton { ... }` and
lower to typed AST nodes:

```lean
Op   : TileDType → TileShape → Type
Stmt : Type
```

The current shape model is ND tiles with outermost-first shape lists. Scalar
values have shape `[]`; a matrix `[M, D]` has index shape
`TileIndex [M, D]`.

## Supported Surface Syntax

### Control and Program IDs

- `tl.program_id(axis)` where `axis` is a numeric literal or `$(n)`.
  The runtime state stores `pids : Nat → Nat`, so every axis is total.
- `tl.for i in $(n) { ... }` and `tl.for i in N { ... }`.
  The loop is operationally modeled and proved through `forLoop_inv`.
- `tl.if cond { ... }` for scalar boolean conditions (`Op .bool []`).
  There is no `else`, `break`, or `continue`; Triton block-skipping patterns
  should be written by negating the skip condition and wrapping the useful
  body. Elementwise conditional selection remains `tl.where`.
- Register assignment:
  `x := expr`.

### Constants and Dtypes

- Real literals: `0`, `1`, `3.5`, etc. lower to the `.real` channel.
- Nat antiquotation: `$(n)` lowers to the `.nat` channel.
- Real antiquotation: `$ℝ(x)` lowers a Lean `ℝ` term to the `.real` channel.
- `-inf` lowers to `Op.negInf`, represented internally by `⊥ : WithBot ℝ`.
- `tl.toReal(x)` converts a `.nat` scalar/tile to `.real`.
- `tl.cast(x, tl.float64|tl.float32|tl.float16|tl.bfloat16)` changes the
  floating dtype index. In the current semantic model this preserves the
  underlying `WithBot ℝ` value; it does not model rounding.

Supported channels:

- `.real`: modeled as `WithBot ℝ`, mainly to represent `-inf`.
- `.fp32`, `.fp16`, `.bf16`: explicit floating dtype channels, currently
  backed by the same `WithBot ℝ` mathematical carrier as `.real`.
- `.int32`: AST-level signed-integer channel with arithmetic/comparison
  semantics; DSL surface casts to/from int32 are not modeled yet.
- `.nat`: used for offsets, loop counters, sizes, and address arithmetic.
- `.bool`: produced by comparisons and consumed by masks / `tl.where`.

### Tile Construction and Shape Operations

- `tl.arange(n)` and `tl.arange(start, end)`.
  The two-argument form lowers to `start + tl.arange(end - start)`;
  `tl.arange(0, end)` collapses to `tl.arange(end)`.
- `tl.full([dims...], value)`.
- `tl.zeros([dims...])`, sugar for `tl.full([dims...], 0)`.
- `e[:, None]` and `e[None, :]` for rank-1 inputs only.
  These lower to `Op.expandDim`.
- `tl.trans(e)` swaps the trailing two axes, with any leading axes treated as
  a batch prefix.

### Arithmetic, Comparisons, and Broadcasting

- Arithmetic: `+`, `-`, `*`, `/` on `.real` or `.nat` values.
  Mixed-channel arithmetic is rejected by the DSL.
- Pointwise comparisons: `<`, `<=`, `==`, `>`, `>=`, `!=` on `.real` or
  `.nat`, producing `.bool`.
- Boolean conjunction: `tl.logical_and(a, b)` on `.bool` values.
- Two-argument `tl.max(a, b)` as pointwise max on `.real`.
- Broadcasting is ND and follows the current `Broadcast` witness:
  same dimension, scalar-to-tile, or dimension `1` expanded to the other side.
  The DSL constructs the broadcast proof syntactically, so equivalent but
  non-identical dimension expressions may still need to be written in a
  matching form.
- Elementwise selection: `tl.where(cond, a, b)`.
  The condition must be `.bool`, the branches must have the same dtype, and
  scalar-to-tile lifting is accepted. Non-scalar operands must already have
  the same surface shape; use the supported unit-axis slicers to make shapes
  agree before calling `tl.where`.

### Unary Math

- `tl.exp`
- `tl.log`
- `tl.sigmoid`
- `tl.sqrt`
- `tl.tanh`

These operate on the `.real` channel.

### Reductions

- `tl.sum(x)`
- `tl.max(x)`
- `tl.sum(x, axis=N)`
- `tl.max(x, axis=N)`
- `keep_dims=true|false`

Omitted `axis` follows Triton's `axis=None` behavior: reduce over all
dimensions. Explicit `axis=N` reduces one axis. Reductions are currently over
`.real` tiles.

### Matrix Operations

- `tl.dot(a, b)`.
  It operates on the trailing two dimensions:
  `[..., M, K] × [..., K, N] → [..., M, N]`.
- `tl.dot(a, b, acc)` is accepted as the Triton accumulator form and lowers to
  `acc + tl.dot(a, b)`.
- `tl.trans(e)` is the trailing-two-axis transpose used by `tl.dot(Q, Kᵀ)`.

The current `tl.dot` model is mathematical real matrix multiplication over the
`.real` abstraction, not Triton's hardware-specific dot instruction semantics.

### Memory Operations

Supported loads:

- `tl.load(tl.ptr($(region)))`
- `tl.load(tl.ptr($(region)) + offset)`
- `ptrs := tl.ptr($(region)) + offset; tl.load(ptrs)`
- `tl.load(ptr, mask=mask)`
- `tl.load(ptr, mask=mask, other=other)`
- `tl.load(ptr, other=other, mask=mask)`

Supported stores:

- `tl.store(tl.ptr($(region)), value)`
- `tl.store(tl.ptr($(region)) + offset, value)`
- `ptrs := tl.ptr($(region)) + offset; tl.store(ptrs, value)`
- `tl.store(ptr, value, mask=mask)`

Unknown kwargs are rejected. In particular, `tl.load(..., boundary_check=...)`
and `tl.store(..., boundary_check=...)` are not silently ignored.

Masked load semantics:

- If `mask` is true, read memory.
- If `mask` is false and `other` is supplied, return `other`.
- If `mask` is false and `other` is omitted, Triton leaves the value
  undefined; VeriTile models this through `BlockState.undef`.

Masked store semantics:

- If `mask` is true, write the lane.
- If `mask` is false, leave memory unchanged.

## Memory Model

VeriTile models a limited first-class pointer value as:

```lean
RegionName × Nat
```

The memory model remains:

```lean
RegionName → Nat → ℝ
```

Pointer values can be used inline, assigned, and reused:

```lean
ptrs := tl.ptr($(xReg)) + offs
x := tl.load(ptrs)
ptrs2 := ptrs + stride
```

This is intentionally narrower than CUDA/Triton pointers: VeriTile supports
pointer base creation from a `RegionName`, pointer plus `.nat` offsets, and
load/store through pointer-valued registers. It does not yet model pointer
casts, pointer comparison, block pointers, or a typed address space.

Offsets are explicit `.nat` expressions. For higher-dimensional tensors, the
user supplies strided offset formulas such as:

```lean
b * stride_b + h * stride_h + i * stride_s + d * stride_d
```

Public theorem surfaces use `TensorView.loaded` / `TensorView.observe` to
connect those formulas to mathematical tensor slices. Internally, proofs may
still use the lower-level `InputAt` escape hatch for arbitrary offset maps and
then package the result as a `TensorView`. Aliasing is represented by choosing
equal or distinct `RegionName`s; arbitrary pointer alias analysis beyond those
named regions is not modeled. See
[`GpuMemoryModel.md`](./GpuMemoryModel.md) for the GPU memory hierarchy scope
and the sequential-consistency assumptions.

## Floating-Point Model

Arithmetic is currently an `ℝ` abstraction:

- Real data is modeled as `WithBot ℝ`.
- `-inf` is modeled exactly as `⊥`.
- `exp ⊥ = 0` and `sigmoid ⊥ = 0` are built into the semantics.
- Memory stores demote `⊥` with a default value; well-formed kernels should not
  store `⊥`.

What this means: theorems prove real-valued mathematical correctness, not
bit-level IEEE-754 equivalence. Rounding, NaNs, signed zeros, overflow,
underflow, denormals, exception flags, hardware dot precision, and fast-math
rewrites are not modeled.

The core AST has typed floating load/store constructors such as
`Op.loadFloat`, `Op.loadPtrFloat`, and `Stmt.storeFloat`. The public DSL still
defaults `tl.load` / `tl.store` to `.real`; use `tl.cast` where a kernel needs
to carry an explicit floating dtype through intermediate registers.

Float theorem policy: algorithmic correctness theorems should be proved over
the erased `.real` kernel. A float-facing theorem can use
`Kernel.CorrectViaFloatErasure` plus an erasure equality
`k.eraseFloat = realK` to expose a theorem for the dtype-annotated kernel
without re-proving the algorithm in each floating channel.

## Operator and Syntax Coverage Checklist

This table is the current operator-coverage contract for GitHub issue #15.
`Supported` means the syntax has a Lean AST constructor or accepted DSL
lowering, operational semantics, and at least the proof surface needed by the
current examples. `Limited` means VeriTile has a deliberately narrow version
of the Triton feature. `Gap` means kernels using the feature are outside the
current semantic contract.

| Area | Status | Coverage |
| --- | --- | --- |
| Scalar/tile constants | Supported | Real literals, `$(n)`, `$ℝ(x)`, `-inf`, register refs |
| Program IDs | Limited | `tl.program_id(axis)` for literal or antiquoted `Nat` axes; no whole-grid execution semantics |
| Loops | Supported | Bounded `tl.for`, backed by `forLoop_inv` |
| Conditionals | Limited | `tl.if cond { ... }` only; no `else`, `break`, or `continue` |
| Arithmetic | Supported | `+`, `-`, `*`, `/` on same-channel `.real` or `.nat` operands; `ptr + nat` for pointer offsets |
| Comparisons | Supported | `<`, `<=`, `==`, `>`, `>=`, `!=` on `.real` or `.nat` |
| Boolean ops | Limited | `tl.logical_and`; no general boolean operator family |
| Pointwise select | Supported | `tl.where(cond, a, b)` with scalar lifting and matching non-scalar shapes |
| Unary math | Supported | `tl.exp`, `tl.log`, `tl.sigmoid`, `tl.sqrt`, `tl.tanh` |
| Reductions | Supported | `tl.sum`, `tl.max`, optional `axis`, optional `keep_dims` over `.real` tiles |
| Broadcast | Supported | ND same-dim, scalar-to-tile, and dimension-`1` expansion |
| Shape construction | Limited | `tl.arange`, `tl.full`, `tl.zeros`, rank-1 `[:, None]` / `[None, :]` |
| Transpose | Limited | `tl.trans(e)` swaps trailing two axes only |
| Matrix multiply | Supported | `tl.dot(a, b)` and accumulator form `tl.dot(a, b, acc)` over mathematical `ℝ` |
| Loads | Limited | Pointer-expression load, optional `mask`, optional `other` |
| Stores | Limited | Pointer-expression store, optional `mask` |
| Tensor views | Supported | Strided `TensorView.loaded` / `TensorView.observe` wrappers for theorem statements |
| Integer memory | Gap | Memory is `RegionName → Nat → ℝ`; no int/Nat tensor memory yet (#20) |
| Randomness | Gap | No `tl.rand` or RNG state model yet (#41) |
| Indirection | Gap | No gather / paged-KV style data-dependent address model yet (#42) |
| Block pointers | Gap | No `tl.make_block_ptr`, `tl.advance`, or block-pointer load/store |
| Atomics / async / barriers | Gap | No `tl.atomic_*`, async copy, TMA, barriers, or scheduling semantics (#12) |
| Floating point fidelity | Gap | Real-valued model only; no IEEE-754 or mixed-precision hardware semantics (#11) |

## Unsupported or Not Yet Faithfully Modeled

- Full IEEE-754 floating-point semantics.
- Triton block pointers and block-pointer-only restrictions.
- `boundary_check` / `padding_option`.
- Full CUDA/Triton pointer semantics beyond `RegionName × Nat` pointer values.
- Arbitrary pointer alias analysis beyond named-region equality.
- General Python/Triton JIT semantics, decorators, meta-parameter execution,
  and Python control flow outside the embedded `triton { ... }` block.
- Atomic operations.
- Async copy / TMA / shared-memory staging.
- Barriers and inter-program or inter-warp synchronization.
- Grid launch semantics. A theorem describes one symbolic program instance
  with `BlockState.pids`; whole-grid coverage is expressed manually by
  quantifying over valid program IDs.
- Caches and performance hints such as `cache_modifier`, `eviction_policy`,
  `volatile`, or `is_volatile`.
- Arbitrary axis permutation. `tl.trans` only swaps trailing two axes.
- Higher-rank surface slicing beyond the currently supported rank-1
  `[:, None]` / `[None, :]` forms.
- Integer widths, overflow, and signedness. The `.nat` channel is mathematical
  `Nat`.

## Documentation Generation

There is no automatic documentation generator for this subset yet. A useful
future tool would extract the raw DSL syntax / `Op` constructors into a table,
then compare it against this document. That would catch drift such as
"implemented but undocumented" or "documented but no longer accepted".

For now this document is manually maintained because the important artifact
claim is semantic, not just syntactic: the gaps above require human judgment.
