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
- `tl.static_range i in $(n) { ... }` and `tl.static_range i in N { ... }`
  are surface aliases for the same bounded-loop AST. Unroll and pipeline
  attributes are not modeled.
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
- `(x).to(tl.float64|tl.float32|tl.float16|tl.bfloat16)` is accepted as the
  method-style cast spelling. Parentheses around bare identifiers avoid
  Lean's hierarchical-name parser treating `x.to` as one identifier.

Supported channels:

- `.real`: modeled as `WithBot ℝ`, mainly to represent `-inf`.
- `.fp32`, `.fp16`, `.bf16`: explicit floating dtype channels, currently
  backed by the same `WithBot ℝ` mathematical carrier as `.real`.
- `.int`: AST-level signed mathematical-integer channel with
  arithmetic/comparison semantics. `tl.int32` and `tl.int64` both map here;
  bit width, overflow, and signed cast fidelity are not modeled yet.
- `.nat`: used for offsets, loop counters, sizes, and address arithmetic.
- `.bool`: produced by comparisons and consumed by masks / `tl.where`.

### Tile Construction and Shape Operations

- `tl.arange(n)` and `tl.arange(start, end)`.
  The two-argument form lowers to `start + tl.arange(end - start)`;
  `tl.arange(0, end)` collapses to `tl.arange(end)`.
  Numeric literals and `$(...)` Lean `Nat` meta-expressions are accepted for
  both bounds.
- `tl.full([dims...], value)`.
- `tl.zeros([dims...])`, sugar for `tl.full([dims...], 0)`.
- `e[:, None]` and `e[None, :]` for rank-1 inputs only.
  These lower to `Op.expandDim`.
- `tl.expand_dims(e, axis=N)` and `tl.expand_dims(e, N)` insert a unit axis
  at a literal position for any macro-known rank.
- `tl.trans(e)` swaps the trailing two axes, with any leading axes treated as
  a batch prefix.

### Arithmetic, Comparisons, and Broadcasting

- Arithmetic: `+`, `-`, `*`, `/` on numeric values.
  Mixed-channel arithmetic is rejected by the DSL.
- Integer floor division and remainder: `//` and `%` on `.nat` / `.int`.
- `tl.cdiv(x, y)` on `.nat`, lowered as `(x + y - 1) / y` with the current
  mathematical `Nat` semantics.
- Pointwise comparisons: `<`, `<=`, `==`, `>`, `>=`, `!=` on `.real` or
  `.nat`, producing `.bool`.
- Boolean ops: `tl.logical_and`, `tl.logical_or`, `tl.logical_not`, plus mask
  operator spellings `a & b`, `a | b`, and `~a` on `.bool` values.
- Two-argument `tl.max(a, b)` as pointwise max on `.real`.
- `tl.maximum(a, b)` and `tl.minimum(a, b)` as pointwise select-based sugar
  over comparable channels. Branch broadcasting is currently limited to
  scalar-to-tile lifting, matching `tl.where`.
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
- `tl.abs`

These operate on the `.real` channel. `tl.abs(x)` is desugared to
`tl.where(x < 0, 0 - x, x)` in the current real-valued semantics.

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

- `tl.load($(region))`
- `tl.load($(region) + offset)`
- `ptrs := $(region) + offset; tl.load(ptrs)`
- `tl.load(ptr, mask=mask)`
- `tl.load(ptr, mask=mask, other=other)`
- `tl.load(ptr, other=other, mask=mask)`
- `tl.load(ptr, dtype=tl.float32|tl.float16|tl.bfloat16|tl.int32|tl.int64|tl.uint32|tl.uint64)`
- `tl.load(ptr, mask=mask, other=other, dtype=...)`
- `bp := tl.make_block_ptr($(region), base=$(base), shape=[...],
  strides=[...], offsets=[...], block_shape=[...])`
- `bp2 := tl.advance(bp, [deltas...])`
- `tl.load(bp, boundary_check=([axes...] : List Nat), padding_option="zero")`

Supported stores:

- `tl.store($(region), value)`
- `tl.store($(region) + offset, value)`
- `ptrs := $(region) + offset; tl.store(ptrs, value)`
- `tl.store(ptr, value, mask=mask)`
- `tl.store(ptr, value, dtype=...)`, where `dtype` must match the value dtype.
  Without `dtype=`, stores infer the dtype from `value`.
- `tl.store(bp, value, boundary_check=([axes...] : List Nat))`

Unknown kwargs are rejected. For block pointers, `boundary_check` is supported
only on block-pointer `tl.load` / `tl.store`; it cannot be mixed with `mask` or
`other`. The only modeled `padding_option` is `"zero"`.

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

The runtime memory model is typed at the cell boundary:

```lean
RegionName → Nat → MemCell
```

The proof-facing Real compatibility API remains `BlockState.readMem` /
`BlockState.writeMem`, so existing mathematical correctness theorems keep their
Real-valued observation surface. In other words, the old theorem-facing
`RegionName → Nat → ℝ` view still exists as an API layer, but it is no longer
the runtime storage representation.

Region dtype contracts are modeled separately in `VeriTile.Triton.MemoryTyping`:

```lean
RegionTyping := RegionName → TileDType
Kernel.RespectsRegionTyping Γ k
```

This is a lightweight static layer. It checks that named-region loads/stores
use the dtype declared by `Γ`, while execution stores typed `MemCell`s. For
pointer-valued loads/stores, statically visible `$(region)` pointer bases are
checked against `Γ`; pointer registers are treated as dynamic/external values
in this layer.

Pointer values can be used inline, assigned, and reused:

```lean
ptrs := $(xReg) + offs
x := tl.load(ptrs)
ptrs2 := ptrs + stride
```

This is intentionally narrower than CUDA/Triton pointers: VeriTile supports
pointer base creation from a `RegionName`, pointer plus `.nat` offsets, and
load/store through pointer-valued registers. It also models block pointers as
first-class `.blockPtr` tile values carrying base region, base offset, parent
shape, block shape, strides, and logical offsets. Block-pointer load/store
computes each lane address from that layout; out-of-bounds checked load lanes
return zero, and out-of-bounds checked store lanes leave memory unchanged. It
does not yet model pointer casts, pointer comparison, hardware/TMA block-pointer
behavior, or a typed address space.

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

The core AST uses one dtype-indexed memory form:

```lean
Op.load    : TileDType → MemAccess shape → MaskOpt dtype shape → Op ...
Stmt.store : TileDType → MemAccess shape → Op ... → MaskOpt dtype shape → Stmt
```

The public DSL defaults `tl.load` to `.real`, but accepts `dtype=...` with
`tl.float32`, `tl.float16`, `tl.bfloat16`, `tl.int32`, `tl.int64`, `tl.uint32`,
or `tl.uint64` to produce typed memory nodes. `tl.uint64` and `tl.uint32` map
to VeriTile's `.nat` channel for nonnegative index/block-table values.
`tl.int32` and `tl.int64` map to VeriTile's `.int` channel, a mathematical
signed-integer abstraction with no bit-width or overflow semantics. `tl.store`
infers its dtype from the value being stored, with optional matching `dtype=`
syntax for Triton-like surface spelling.

Float theorem policy: algorithmic correctness/refinement theorems are proved
over erased `.real` kernels via `Kernel.AlgorithmCorrect` and
`Kernel.AlgorithmRefine`. A float-facing theorem uses erasure equations such as
`k.eraseFloat = realK` to expose a theorem for dtype-annotated kernels without
re-proving the algorithm in each floating channel. Computational
correctness/refinement is represented separately by `Kernel.ComputeCorrectAt?`
and `Kernel.ComputeRefineAt?`, epsilon-bound predicates over observed outputs.
That layer is currently supported by smoke and differential tests rather than
IEEE-754 proof. These definitions live in `VeriTile.Triton.Float`.

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
| Loops | Supported | Bounded `tl.for`; `tl.static_range` alias backed by the same loop AST |
| Conditionals | Limited | `tl.if cond { ... }` only; no `else`, `break`, or `continue` |
| Arithmetic | Supported | `+`, `-`, `*`, `/` on same-channel numeric operands; `//`, `%` on integer channels; `tl.cdiv` on `.nat`; `ptr + nat` for pointer offsets |
| Comparisons | Supported | `<`, `<=`, `==`, `>`, `>=`, `!=` on `.real` or `.nat` |
| Boolean ops | Supported | `tl.logical_and`, `tl.logical_or`, `tl.logical_not`, plus `&`, `|`, `~` mask spellings |
| Pointwise select | Supported | `tl.where(cond, a, b)` with scalar lifting and matching non-scalar shapes |
| Unary math | Supported | `tl.exp`, `tl.log`, `tl.sigmoid`, `tl.sqrt`, `tl.tanh` |
| Reductions | Supported | `tl.sum`, `tl.max`, optional `axis`, optional `keep_dims` over `.real` tiles |
| Broadcast | Supported | ND same-dim, scalar-to-tile, and dimension-`1` expansion |
| Shape construction | Limited | `tl.arange`, `tl.full`, `tl.zeros`, rank-1 `[:, None]` / `[None, :]`, literal-axis `tl.expand_dims` |
| Transpose | Limited | `tl.trans(e)` swaps trailing two axes only |
| Matrix multiply | Supported | `tl.dot(a, b)` and accumulator form `tl.dot(a, b, acc)` over mathematical `ℝ` |
| Loads | Limited | Pointer-expression load, optional `mask`, optional `other`, optional `dtype=` for float/int32/int64/uint32/uint64; block-pointer load with `boundary_check` and `padding_option="zero"` |
| Stores | Limited | Pointer-expression store, optional `mask`, dtype inferred from value with optional matching `dtype=`; block-pointer store with `boundary_check` |
| Tensor views | Supported | Strided `TensorView.loaded` / `TensorView.observe` wrappers for theorem statements |
| Integer memory | Limited | Typed cells plus typed load/store support Nat/index and mathematical signed-Int HBM values; no richer signed/unsigned width lattice yet |
| Randomness | Gap | No `tl.rand` or RNG state model yet (#41) |
| Indirection | Gap | No gather / paged-KV style data-dependent address model yet (#42) |
| Block pointers | Limited | `tl.make_block_ptr`, `tl.advance`, block-pointer load/store with checked-axis zero padding / store skip; no hardware/TMA behavior |
| Atomics / async / barriers | Gap | No `tl.atomic_*`, async copy, TMA, barriers, or scheduling semantics (#12) |
| Floating point fidelity | Gap | Real-valued model only; no IEEE-754 or mixed-precision hardware semantics (#11) |

## Expressiveness Matrix

This matrix answers a different question from the operator checklist: if a
user starts with a real Triton kernel, what kind of gap would block writing it
faithfully in the current Lean DSL?

| Pattern | Status | Gap type | Practical impact |
| --- | --- | --- | --- |
| Dense elementwise kernels | Mostly supported | Proof/theorem surface | Pointwise arithmetic, masks, casts, pointer values, and TensorView observation are available; richer dtype/memory claims still use the real abstraction. |
| Softmax / reductions / LayerNorm / Welford | Supported for current examples | Proof engineering | Core reductions, loops, masks, and TensorView wrappers exist; new kernels mainly need invariants and theorem packaging. |
| FlashAttention-style dense tiled kernels | Supported for FA-1 forward subset | Proof engineering + limited semantics | Dot, transpose, causal/boundary masks, D-tail, and 4D views are covered; async/shared-memory/hardware dot fidelity remain out of scope. |
| First-class pointer expressions | Limited | Surface + lightweight semantics | `ptrs := $(r) + offs`, pointer registers, pointer load/store, and pointer offset updates work for `RegionName × Nat`; no pointer casts/comparison/alias analysis. |
| Block pointers / `boundary_check` | Limited | Surface + sequential semantics | `tl.make_block_ptr`, `tl.advance`, zero-padded checked loads, and checked store-skip work; no `order`, non-zero padding, TMA, or hardware behavior. |
| Typed floating memory | Limited | Semantic abstraction | `dtype=tl.float32/fp16/bf16` creates typed floating nodes and erases to real for algorithm proofs; IEEE rounding is not modeled. |
| Integer / bool tensor memory | Limited | Dtype coverage | Typed cells plus typed load/store support Nat/index and mathematical signed-Int HBM values; no complete Triton integer-width lattice yet. |
| Indirect / gather addressing | Gap | Core addressing semantics (#42) | Blocks paged attention, embedding/table lookup, cross-entropy index lookup, and data-dependent pointer chasing. |
| RNG / dropout | Gap | State/probabilistic semantics (#41) | Blocks faithful dropout and stochastic kernels. |
| Atomics / async / shared memory / barriers | Gap | Concurrency semantics (#12) | Blocks production-style backward kernels, reductions using shared memory phases, async/TMA pipelines, and race/scheduling reasoning. |
| Whole-grid launch semantics | Gap | Execution model (#5) | Theorems currently quantify over one symbolic program instance through `BlockState.pids`; full launch coverage is manual. |
| Python/Triton source ingestion | Gap | Front-end/lifter (#10) | Users must write Lean `triton { ... }`; decorators, Python-side constexpr execution, and general Python control flow are not parsed. |
| Type checking / pointer provenance | Gap | Static analysis (#46) | DSL rejects many type mismatches syntactically, but there is no full checker for pointer provenance, block-pointer rank/stride consistency, or bounds assumptions. |

Recommended near-term priority for expressiveness is to remove core semantic
gaps before building a full Python lifter: typed/int/bool memory (#20),
pointer provenance/type checking (#46), and indirect addressing (#42). A
lifter is only useful for kernels whose operations are already representable.

## Unsupported or Not Yet Faithfully Modeled

- Full IEEE-754 floating-point semantics.
- Block-pointer hardware/TMA behavior and unsupported padding options beyond
  `"zero"`.
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
- Higher-rank bracket slicing beyond the currently supported rank-1
  `[:, None]` / `[None, :]` forms. Use `tl.expand_dims(e, axis=N)` for
  explicit unit-axis insertion.
- Integer widths, overflow, and signedness. The `.nat` channel is mathematical
  `Nat`.

## Documentation Generation

There is no automatic documentation generator for this subset yet. A useful
future tool would extract the raw DSL syntax / `Op` constructors into a table,
then compare it against this document. That would catch drift such as
"implemented but undocumented" or "documented but no longer accepted".

For now this document is manually maintained because the important artifact
claim is semantic, not just syntactic: the gaps above require human judgment.
