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

Supported channels:

- `.real`: modeled as `WithBot ℝ`, mainly to represent `-inf`.
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
- Two-argument `tl.max(a, b)` as pointwise max on `.real`.
- Broadcasting is ND and follows the current `Broadcast` witness:
  same dimension, scalar-to-tile, or dimension `1` expanded to the other side.
  The DSL constructs the broadcast proof syntactically, so equivalent but
  non-identical dimension expressions may still need to be written in a
  matching form.

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

- `tl.load($(region))`
- `tl.load($(region) + offset)`
- `tl.load(ptr, mask=mask)`
- `tl.load(ptr, mask=mask, other=other)`
- `tl.load(ptr, other=other, mask=mask)`

Supported stores:

- `tl.store($(region), value)`
- `tl.store($(region) + offset, value)`
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

VeriTile does not model first-class Triton/CUDA pointers. The surface syntax
looks pointer-like:

```lean
tl.load($(xReg) + offs)
tl.store($(outReg) + offs, value)
```

but `$(xReg)` is a Lean `RegionName`, not a pointer value. The memory model is:

```lean
RegionName → Nat → ℝ
```

Offsets are explicit `.nat` expressions. For higher-dimensional tensors, the
user supplies strided offset formulas such as:

```lean
b * stride_b + h * stride_h + i * stride_s + d * stride_d
```

Public theorem surfaces use `TensorView.loaded` / `TensorView.observe` to
connect those formulas to mathematical tensor slices. Internally, proofs may
still use the lower-level `InputAt` escape hatch for arbitrary offset maps and
then package the result as a `TensorView`. Aliasing is represented by choosing
equal or distinct `RegionName`s; pointer values, pointer casts, pointer
comparison, and block pointers are not modeled.

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

## Unsupported or Not Yet Faithfully Modeled

- Full IEEE-754 floating-point semantics.
- Triton block pointers and block-pointer-only restrictions.
- `boundary_check` / `padding_option`.
- First-class pointers and pointer-valued expressions.
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
