---
title: Translating a kernel
description: When a .py → .lean transcription is faithful, what gaps are tolerated, and which deviations must be fixed before merge.
---

The gold standard is
[`bench/tritonbench_g/add_example/AddExample.lean`](https://github.com/Lizn-zn/VeriTile/blob/main/bench/tritonbench_g/add_example/AddExample.lean):
the `def kernel ... : ComputeKernel := triton { ... }` body is the `.py`
kernel function body **line-for-line**, no extra statements, no merges, no
silent fixups.

The reason for that strictness is the project's mental model: users come
expecting to paste their `.py` kernel in and read the proof. Any implicit
simplification or merge turns the proof into "VeriTile's algebraic
rearrangement of this kernel" rather than "this kernel".

Use the categories below when reviewing a new translation (or when writing
one — they tell you what to push back against).

## ✓ Faithful

Line-for-line correspondence between `kernel.py` and the `triton { ... }`
body. No extra statements, no merges, no reordering. The matrix of `tl.*`
calls is the same; only the lexical wrapping (Lean syntax) differs.

## ⚠️ Minor (tolerated) gaps

Two classes count as MINOR — they don't block merge but should be noted.

### 1. Lean-syntax forced changes — whitelist only

Only these are tolerated as Lean-syntax forced changes:

- **`$(x)` interpolation** to bridge Lean parameters or compile-time
  constants (including `BLOCK_SIZE: tl.constexpr` modelled as a `Nat`
  parameter used as `$(BLOCK_SIZE)`).
- **`if HAS_X { ... }` DSL gate** in place of Python's `if HAS_X:` for
  constexpr branching.

Everything else outside the whitelist is a **deviation**, including:

- Adding an explicit `dtype=...` to a `tl.load` where the `.py` didn't.
- `keep_dims=true` in place of `[:, None]`.
- `1 / tl.sqrt(x)` in place of `tl.math.rsqrt(x)`.
- `(0.0).to(tl.float32)` explicit casts.
- Pointer mutation `X += offset` rewritten as `base + offset` at each use.
- `x += rhs` rewritten as `x = x + rhs`.
- "Real-first policy" silently erasing `.to(tl.float32)`.

If the DSL can't express what the `.py` wrote, that's a DSL gap — open an
issue against the DSL, don't paper over it by rewriting the kernel body.

### 2. Documented specialization or slice

A documented partial specialization is tolerated, but the preamble doc
comment must say so explicitly. Examples:

- "proof-oriented one-block slice" / "single-tile" / "one-iteration" —
  upstream is a loop, Lean only proves a single iteration.
- Single-constexpr-branch specialization (`IS_RMS_NORM=true`,
  `HAS_BIAS=false`, `LOG=false`, …).
- Single-output-channel slice (e.g. `FifthOrderSphHarmonics` taking only
  `Y00`).
- Unused-parameter elision (`*_row_stride` in a 1D path).

The preamble must say "this is a slice / specialization", not "this is the
equivalent form of the full kernel".

## ✗ Deviates — must be fixed

The following are not gaps; they are bugs in the translation.

### Added a statement the `.py` does not have

Any extra `tl.where`, extra cast, extra mask handling beyond what the `.py`
writes. Example (real): `RmsnormFused.lean` once introduced an extra
`x_masked = tl.where(cols < N, x_for_var, 0.0)` while the `.py` handled the
same padding via `tl.load(..., other=0.0)`. Both are equivalent on paper;
the proof now talks about a *different* kernel.

### Added arithmetic structure the `.py` does not have

Stride coefficients, broadcast dims, or scale factors added to an address
or value that the `.py` didn't write. Example: changing
`tl.load(W + col_offsets, ...)` to `tl.load(W + col_offsets * W_row_stride, ...)`.
When `W` has a non-trivial stride, results differ — even when the change
"looks more correct".

### Folded a reduction loop into a single `tl.sum`

Upstream is `for off in range(0, N, BLOCK_N): var += xf*xf`; the Lean
translation writes `var = tl.sum(x*x, axis=0)`. Even when mathematically
equivalent, the body shape is different and the proof "proves something
that isn't this kernel". Two acceptable fixes:

- Put the `for` loop back (faithful — but the proof has to change).
- Declare it in the preamble as a specialization: "the N-block accumulation
  loop is folded to a single-tile `tl.sum`, requiring `BLOCK ≥ N`."

### Reordered compute statements

Even if data flow is equivalent, 1:1 line correspondence is lost.

### Merged or omitted `tl.*` calls without documenting

Chained cast where the `.py` did two — collapsed into one. Masked load
quietly downgraded to a bare load. Same family as the above.

### Changed control flow

Python multi-branch `if/elif/else` collapsed into `tl.where`; `for`
unrolled; `while` rewritten as `for`. All require explicit documentation
if intentional.

## Suggested audit format

When auditing a translation (or self-reviewing), classify each file:

| Verdict | Meaning |
|---|---|
| `✓ FAITHFUL` | Line-by-line 1:1. |
| `⚠️ MINOR` | One of the tolerated gaps above (cite the preamble doc comment for category 2). |
| `❌ DEVIATES` | One of the must-fix items. Give `.py` line + `.lean` line + a one-line description of the divergence. |

The audit is parallelizable — batches of ~15 kernels per agent works well.

## What surface to write

Once translation is settled, pick a theorem surface. The chooser is in
[Correctness surfaces](/VeriTile/proofs/correctness-surfaces/); a one-glance summary:

| Goal | Surface |
|---|---|
| One kernel matches an output spec | `ComputeCorrect.Realizes` |
| Two kernels satisfy an output relation | `ComputeRefine.Realizes` |
| Value + index output (e.g. `return_indices=True`) | `ComputeCorrect.OutputPairWhere` |
| Custom final-state postcondition | `ComputeCorrect.Post` / `ComputeRefine.Post` |
| Relation over arbitrary initial states | `ComputeCorrect.General` / `ComputeRefine.General` |
