# TritonBench-G Translation Review Criteria

**English** | [中文](review_criteria_zh.md)

What gaps are tolerable, and what must be fixed, when reviewing
`bench/tritonbench_g/<kernel>/<Kernel>.lean` against its upstream
`<kernel>.py`.

## Gold standard

`bench/tritonbench_g/add_example/AddExample.lean` is the canonical reference.

A faithful translation has two parts:

1. The `def kernel ... : ComputeKernel := triton { ... }` body is a
   **line-for-line 1:1 transcription** of the upstream `.py` Triton kernel
   function body. No extra statements, no omitted statements, no
   reordering, no merging.
2. Correctness uses the standard `ComputeCorrect.Realizes` form.

Why this bar is high: VeriTile's user-facing promise is "paste your `.py`
kernel into Lean and the proof is about *that* kernel." Every implicit
simplification, merge, or padding turns the translation into "VeriTile's
kernel" instead of "your kernel."

## Tolerable gaps (⚠️ MINOR — does not block)

### A. Lean-syntax forced changes (whitelist, exhaustive)

Only the following two transformations are accepted as Lean-syntax-only:

- `$(x)` interpolation around Lean-level parameters or compile-time
  constants. Includes the structural `BLOCK_SIZE: tl.constexpr` →
  `Nat` parameter rewrite, with `$(BLOCK_SIZE)` at use sites.
- `if HAS_X { ... }` DSL gate replacing Python `if HAS_X:` for
  `tl.constexpr` branches.

**Everything else is not in this whitelist** and falls under DEVIATES
(see §1–§6 below). In particular, the following transformations are
**not** accepted as mechanical:

- Explicit `dtype=...` annotation on loads when `.py` doesn't have it.
- `tl.max(x, axis=1, keep_dims=true)` substituted for
  `tl.max(x, axis=1)[:, None]`.
- `1 / tl.sqrt(x)` substituted for `tl.math.rsqrt(x)`.
- `(0.0).to(tl.float32)` explicit cast for type inference.
- Pointer mutation `X += offset` rewritten as explicit `base + offset`.
- `+=` rewritten as `= + ...`.
- `.to(tl.float32)` erased under any "Real-first policy."

If the DSL cannot express what `.py` writes, that is a **DSL gap to file
as an issue**, not a green light to rewrite the kernel body. Translations
must match `.py` exactly; tolerated mechanical changes are limited to the
whitelist above.

### B. Documented partial slice / specialization

These are allowed only when the kernel preamble doc-comment **explicitly
states** the scope. The preamble must not claim equivalence to the full
upstream kernel.

- "proof-oriented one-block slice" / "single-tile slice" /
  "single-iteration slice": upstream loops, Lean proves one iteration
  only.
- One `tl.constexpr` branch specialization (e.g. `IS_RMS_NORM=true`,
  `HAS_BIAS=false`, `LOG=false`).
- Single output channel (e.g. `FifthOrderSphHarmonics` covers `Y00`
  only).
- Unused parameters dropped (e.g. `*_row_stride` in a 1D path with no
  caller).

## Must-fix gaps (❌ DEVIATES)

### 1. Statement added that's not in `.py`

Any extra `tl.where`, extra cast, extra mask-handling line that
`.py` doesn't have. This violates "one line more, one line less."

Example: `RmsnormFused.lean` had a separate `x_masked = tl.where(cols < N,
x_for_var, 0.0)` while `.py` handled padding via
`tl.load(..., other=0.0)`. The `tl.where` is an extra statement.

### 2. Arithmetic structure added that's not in `.py`

Stride coefficients, broadcast dimensions, scale factors that the
upstream load/store doesn't have. Even if "more correct under some
configurations," it's a real semantic divergence.

Example: `FastRmsLayernorm.lean` had `tl.load(W + col_offsets *
W_row_stride, ...)` where `.py` had `tl.load(W + col_offsets, ...)`. The
two computations diverge whenever `W` has a non-trivial stride.

### 3. Reduction loop folded into a single `tl.sum` / `tl.max`

Upstream has `for off in range(0, N, BLOCK_N): var += xf*xf`; Lean writes
`var = tl.sum(x*x, axis=0)`. Even when math-equivalent, the kernel body
structure changes — the proof now talks about a different kernel.

Two acceptable fixes:
- Restore the `for` loop (faithful — proof must adapt).
- Acknowledge in preamble: "Specializes to single-tile, folding the
  N-block accumulation loop into one `tl.sum`; requires `BLOCK ≥ N`."

### 4. Compute statement reordering

Even with equivalent dataflow, losing the 1:1 line correspondence
breaks faithfulness.

### 5. Unmodified omission of `tl.*` calls

Skipping a chained cast, collapsing a masked-load to a bare load, etc.,
without preamble documentation.

### 6. Control-flow change

Folding Python multi-branch `if` into `tl.where`, unrolling a `for`
loop, converting `while` into `for`, without explicit preamble doc.

## Review output format

Per file, one of three verdicts:

- `✓ FAITHFUL` — line-by-line 1:1.
- `⚠️ MINOR` — falls under one of the tolerable categories. Cite the
  preamble line if the gap is a documented slice/specialization.
- `❌ DEVIATES` — falls under any must-fix category. Cite `.py` line
  number, `.lean` line number, and the divergence concretely.

## Workflow at scale

For a sweep across many kernels, parallel batches of ~15 files per
read-only audit agent are the right granularity. Each agent reads each
pair, applies this rubric, and emits one block per file.
