# geglu_tanh_triton

- Source file: `geglu_tanh_triton.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The migrated layout uses three public entries: `Kernels.lean` holds both complete legacy source bodies,
`Proofs/Real.lean` holds their successful execution, bounds and frame proofs,
and `Correct.lean` holds the preserved public contracts. All 19 original
headers and 18 declaration docstrings are retained. `FPEquiv.lean` contains
the complete typed FP proofs; supporting lemmas live in `Proofs/`.

`Proofs/Math.lean` supplies independent scalar forward and backward formulas
using the source coefficient `0.7978845608028654` exactly. Checked bridges
connect these to both complete real IO contracts, including the two in-place
backward stores. This establishes the written backward expression, not an
additional analytic derivative theorem. `Proofs/FP.lean` proves exact signed
row-address multiplication commutation, including conversion to pointer offsets
and failed evaluation, without floating-point atoms.

The new typed source family includes 36 forward A/B storage combinations and
216 backward A/B/DC combinations across fp32, fp16, bf16, fp64, fp8e4nv and fp8e5, each with an
original and signed-address-commuted candidate. Both output buffers preserve
the source storage type. The Python `tanh` import is resolved explicitly as
`libdevice.tanh`, preserving its distinct floating backend and fp32 argument
computation. Store conversions are explicit only where the computed value and
destination storage differ.

All six-storage source branches, mathematical projections, source-shape checks
and full forward/backward FP equivalences pass their compiler gates. Both FP
headlines print no numerical assumptions. The corrected integrated public-entry dependency build passes (3445 jobs). Eight
contracts have validated source/support fingerprints and `test-gap` status. Nonempty numerical witnesses
also pass: fp16 A, fp64 B and bf16 DC, a nonzero row and a masked adjacent lane,
with forward zero and backward gradient 6 while preserving padding. These use
exact-real arithmetic rather than IEEE rounding. No new independent comparator
acceptance is claimed.

Source review correction: Python/Triton folds the literal-only
`3 * 0.044715` before converting the scalar to fp32. The typed backward source
therefore uses `0.134145 * a_row * a_row`, preserving both tensor multiplications.
The old runtime fp32 scalar multiplication gave a coefficient one ULP smaller.
The corrected real bridge, all dependent gates and exact precision regressions
now pass. Legacy declarations are unchanged.

The typed whole-row real contracts require `0 < BLOCK_SIZE` and
`n_cols ≤ BLOCK_SIZE`; backward additionally requires distinct A/B output
regions. Python's fixed BLOCK_SIZE=128 means its bundled n_cols=256 example
writes only the first 128 columns per row. The preserved legacy contracts
state active-tile behavior and do not claim the missing columns were written.
Signed row arithmetic uses mathematical Int: fixed-width overflow, negative
strides and target-specific fp8 availability remain frontend/host boundaries.

The scalar-fold normalization follows Triton v3.5.1's
[constant binary-expression dispatch](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/compiler/code_generator.py#L665)
and [constexpr multiplication](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/core.py#L220).
`Proofs/Precision.lean` pins the folded term2 AST for every backward storage
combination, as well as every load/store and selected mixed-precision gradients.
