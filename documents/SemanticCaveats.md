# Semantic Caveats

[中文](SemanticCaveats_zh.md) | **English**

This note records the semantic places where VeriTile is intentionally less
faithful than real Triton/CUDA. These are not Lean soundness issues: they are
interpretation rules for theorem statements. A theorem proved under these
models should be read as a theorem about VeriTile's mathematical algorithm
semantics unless its statement or `GapPolicy` explicitly covers the hardware
gap.

## High-Risk Boundaries

| Boundary | Current model | Risk | Required discipline |
| --- | --- | --- | --- |
| IEEE floating point | Floating carriers are `WithBot ℝ`; dtype tags erase to mathematical Real semantics. | NaN, signed zero, rounding, overflow, underflow, denormals, exception flags, fast-math rewrites, and hardware dot precision are not proved. | State domain/range hypotheses for math operators and use `GapPolicy` / external checks for bit-level claims. |
| Partial math functions | `tl.log`, `tl.sqrt`, `tl.rsqrt`, and `tl.extra.cuda.libdevice.pow` use Mathlib total functions on finite Real inputs. | Invalid hardware inputs that would produce NaN or infinities can become ordinary mathematical values. | Require hypotheses such as `x > 0`, `x ≥ 0`, nonzero denominator, or positive `pow` base when claiming Triton/CUDA fidelity. |
| Fixed-width integers | `tl.int*` maps to mathematical `Int`; `tl.uint*` maps to `Nat`. | No width, overflow, wraparound, saturation, sign-extension, or signed fixed-width bitwise semantics. | For quantization/int kernels, state range predicates such as `tritonInt8CastInRange`; do not infer hardware-width behavior from `.int` proofs. |
| Addressing | Pointers are `RegionName × Nat`; block-pointer offsets, strides, and base offsets are `Nat`. `BlockPtr.AdvanceNonnegative` records the theorem-side no-underflow obligation for signed `tl.advance` deltas. | Negative pointer arithmetic and underflow are truncated or unrepresentable in execution; the model is cell-offset based, not byte-address based. | State `BlockPtr.AdvanceNonnegative` and in-bounds obligations for `tl.advance`, pointer subtraction, and block-pointer rewinds. |
| Total memory reads | `readMem` / `readMemAs` return zero on dtype mismatch or non-Real cells; `readMemValue` uses dtype defaults for mismatches. | A malformed load can be silently totalized instead of failing or producing undefined hardware behavior. | Prefer typed-load hypotheses, `RegionTyping` / `Kernel.checkStrict`, and explicit `TensorView.loaded` assumptions. |
| Storing `⊥` | Floating stores demote `⊥` through a finite fallback (`0`). | A path that should be impossible, undefined, NaN, or `-inf` can become an ordinary zero write. | Headline proofs should establish that stored floating values are finite, or explicitly treat the fallback as a model assumption. |
| Atomics and concurrency | Atomic support is limited to narrow algorithm-level markers and explicit trace/linearization relations. | There is no full scheduler, memory ordering, warp/block interleaving, barrier, shared-memory, async, or TMA semantics. CAS compares `MemCell`s mathematically, not hardware bit patterns. | Read atomic theorems as abstract RMW/linearization theorems, not CUDA memory-model theorems. |
| Block-pointer metadata | Runtime block-pointer addressing remains total through list `getD` defaults, while the optional checker rejects rank-mismatched metadata and statically visible `tl.advance` underflow. It propagates a unified `BlockPtrSummary` (region, optional parent rank, optional static offsets) through simple block-pointer expressions and registers. The theorem-side predicates are `BlockPtr.WellFormed`, `BlockPtr.CheckedAxesValid`, and `BlockPtr.AdvanceNonnegative`. | Dynamic block pointers can still require proof assumptions; malformed metadata in unchecked execution can be more defined than real Triton IR would be. | Use `Kernel.checkStrict` where possible, and carry `BlockPtr.WellFormed` / `BlockPtr.CheckedAxesValid` / `BlockPtr.AdvanceNonnegative` in theorem statements for block-pointer consumers. |

Local bridge theorems connect checker success to these predicates:
`checkBlockPtrMetadata_ok`, `checkBoundaryAxes_ok`, and
`checkStaticAdvanceNonnegative_ok`. The summary layer also has local bridges
for its construction and propagation steps:
`BlockPtrSummary.ofStaticChecked_ok`,
`BlockPtrSummary.ofDynamicOffsetsChecked_ok`,
`BlockPtrSummary.checkedAdvance_ok`, and
`BlockPtrSummary.checkBoundary_ok`.
For new local checker obligations, follow the same split: one shared executable
Bool helper, one Prop-facing contract when theorem statements need it, and an
`_ok` theorem from checker success to the Prop contract.

## Practical Review Checklist

Before calling a theorem "faithful to Triton" rather than "correct in
VeriTile's Real semantics", check:

- Every `log`, `sqrt`, `rsqrt`, division, and `pow` use has the relevant domain
  preconditions.
- Every fixed-width integer or quantization claim has range/width assumptions,
  or explicitly stays at the mathematical `Int` / `Nat` level.
- Every pointer or block-pointer offset that may move backward has a
  nonnegative/in-bounds obligation, preferably `BlockPtr.AdvanceNonnegative`
  for signed block-pointer advances.
- The proof does not rely on dtype-mismatch reads returning zero.
- The proof does not rely on storing `⊥` as zero unless that behavior is part
  of the stated model contract.
- Atomic/whole-grid theorems mention whether they use disjoint-frame merge,
  atomic-add sum merge, or explicit RMW linearization.

## When To Strengthen The Model

Strengthen the internal semantics when a consumer needs to prove a claim that
cannot honestly be expressed as a Real-semantics theorem plus an external gap
contract. Good candidates are:

- an IEEE or mixed-precision theorem whose conclusion depends on rounding
  magnitude;
- a quantization theorem whose conclusion depends on fixed-width wraparound or
  saturation;
- a pointer-heavy theorem whose correctness depends on byte addressing or
  negative offsets;
- a concurrent kernel whose correctness depends on memory ordering, barriers,
  async/TMA completion, or scheduler interleavings.
