# softmax_optimize

- Source file: `softmax_optimize.py`
- Corpus: TritonBench-G v1
- Public contracts: five real specifications (three preserved historical, two complete signed-source) and one full-source FP equivalence.
- Validation: the complete real/FP public-entry build passes (3486 jobs), all six public contracts have only standard axioms, and the FP assumption list is empty. Source/support fingerprints are registered as `test-gap`; independent comparator replay remains pending.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

## Exact source and proof scope

`Kernels.lean` transcribes all four loops, both reductions, named input/output
pointer assignments, casts and masks. Its candidate commutes the two integer
factors in each row-base address. The helper uses signed arithmetic, including
`prev_multiple_of(0, B) = -B` for positive `B`; it does not Nat-truncate that
boundary. `Proofs/Geometry.lean` checks both this negative value and the fact
that the zero-column tail mask admits its negative-offset lanes.

The source-facing storage family has six input formats and fp32/fp64 output
formats. This is a direct-call family; the Python `empty_like` wrapper correlates
input/output formats. [Pinned Triton v3.5.1 `language/math.py`](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/math.py#L86-L92) checks that `exp`
receives fp32 or fp64. Because this kernel initializes its accumulators and casts
loads to the output dtype, half/bfloat output profiles are not admitted by that
source check. This is a conclusion from primary source, not a native GPU compile
observation.

The new source-facing real contracts use the nonempty numerical domain
`0 < B ∧ B ≤ N`. This avoids a concrete source-level IEEE hazard: when `B > N`,
a never-active lane keeps both old/new maxima at negative infinity, and the
source computes `exp(-inf - -inf) * 0`. A resulting NaN can contaminate the
reduced denominator. Small non-power-of-two wrapper shapes therefore must not
be declared safe merely because only one tile is launched. This is primary-source
analysis, not a checked native IEEE counterexample. The real interpreter
totalizes extended-real operations; its formula theorem is not an assertion of
bit-exact IEEE agreement with real exponentials.

`Correct.lean` exposes successful execution, the independent full-row formula
`exp(input[j]) / sum(exp(input[k]))`, every-cell preservation outside the logical
output row, and access safety for both actual signed programs. Bounds quantify
over arbitrary row count, column count, tile width and in-bounds program id.
Input/output regions are distinct; no execution or output-value assumption is
required. All four loops and the cross-lane normalization are covered.
`Proofs/SignedProgram.lean` proves signed/Nat simulation with simultaneous
failure and exact raw memory agreement; `Proofs/SignedIO.lean` derives the
successful source theorem and a nonempty `N=3, B=2` partial-tail witness, including
preservation of the masked next cell. The region theorem permits arbitrary
undefined memory. The flat bridge explicitly requires zero undef because that
is a premise of the existing `flattenState` theorem.

`FPEquiv.lean` proves full-state and failure equivalence for all 12 direct-call
storage pairs, every shape and every initial state, including `N=0` and zero
stride. It uses only exact integer multiplication commutation; it requires no
floating rewrite atoms or admitted numerical identities. All casts, memory
operations and floating operations remain unchanged.

The three historical public declarations and their docstrings are preserved
verbatim in `Correct.lean`; their helpers are in `Proofs/LegacyReal.lean` and
their kernel definitions in `Kernels.lean`. Their old Nat-helper and one-tile
models remain explicitly historical. In particular, the historical claim of a
faithful helper at zero and the mathematical results allowing `B > N` are not
promoted to native source correctness. The new signed-source contracts cover
the full program separately.

The arithmetic/region semantics use mathematical integers and natural memory
addresses; negative pointer offsets are converted at the framework pointer
boundary. Retaining the zero-column signed control is not a claim that its
negative native accesses are valid. Likewise, `NumericalDomain` is a numerical
condition, not a GPU launch validator: native `arange` tile restrictions and
fixed-width address overflow are separate source/backend conditions.
