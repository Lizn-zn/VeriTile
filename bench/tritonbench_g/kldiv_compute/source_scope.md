# Source obligations under active verification

The source computes the elementwise term `x * log(x / y)`, without a reduction.
Both loads use the same lane mask and omit `other`; the store uses that mask.
Both inputs are loaded before any output is written, so per-program snapshots
can allow output/input aliases. Cross-program races are a separate obligation.

The typed source preserves all eight statements. Its candidate commutes
only the natural product forming `block_start`; it changes no floating
operation. The floating family independently chooses X, Y and output storage
from six formats (216 triples); the wrapper's `empty_like(x)` correlates X and
output. Store conversion is explicit when its destination differs from the
computed result. Division/log/product precision and every destination cast
pass exact AST checks for the supported source profiles. Native float8 arithmetic support is not
established by a Lean storage label; discrete input and mixed conversions
remain separate coverage obligations.

The mathematical oracle is the existing independent `klDivSpec`. Real logarithm
and division are total in that model; interpreting this as ordinary finite KL
terms requires positive active inputs. Inactive lanes are evaluated too, so a
floating proof must retain arbitrary undefined loads and failure behavior.
The planned structural equivalence leaves all these operations untouched and
requires no numerical rewrite atom.

The public real IO uses the existing `MaskedKernelIO₂.Implements` flat bridge:
its launch state explicitly has zero undefined values, with arbitrary remaining
registers/memory, symbolic program id and active-lane bounds. Region execution
and structural FP equivalence preserve their broader undefined-value domains.

The historical declarations and docstrings are preserved verbatim across
`Kernels.lean`, `Proofs/LegacyReal.lean` and `Correct.lean`; the obsolete combined
module has been removed. Typed source, real projection, candidate flat-execution
transport, complete real IO/frame and precision checks pass their leaf compiler
gates. Actual trace safety and full backend state/memory/failure equivalence
also pass. The mixed fp64/fp16/bf16 exact-real execution witness now passes for
both variants: program id one writes offset two with `4 * log 2` and preserves
the masked offset-three sentinel seven. Both public entries pass their leaf
checks, and the FP assumption list is empty. The dependency-aware public build
and shared-library refresh pass all 3538 jobs. All eight final public/support
axiom checks use only standard Lean axioms. Three public contracts are registered
with validated source/support fingerprints as `test-gap`; independent comparator
replay is still pending, so this is local verification, not final acceptance.

The typed source and all-supported-profile arithmetic AST check now compile
after the shared division correction. The initial AST check failed, including
implicit/explicit precision mismatches. Static comparison identified
a promotion discrepancy in the previous mixed-half implementation:
`expandArith` previously applied generic binary promotion before division promotion.
For fp16/bf16 operands this introduces bf16-to-fp16-to-fp32 conversion, whereas
the pinned Triton frontend selects fp32 directly and casts both original operands
to that type. The compiler accepting the source does not certify this precision
boundary. The shared correction now chooses division's type from the original
operands, with an explicit fp32 compute annotation, before generic promotion;
its DSL build passes all 789 jobs and `FloatDivisionPromotion.lean` passes exact
AST checks for both operand orders, mixed-half multiplication and fp64 dominance.
The rebuilt source passes exact load checks for all 216 formal profiles and
division/logarithm/multiplication/store checks for all 192 supported profiles,
in both variants. Dependent execution and contract proofs are being checked.
The independent AST expectation retains direct promotion.

Pinned frontend check: Triton v3.5.1
[computation_type_impl and truediv](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/semantic.py)
keep identical fp8 pairs in fp8 and promote differing fp8 pairs to fp16.
The [log dtype decorator](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/math.py)
accepts only fp32/fp64 and explicitly rejects other tensor dtypes. Thus all
four both-fp8 input pairs are rejected by the pinned frontend (or may fail
earlier during division lowering). This inference is from source inspection,
not a native GPU compile observation. Across six destinations, 24 of the 216
formal carrier triples are not source-supported. The public new contracts
require `sourceSupported x y = true`, retaining 192 triples; abstract helper
lemmas may still cover the larger carrier model. This does not certify target
lowering for the remaining triples or treat invalid source calls as successes.
