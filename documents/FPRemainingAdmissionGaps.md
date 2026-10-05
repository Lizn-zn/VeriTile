# Remaining FP example prerequisites

The migration has 19 real correctness files and 18 completed FP equivalence files.
The current main and supplemental reports select numerical assumptions using
a local-ULP mean-bias budget and a peak absolute-error ratio gate.
The reciprocal softmax cases retain their explicit operand domains. RowWiseSum
binds its conditional derivation to the admitted fp32 ADD-COMMUTE and ADD-ASSOC
instances. StableLogSumExp is the remaining incomplete transformation.

Its `candidateLSEKernel` now composes `log_mul_split` and `log_exp_elim`,
including their product/center thresholds and inactive-argument masks. The
candidate has an independent real correctness proof. The scalar composition
is derived in `Float/LogSumExpCandidate.finish_eq`; it uses no unconditional
log identity. The original direct reference uses `tl.log`, while these atoms
use `libdevice.log`. Completing a source-pair FP certificate therefore still
requires an admitted intrinsic bridge or an explicitly revised reference.

## Current primitive experiment results

All three probes completed on H200 in DLC job `dlc1qojvuygw2t0b`, followed by
independent CPU replay. Each uses 4096 replicates of shape `[4096, 4096]`,
the 0.05 local-ULP bias budget, and U thresholds 10/100.

| Primitive | z | B (local ULP) | U | Decision |
|---|---:|---:|---:|---|
| `tl.exp(a-b)` vs `tl.exp(a)/tl.exp(b)`, fp32 | 33630.25971 | 0.1608954387 | 3.958463781 | REJECT: bias |
| `fromNat_fp32(0)` vs literal fp32 zero | 0 | 0 | 0 | ACCEPT: constant relation |
| `fromNat_fp32(i+1)` vs fp32 `fromNat_fp32(i)+1` | 0 | 0 | 0 | ACCEPT: integer `0 <= i < 2^24` |

The intrinsic exp-sub uses independent Normal(1,1) operands and fails the bias
gate. At the user's request, the stable softmax, online softmax and logsumexp
examples now use `libdevice.exp`. Their independent Correct and FP source
copies explicitly document why `tl.exp` cannot supply this rewrite under the
configured probe. `Op.libdeviceExp` and the FP unary symbol retain a distinct
implementation identity; the real evaluator gives both exponential operations
the same mathematical meaning.

The successor probe draws uniform int32 counts in `[0, 2^24)`, converting them
inside the kernel. A separate exhaustive GPU check also verifies every one of
those 16,777,216 integers exactly. The out-of-range `i=16,777,217` counterexample
is preserved: reference 16,777,218 versus candidate 16,777,216. Thus a Welford
binding retains `N <= 2^24`; `Float/CountConversion.conversion` now derives
the symbolic-N conversion premise within that bound. COUNT-ZERO is constant; its repeated
execution adds no stochastic coverage. Both count U values use empirical-max
fallback with zero reference and candidate errors.

The integer experiment is numerical evidence. `scripts/export_count_rules.py`
now exports the two accepted rows with their integer bound. The successor atom
encodes its guard in scalar syntax: outside the admitted range both fragments
return the same zero. Extracting the actual conversion equality requires
`i < upperExclusive`. The generic scalar exporter still rejects these rules,
so they cannot be accidentally imported without this integer-range-aware binding. See the [current report](../experiments/floating_point/primitives/report/summary.md)
and [reproduction instructions](../experiments/floating_point/primitives/README.md).

### Current log experiments: guarded log-exp and product rewrites pass

The fixed-reference FP32 LOG-EXP-GUARDED candidate passes on H200 with both
seeds. DLC job `dlc1713xu2g5t8e2` uses the existing sampling and gates; all primary
and confirmation tables match independent CPU replay.

```python
reference = libdevice.log(libdevice.exp(a))
if 0.5 < abs(a) <= 80.0:
    candidate = a
else:
    candidate = libdevice.log(libdevice.exp(a))
```

All operations remain FP32. The candidate eliminates both calls away from zero;
near zero it preserves the reference's intermediate rounding. The upper bound
retains the original overflow, underflow and subnormal-exp behavior on extreme
inputs. These are implementation branches, not input filters. The reference,
sampling distribution and two-gates thresholds remain unchanged.

The GPU skips exp/log when every active lane of a block selects the identity.
Mixed blocks use zero-masked fallback arguments and select the result per lane.
The Lean example represents this lane-wise expression; its proof does not verify
the GPU compiler or the block-level scheduling optimization.

| Rule | R | z | B (local ULP) | U | Accept |
|---|---:|---:|---:|---:|---|
| LOG-MUL / fp32 | 4096 | 2.924915168 | 0.1679352504 | 6.685560237 | No: bias INCONCLUSIVE |
| LOG-MUL-LIBDEVICE / fp32 | 4096 | 2.924915168 | 0.1679352504 | 6.685560237 | No: bias INCONCLUSIVE |
| LOG-EXP-LIBDEVICE / fp32 | 4096 | 68.59740095 | 0.7300322166 | 0 | No: bias FAIL |
| LOG-EXP-FULL-LIBDEVICE / fp32 | 4096 | 68.59740095 | 0.7300322166 | 0 | No: bias FAIL |
| LOG-EXP-GUARDED / fp32 | 4096 | 28014.76841 | 0.04578995059 | 0.625 | Yes |
| LOG-MUL-LOG1P / fp32 | 4096 | 2.061440595 | 0.1240315873 | 6.685560237 | No: bias INCONCLUSIVE |
| LOG-MUL-GUARDED / fp32 | 4096 | 296.6221498 | 0.0006248690033 | 6.685560237 | Yes |

The independent seed 20261005 gives LOG-EXP-GUARDED z=28421.27843,
B=0.04579159812, U=0.625; it also accepts LOG-MUL-GUARDED.
The current table binds `log_exp_elim` and `log_mul_split`, preserving their
full conditional expressions. Neither admits unconditional log cancellation
or product splitting, so the original StableLogSumExp remains incomplete.
The separate FMA/log1p product-reference probe remains bias-INCONCLUSIVE.

The 25-input GPU log-exp fixture checks fixed-reference bits, endpoint neighbors,
near-zero rounding and extreme-tail fallback; 16 product boundary inputs also
pass. The standalone timing check shows no speedup, so numerical acceptance is
not presented as a performance result. See the [current experiments](../experiments/floating_point/supplement/README.md)
for timing measurements, full tables and reproduction commands.

## Checked algebraic evidence

[FPAdmissionCoverage.lean](../bench/tests/FPAdmissionCoverage.lean) checks
countermodels for the **scalar equations** in the frozen admission table. Its
list of rule families is computed from `ReportedAdmission.all`; an added family
changes the coverage obligation. The examples below use identity casts, so the
countermodels satisfy every main-table precision instance of each
family. All numerical symbols are interpreted on rational numbers.

These are not IEEE executions, GPU failures, or two-gates results. They show
that the selected equations alone leave the proposed conclusion undetermined.
The fixture is scoped to `ReportedAdmission`, not the supplemental table.
Its add-zero, ordinary-division and opaque-exp countermodels need not satisfy
`SupplementalAdmission`; they cannot establish a gap in the combined theory.

| Missing conclusion | Checked countermodel |
|---|---|
| `a + 0 = a` | Interpret addition as `a + b + 1`, subtraction as `a - b - 1`, multiplication as `(a+1)(b+1)-1`, and both divisions as constant `-1`. Every admitted family holds, but adding literal zero changes the value. |
| Welford's singleton initialization | In that same model, batch mean is `-1`; the online update from literal zero is `0`. This persists even when ordinary division and `div_rn` are the same function. |
| Ordinary division to reciprocal multiplication | With otherwise ordinary arithmetic, interpret `div_rn` as rational division and ordinary division as `a + 1`. The admitted `DIV-RCP` law holds; `2 / 3` versus `2 * (1 / 3)` under the ordinary symbol gives `3` versus `4`. |
| Stable softmax shift | Use ordinary arithmetic/division but interpret `exp(a)` as `a+2`. On `[0,1]`, the first naive output is `2/5`; the stable output is `1/3`. All admitted families still hold. |
| Stable logsumexp shift | In the preceding model, interpret `log` as identity. The direct result on `[0,1]` is `5`; the shifted result is `4`. |
| LayerNorm output after replacing statistics | Use the translated arithmetic above, rational ordinary division, constant `-1` for `div_rn`, and constant `1` for sqrt. For the singleton `x=2`, gamma `1`, beta `0`, epsilon `1`, the two-pass output is `0`, while the online-statistics output is `2`. |

The fixture also checks that the frozen table has only bf16/fp32 compute
formats. It supplies no fp64 instance.

## Work still needed

| Case | Numerical prerequisites still to settle | Implementation work after admission |
|---|---|---|
| `RowWiseSum` | Both fp32 addition assumptions are admitted and bound. | The conditional reduction-tree derivation is connected; no whole-reduction numerical guarantee is inferred. |
| `SoftmaxStable` | fp32 libdevice EXP-SUB is admitted and bound together with the scalar arithmetic rules. | `SoftmaxStableFPEquiv.softmax_stable_equiv` completes the guarded, scheduled equivalence; max, bf16 stores, output frames and symbolic positive row length are retained. |
| `StableLogSumExp` | libdevice EXP-SUB is bound. LOG-MUL/fp32 is bias-INCONCLUSIVE; LOG-EXP-LIBDEVICE/fp32 is bias-REJECT. Neither is admitted. | Both libdevice source variants remain conditional on the two log obligations; the completed GPU experiments do not close the specification. |
| `OnlineSoftmax` | libdevice EXP-SUB is bound. The normalized-value derivation needs no max identity or EXP-NEG-INF-SUB atom. | `OnlineSoftmaxFPEquiv.online_softmax_equiv` completes the original Correct observation scope: batch stored values versus read-only normalization using actual final online m/l registers. Both original executions and separate memory frames are retained; no output store is added. |
| `Welford` | Both count atoms are admitted and bound for integer `0 <= i < 2^24`. | `WelfordFPEquiv.welford_equiv` closes the original comparison for `0 < N <= 2^24`, retaining both bf16 outputs and frames. |
| `FusedLayerNorm` | The same two count atoms are admitted and bound. | `FusedLayerNormFPEquiv.layernorm_equiv` closes the original comparison for `N <= 2^24`; empty output rows remain covered. |

This table lists prerequisites, not newly available assumptions. It does not
assert that any proposed numerical experiment will pass.

### Implemented scalar and reduction derivations

`Float/ScalarArithmetic` binds eleven accepted fp32 arithmetic instances to
explicit scalar fragments, including their operand guards. The binding checks
the rule ID, all precision fields and domain against the current report.
Executing the fragments in a model of these assumptions yields the scalar
laws; no Real ring instance or extra numerical axiom is supplied.

The module derives subtraction by zero, multiplication by zero, multiplicative
cancellation and shared-scale division normalization. `Float/ScalarReduction`
then derives common-factor extraction through an arbitrary explicit addition
tree, with its padding retained, and shared-scale row normalization. The
finite/nonzero conditions include intermediate partial sums and reciprocal
values; finite input leaves alone do not discharge these conditions. Its
execution adapter expands fp32 sums and preserves other precisions, casts and
opaque max/exp operations.

`Float/Exponential` binds fp32 libdevice EXP-SUB to its exact scalar syntax,
combines it with the arithmetic table and derives `LibdeviceExpSub`. The public
stable-softmax specification now uses that admitted law.

### Original stable softmax connection

`Float/SoftmaxShift` derives the shifted exponential row's common factor and
then its normalized output using the scalar arithmetic theory and an explicit
sum tree. `LibdeviceExpSub` is obtained from `Float/Exponential`
and the accepted libdevice EXP-SUB row. No softmax or reduction equality is a premise.
The shift may be any opaque finite value: the derivation needs no law for max.

[SoftmaxStable/Execution.lean](../bench/examples/SoftmaxStable/Execution.lean)
imports both libdevice kernels from the example's shared `Kernels.lean`.
`original_runs_under_exp` in
[SoftmaxStable/Contract.lean](../bench/examples/SoftmaxStable/Contract.lean)
connects their successful runs,
all bf16 output cells and memory frames to this conditional derivation. Row
length remains symbolic and positive because the original max rejects an empty
axis; input/output aliasing is allowed. The one-input scheduled IO signature
retains precision, layout and the syntactic finite/nonzero domain, including
intermediate sums and reciprocals. Domain syntax contains no equality premise.

The regression checks source identity, independence from Correct, empty-row
failure, in-place execution, contract satisfiability and the nonzero boundary.
Its rational fixtures are logical checks, not experimental evidence. An opaque
whole-kernel premise prints `unresolved FP proof`; it cannot be reported as
having no atomic assumptions. `SoftmaxStableFPEquiv.softmax_stable_equiv`
closes this case with admitted arithmetic and EXP-SUB atoms; the FP count is 17.

### Original stable logsumexp connection

`Float/LogSumExpShift.recover_sum` factors the shifted exponential row through
its explicit addition tree and cancels the common reciprocal using accepted
scalar arithmetic. `shifted_result` uses libdevice EXP-SUB and the still-pending
LOG-MUL and `LogLibdeviceExp` obligations. Each is a scalar equation with the
experiment's operand-domain shape; none is a reduction or logsumexp identity.
The checked domain includes actual partial sums, reciprocal intermediates and
positive log-product operands, and contains no equality premise.

[StableLogSumExp/Execution.lean](../bench/examples/StableLogSumExp/Execution.lean)
uses the shared libdevice.exp sources from `Kernels.lean`, retaining
their single bf16 output at `pid`, symbolic positive row length and unchanged
memory outside that one cell.
[StableLogSumExp/Contract.lean](../bench/examples/StableLogSumExp/Contract.lean) connects the scalar
derivation to both successful executions under the scheduled fp32 profile.
`original_runs_under_log` discharges exp-sub from the admitted table; the
two log obligations remain explicit.
In-place output is allowed. The original max and final bf16 conversion remain
opaque. This is a conditional connection, not a completed admitted FP example.

The dedicated `experiments/floating_point/supplement/log_config.py` profile runs
LOG-MUL/fp32 with input-domain filtering and the new LOG-EXP-LIBDEVICE/fp32 probe.
The latter evaluates `tl.log(libdevice.exp(a))` with a separate rule ID and
implementation contract; the old LOG-EXP results cannot authorize it. Both use
the existing shape, Normal(1,1) distribution and gates. Invalid inputs are skipped
without resampling, and nonfinite outputs on valid inputs remain failures.
The current H200 report is complete: LOG-MUL is bias-INCONCLUSIVE and
LOG-EXP-LIBDEVICE is bias-REJECT. Neither supplies the required admission.
The paired `log_accuracy_config.py` additionally accepts LOG-EXP-GUARDED, but its
conditional identity retains fallback behavior and cannot supply the unconditional identity needed by this proof.
See the supplemental README for the reproduction commands and report.

The regression shows that EXP-SUB and LOG-MUL can hold with the domain while
the original stored outputs still differ without LOG-EXP. It also checks why
an equality after a noninjective bf16 cast cannot replace an uncast equality
inside the final addition. These are logical fixtures, not GPU results.
The assumption printer recognizes libdevice exp, pending log and generic count-conversion
records, including their projected fields, and reports external premises as
`unresolved FP proof`. Reconstructing a record does not hide its provenance.

### Implemented loop execution

The opaque FP interpreter now executes compute-level counted loops, static and
dynamic ranges, and conditional branches, including nesting. It captures
dynamic range bounds once, resets the exact natural index before each
iteration, propagates reached failures, and skips inactive bodies. The range
zero-step behavior matches the existing operational semantics. `Float/Control`
provides successful-execution induction principles for both loop forms; it
does not add arithmetic assumptions.

The independent [Welford/Execution.lean](../bench/examples/Welford/Execution.lean)
and [OnlineSoftmax/Execution.lean](../bench/examples/OnlineSoftmax/Execution.lean)
modules import their original Triton sources from each example's `Kernels.lean`. Welford's
online loop computes the opaque recurrence and writes both bf16 output cells;
the two-pass execution retains both original sum operations. Their proofs
cover symbolic row length and stride, including empty rows, and frame every
untouched cell. OnlineSoftmax computes its recurrence in `m` and `l` and
preserves all memory, matching the original source's lack of an output store.
Source-binding tests check that execution and correctness use the same kernels;
the execution proofs do not import the correctness files.

These are execution prerequisites, not new completed FP equivalences. The
Welford scalar derivation, loop induction, schedule comparison and public
domain contract are connected below. Count conversion remains an opaque
operation: loop support does not imply
`toReal(i + 1) = toReal(i) + 1`. The OnlineSoftmax connection below derives its
recurrence invariant and normalized-value comparison, conditional on the
admitted libdevice exponential law.

### Original online softmax normalization

`Float/OnlineSoftmax` derives `l * exp(m) = prefixSum(exp(x))` for every
positive-length prefix of the original recurrence. The first step preserves
the actual `exp(-inf - newMax)` expression and requires its result and the
zero-product intermediates to be finite. It does not declare `-inf` finite or
silently remove that expression. Subsequent steps derive the invariant using
scalar distribution, association and reciprocal cancellation plus the admitted
libdevice EXP-SUB relation. Every update's domain is checked.

`normalized_prefix` then derives the online normalized values using the actual
computed m/l. The prefix sum is an explicit valid addition tree with its seed
retained. `OnlineSoftmaxContract.normalized_values` compares it with an
arbitrary valid batch schedule through the existing scalar rewrite paths and
their intermediate domains. Both centers remain opaque; neither a max-tree
identity nor equality of the online and batch maxima is a premise.

`OnlineSoftmaxComparison` connects the invariant to the original loop's actual
registers. `OnlineSoftmaxBatch` independently copies the original batch kernel,
including its real-typed row store (there is no bf16 conversion in this case).
`original_normalization_runs` proves both successful executions, identifies
every batch output with the normalized online result, frames untouched batch
cells and preserves all memory on the online side. The reified comparison
domain contains only the iteration, normalization and schedule checks.

The fixtures check complete-domain satisfiability without declaring the
negative-infinity sentinel finite, a later step whose valid domains do not
cover an invalid initialization, retained prefix padding, both original source
copies and aliased memory behavior. They supply no experimental evidence.

`OnlineSoftmaxFPEquiv.online_softmax_equiv` retains the existing Correct
example's observation scope through `Float/ObservedRow`: successful batch
output reads equal read-only normalization using the actual final online m/l.
It uses admitted libdevice exp-sub and arithmetic atoms. An absent register
makes the readback fail; two failures cannot establish a numerical step.
Each source retains its own memory frame, and no online output store is added.
The completed FP count is 17.

### Welford mean step and integer conversion

`Float/Welford.mean_step` derives
`(m + (x-m)/(n+1)) * (n+1) = m*n + x` from the accepted scalar arithmetic
atoms. Every intermediate finite/nonzero requirement is explicit. The result
is connected to the original loop update by `fp32_mean_step` in
[Welford/Execution.lean](../bench/examples/Welford/Execution.lean).
Here `n` remains the actual `fromNat(i)` value: identifying its successor with
`fromNat(i+1)` is not part of the proof.

`Float/ExecutionProfile` selects fp32 for implicit arithmetic in these source
kernels while retaining explicit ComputeOp precisions, integer conversions,
reductions and output casts. It does not supply numerical equalities. The
execution and scalar-law connection therefore uses an explicit precision
selection, without treating algorithm-typed operations as Real arithmetic.

[FPWelfordArithmetic.lean](../bench/tests/FPWelfordArithmetic.lean) checks a
countermodel for all eleven guarded arithmetic equations in
`Float/ScalarArithmetic`, including the supplemental identity and inverse
laws. Arithmetic and reductions use rational numbers, casts are identities,
and `fromNat(n)` is interpreted as `2*n`. Conversion of zero is correct and
every positive count is nonzero, but on input `[2]` the two-pass mean is `1`
and the online mean is `2`. This is a proof-library coverage check, not a GPU
failure or a model of the transcendental rule families.

An integer-conversion relation consequently needs a matching primitive binding
before the original count-based invariant can close. PR #12 now supplies accepted
integer probes, including successor conversion for `0 <= i < 2^24`. Their Lean
binding must retain that range; no unrestricted conversion law has been added.

### Welford variance step and recentering

Let `q = (x-m)/(n+1)` and `m' = m+q`. `Float/Welford.residual_step`
derives `x-m' = n*q`; `variance_step` then derives
`(x-m)*(x-m') = (x-m')² + n*q²`. The original loop's variance update is
connected to this identity by `fp32_variance_step` in
[Welford/Execution.lean](../bench/examples/Welford/Execution.lean), retaining
the actual floating conversion of the loop index. These derivations use the
same admitted scalar theory, with explicit guards for the new residual and
other intermediate operands.

`square_shift` expands `(x-m')²` around the old mean `m`, keeping its two
cross terms separate. `ScalarReduction.value_add` and `square_shift_tree`
lift that expansion through an arbitrary explicit addition tree. Literal-zero
padding and the intermediate domains of both component trees and their sum
are retained. The repeated sum of the constant shift square remains a tree;
it is not silently replaced by an integer-count product.

The arithmetic fixture checks why the additional domains matter: valid mean
step guards need not imply a finite residual, and finite component trees,
paired leaves and final result need not imply finite transformed partial
sums. The fixture also audits these new derivations for unexpected axioms.

These are local identities and reduction lemmas, not a completed Welford FP
equivalence. The full recurrence and reduction-schedule comparisons are
connected below under explicit count-conversion obligations. The accepted
bounded count relations are now bound by `Float/CountConversion`; the completed example count is 17.

### Centered sums and vanishing variance cross terms

`Float/WelfordReduction` defines the floating count of an explicit tree as
that tree's sum of literal ones with literal-zero padding. From the accepted
scalar atoms, `ScalarReduction.constant_value` factors a constant row into
its value times this count, and `deviations_add_center` recovers the original
sum by adding back the constant center row.

For a mean defined as the input sum divided by this explicit count,
`centered_sum_zero` derives that the sum of deviations is zero. The denominator
must be nonzero, and the record of required domains contains no equality
premises. `cross_sum_zero` then factors and cancels the cross terms in the
variance expansion. `centered_square_shift` concludes that the recentered
square sum equals the old square sum plus the sum of shift squares.

These lemmas support the variance invariant without assuming a whole-row
identity. They do not replace the original kernel's `fromNat(N)` denominator
with the tree count. The arithmetic fixture checks this distinction in its
existing conversion countermodel: a singleton tree padded by two zeros still
counts as one, while `fromNat(1)` is two. It also checks that the nonzero-count
guard rejects an empty tree. The connection below supplies the bounded
conversion binding and global invariant proof for the original recurrence.

### Appending a sample to the row statistics

`Float/WelfordAppend` embeds every old lane into a row of length `N+1` and
appends the new sample to its explicit addition tree. `appendPlan` proves that
this preserves reduction-plan validity and the old padding count. The count
of the extended tree is consequently the old floating tree count plus one;
this is structural evaluation of the tree, not an integer-conversion law.

`mean_append` derives the equality between the Welford mean update and the
mean of this extended tree. `shift_square` derives equality of the backwards
mean-shift square and correction square using distribution and cancellation.
Combining this with the centered-row lemmas and the local variance update,
`variance_append` derives the appended row's sum of squared deviations.
All extra premises are finite/nonzero conditions on the explicit operations;
the updated means, variances and reduction values are not equality premises.

The arithmetic fixture also separates the two count obligations. Interpreting
`fromNat(n)` as `n+1` satisfies the successor relation and every selected
arithmetic atom, but on singleton input `[2]` the two-pass variance is `1/2`
and the online variance is `1`. Thus the initialization relation
`fromNat(0) = literal(0)` cannot be omitted merely because a successor relation
has been obtained. Neither conversion relation is currently admitted.

The append step alone does not establish the full original-kernel equivalence.
Initialization, loop induction and batch-schedule comparison are connected
below, together with the public contract; count conversion admission remains.
No original source kernel or experiment rule was changed for this derivation.

### Literal-zero initialization and singleton statistics

`Float/WelfordInit` supplies the scalar induction base: starting the mean,
square-deviation sum and floating count at literal zero, the first Welford
update returns mean `x` and square-deviation sum zero. Self-subtraction and
zero multiplication are derived from the existing scalar atoms. The same
results hold for a singleton appended to any empty padding tree, and its
count is literal one. `initial_statistics` aligns these two computations;
`singleton_variance` also establishes the normalized singleton variance.

This base case avoids applying the nonzero-count mean theorem to an empty
row. The domain record checks the actual residual operations; it does not
require `x*x` to be finite. The arithmetic fixture checks that these domains
can hold even when the input's square is outside its finite-value domain.

The original kernel still initializes its converted loop count through
`fromNat(0)`, so the missing conversion binding is not discharged by these
literal-zero lemmas. The induction below retains these primitive conversion
premises, rather than treating initialization as evidence for them.

### Two-output FP specifications

`Structural.IO₁ₓ₂Equiv` and `Guarded.IO₁ₓ₂` now support `KernelIO₁ₓ₂` with the
existing `lhs ≡[R] rhs` notation. The signature retains both output regions,
lengths and address functions, as well as the input, kernel ports and guarded
domain. Both runs must succeed, both complete typed output windows must agree,
and each implementation must preserve cells outside its two output windows and
declared private scratch. Scratch cannot alias the input or either output.

`onlineIO` and `twopassIO` in
[Welford/Execution.lean](../bench/examples/Welford/Execution.lean) expose the original kernels through
this interface, retaining symbolic row length/stride and both bf16 stores.
`online_io_run` and `twopass_io_run` prove the resulting execution and frame
obligations. These are execution results, not the pending numerical equality
between online and two-pass statistics.

The dual-output regression proves a structural store reordering, rejects an
implementation changing only the second output, and checks output dtype,
window, scratch, failure and signature boundaries. Assumption printing stays
unchanged: the structural proof prints `none`; an opaque guarded equivalence
premise prints `unresolved FP proof` instead of claiming an atomic derivation.

### Induction for the original converted-index loop

`Float/WelfordInduction.state_statistics` connects the initialization and
append lemmas for every nonempty row length. Its recurrence uses the original
fp32 `fromNat(i)` operation at each step. The resulting mean and unnormalized
variance equal the statistics of an explicit prefix tree, with the original
empty padding tree retained. A corresponding `ReductionPlan` proves that each
input occurs exactly once and that padding is preserved.

`WelfordInduction.CountConversion` records the two primitive obligations now derived
by `Float/CountConversion.conversion` under the admitted row-length bound:
`fromNat(0) = literal(0)` and, for every `i < N`,
`fromNat(i + 1) = fromNat(i) + literal(1)`. From these, `converted_count`
derives the equality with the prefix tree's sum of ones. There is no supplied
reduction-count equality, statistics invariant or whole-kernel equality.
`IterationDomain` contains the finite/nonzero predicates for initialization
and every subsequent append; checking only the final iteration is insufficient.

`fp32_recurrence_prefix` in
[Welford/Execution.lean](../bench/examples/Welford/Execution.lean) identifies this recurrence with the
original source's executed loop, including every converted index.
`fp32_online_statistics_run` carries the conditional result through both
original bf16 stores and retains the two-output memory frame.
`normalized_statistics` also retains the final division by the converted row
length before deriving its prefix-tree form.

The arithmetic fixture supplies a concrete rational model satisfying the
conversion and iteration conditions. It also checks a countermodel where
conversion is correct at zero and at the final length `2`, but converts the
intermediate index `1` to `100`. All selected scalar arithmetic equations
still hold, yet the online mean of `[2,4]` is `204/101` instead of `3`, and its
normalized variance is `200/101` instead of `1`. This is an algebraic boundary
check, not a GPU result or an IEEE claim.

This does not add a completed FP example. The conversion premises must still
come from admitted atomic relations; no such admission is manufactured here.
The schedule comparison below connects the prefix tree to the batch kernel;
the final public contract must still bind its execution model and all guards.

### Scalar-derived comparison of arbitrary reduction schedules

`Float/ReductionSchedule` generates a deterministic rewrite path from each
tree to a common sorted lane order. Its constructors are only congruence,
composition, reversal and the scalar addition laws. Padding is present in the
original trees and removed only through explicit add-zero steps. The proof
uses the admitted fp32 add-commute, add-assoc and add-zero instances; no
reduction or statistics relation is added to the numerical registry.

Each path computes a list of the exact scalar operands used by its rewrites.
`ScheduleDomain` requires those values to be finite, including transformed
intermediate trees. It does not quantify over every possible regrouping.
`plans_value` compares arbitrary valid plans, including permutations and
different padding counts. The validity proof prevents dropped or duplicated
input lanes from entering the comparison.

`Float/WelfordSchedule` applies this result separately to the input sum, count
ones and squared deviations. It derives both statistics, binds the batch count
to the original integer conversion from the two primitive count premises,
and transfers the loop induction to the batch schedule.

`original_runs` in [Welford/Comparison.lean](../bench/examples/Welford/Comparison.lean) compares both
original kernels under an explicit fp32 execution model. The model resolves
default precision and expands each sum using its supplied valid schedule;
casts, integer conversions and all non-sum operations remain unchanged. The
theorem proves successful runs, equality of both bf16 output windows and both
memory frames. It accepts only primitive count obligations and value-domain
predicates in addition to the admitted scalar theory, never a supplied
reduction, statistics or whole-kernel equality.

The schedule fixture checks that finite original trees do not by themselves
discharge the selected path: a generated operand can leave the finite domain.
It also checks padding dependence in an arbitrary algebra, rejects duplicate
input lanes, and confirms that explicit fp64 reductions remain opaque. The
ordinary rational fixture satisfies the schedule conditions for arbitrary
valid plans, so these predicates are not vacuous.

This support comparison keeps its count premises explicit. The completed
Welford and LayerNorm specifications discharge them through the range-aware
count binding. The scheduled public IO contract below binds the execution
model and all loop/rewrite domains. The original
guarded IO relation continues to run an arbitrary opaque algebra directly; it
does not silently identify its `reduceSum` field with an addition tree.

### Scheduled IO and syntactic loop domains

`Float/ScheduledIO.IO₁ₓ₂` uses the existing `lhs ≡[R] rhs` notation. Its
signature includes the two typed output windows, default compute precision,
whether fp32 sums expand into explicit addition trees, and the exact domain
expression generator. Numerical equivalence requires successful runs of both
kernels under every valid schedule, agreement of both outputs, and both
memory frames. A structural execution proof also works under a shared profile.

`Float/GuardExpression` represents numerical expressions without interpreting
any floating law. Precision, storage dtype, casts, loads, integer conversion,
and reduction layout remain explicit. Conditions contain only a `GuardKind`
check, conjunction, or finite index enumeration. They cannot take arbitrary
semantic predicates or equations as input.

`Float/WelfordConditions` compiles every field of the initialization, update,
centering, append and schedule-domain records into these conditions. Its iff
and interpretation lemmas establish exact correspondence with those records.
All loop iterations and the intermediate operands on both normalization paths
remain covered; checking only the input leaves or final output is insufficient.

[Welford/Contract.lean](../bench/examples/Welford/Contract.lean) defines the original online and
two-pass IO objects independently of the real-correctness file. It keeps row
length and stride symbolic and uses the fp32 scheduled profile. The theorem
`original_runs_under_count` connects the syntactic domain to the original
execution comparison, including both bf16 outputs and both memory frames.
`CountConversion` remains an explicit premise of this reusable support theorem,
outside the domain syntax. `WelfordFPEquiv.welford_equiv` discharges it using
the accepted count atoms with `0 < N <= 2^24`. The positive-length condition
excludes the empty mean's division by zero; the Correct theorem retains its
separate total-real interpretation.

The fixtures check that the reified contract is satisfiable in the ordinary
rational model for arbitrary batch schedules, rejects a later inadmissible
sample despite admissible initialization, and retains intermediate rewrite
operands. Signature checks reject profile or domain changes. The assumption
printer reports an opaque scheduled-equivalence premise as unresolved, while
a structural store-reordering proof still prints `none`.

### LayerNorm execution and unrounded statistics

`original_statistics` in [Welford/Comparison.lean](../bench/examples/Welford/Comparison.lean)
exposes the mean and normalized
variance before their bf16 stores. Equality only after a noninjective output
cast would not justify substituting those statistics into another expression.
The LayerNorm fixture demonstrates this with an opaque cast that equates two
means while their final affine outputs differ; it is a logical countermodel,
not an IEEE claim or an experiment.

[FusedLayerNorm/Execution.lean](../bench/examples/FusedLayerNorm/Execution.lean)
imports both original Triton sources from its shared `Kernels.lean`.
The fused version uses the same Welford loop proof, retaining
its pid register and unchanged input memory for the second x read. Both
execution lemmas preserve symbolic row length, stride and epsilon, the original
sqrt/reciprocal and affine operation order, feature-offset gamma/beta loads,
and the final bf16 cast. They establish every output cell and the frame of all
other cells, including in-place output. The explicit three-input IO declaration
lists x/gamma/beta once each; the DSL's occurrence-based input metadata otherwise
lists x twice in the fused source. The original statement bodies are identical.

`Float/ScheduledIO.IO₃` retains all three input layouts, the output layout,
execution profile and domain syntax in its signature and uses the existing
`≡[R]` notation.
[FusedLayerNorm/Contract.lean](../bench/examples/FusedLayerNorm/Contract.lean)
supplies both original IO objects. Empty
rows have a structural execution comparison with no domain checks or numerical
laws. For nonempty rows, `original_runs_under_count` derives equality of the
unrounded statistics from the scalar theory, then applies congruence through
the common affine suffix. Gamma, beta and epsilon need no additional domain
restriction because this suffix is unchanged.

The support comparison takes the generic primitive `CountConversion`
obligations. `FusedLayerNormFPEquiv.layernorm_equiv` now discharges them from
the two accepted count atoms for `N <= 2^24`, including the empty output case.
No LayerNorm, sqrt, affine, or whole-reduction identity is admitted. The
completed FP example count is 17. Source-identity checks, independence from Correct,
layout/profile/domain signature counterexamples, memory-frame checks, and
opaque-premise assumption-printer checks cover this connection.

## Constraints on the supplemental atom set

Further atom sets need a derivation-level dependency check before a GPU run.
For example, add-zero is now admitted; sub-zero should then
be derived from it and an accepted CANCEL precision instance instead of automatically
adding another experiment. Whole softmax, Welford, LayerNorm and reduction
transformations remain excluded from the atomic registry.

Domains must be explicit when a new relation requires them. In particular,
unconditionally assuming `a * (1 / a) = 1` together with `0 * a = 0` identifies
`0` and `1`. The Lean fixture checks this contradiction. A GPU sample containing
no zero denominators does not justify dropping the nonzero condition from that
particular rule. The current DIV-RCP relation does not assert inverse
cancellation and is not affected by this issue.

The experiment's shape/distribution still select the assumptions; they do not
become fixed dimensions in the later kernel proof. Mathematical domains of
individual operations are a separate concern. Under the current unconditioned
`Normal(1,1)` profile, a raw log operand can be negative; such an experiment
must report its domain event. It must not silently take absolute values,
truncate, resample, or change sigma. Relations involving positive expressions,
such as `log(exp(a))`, have a different explicit expression graph and must be
recorded as such.

Implementation, sampling, checker, compiler and PTX identities remain bound to
each report. Changing any of them requires a matching experiment and CPU replay;
only current reports and generated tables are published. Missing, failed and
inconclusive instances cannot become available through a trust-report export.

## Validation

The [supplemental experiment package](../experiments/floating_point/supplement/README.md)
publishes every configured instance with z, B, tau, U and accept. Only ACCEPT
rows enter `SupplementalAdmission`. Both exporters check the current budget,
source hashes, coverage and gate status; the supplemental exporter also checks
its manifest identity and derives the combined count from the main report.
Export is an explicit trust-report operation, not another GPU replay.

```bash
lake env lean bench/tests/FPAdmissionCoverage.lean
python3 -m unittest scripts.test_fp_equational
python3 scripts/export_numerical_rules.py --trust-report --check
python3 scripts/export_supplemental_rules.py --trust-report --check
python3 -m unittest scripts.test_export_supplemental_rules scripts.test_fp_supplemental
python3 -m unittest scripts.test_fp_scalar_arithmetic
python3 -m unittest scripts.test_fp_control scripts.test_fp_dual_output scripts.test_fp_layernorm
python3 -m unittest scripts.test_fp_softmax_stable
python3 -m unittest scripts.test_fp_logsumexp
python3 -m unittest scripts.test_fp_online_softmax
lake build TritonBenchSpecExamples
```

The coverage fixture also runs the project axiom audit. None of its
countermodels is installed as a floating-point kernel model or a rule-table
entry.
