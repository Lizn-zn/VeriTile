# Remaining FP example prerequisites

The migration has 18 real correctness files and 13 FP equivalence files.
The current main and supplemental reports select numerical assumptions using
a local-ULP mean-bias budget and a peak absolute-error ratio gate.
The reciprocal softmax cases retain their explicit operand domains. RowWiseSum
binds its conditional derivation to the admitted fp32 ADD-COMMUTE and ADD-ASSOC
instances. Five other algorithmic transformations still need derivations.

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
| `SoftmaxStable` | The fp32 EXP-SUB instance is admitted; bind the tested libdevice implementation at the exact precision. This admission does not cover a tl.exp implementation. | Derive the reduction and division rewrites; exp/max operations cannot be erased. |
| `StableLogSumExp` | The fp32 libdevice EXP-SUB instance is admitted; select a compatible accepted LOG-EXP precision instance. LOG-MUL has domain events under the configured distribution. | Derive the sum factorization and log transformation from these atoms. |
| `OnlineSoftmax` | Scalar max identities, EXP-NEG-INF-SUB and fp32 EXP-SUB are admitted; EXP-SUB still needs a compatible libdevice use-site binding. | Relate the now-proved opaque loop recurrence to the batch expression using scalar atoms. The current online source has no output store, so it cannot be presented as a complete stored-output kernel equivalent to the batch kernel. |
| `Welford` | Basic identity, selected inverse-cancellation laws and the fp32 CANCEL instance are admitted under the current profile; the two integer-count conversion laws remain unadmitted. | Both original kernels are compared conditionally under arbitrary valid fp32 reduction schedules. Bind that scheduled execution model and all iteration/rewrite domains in the final specification. |
| `FusedLayerNorm` | The Welford prerequisites at the actual arithmetic precision. | Derive the statistics replacement and preserve the common normalization, affine operations and bf16 output conversion. |

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

These are reusable prerequisites, not another completed example. The original
`SoftmaxStable` kernels still use `tl.exp`; the accepted EXP-SUB experiment uses
`libdevice.exp`. Selecting a libdevice variant or obtaining evidence for the
original intrinsic remains necessary before the full FP equivalence is closed.

### Implemented loop execution

The opaque FP interpreter now executes compute-level counted loops, static and
dynamic ranges, and conditional branches, including nesting. It captures
dynamic range bounds once, resets the exact natural index before each
iteration, propagates reached failures, and skips inactive bodies. The range
zero-step behavior matches the existing operational semantics. `Float/Control`
provides successful-execution induction principles for both loop forms; it
does not add arithmetic assumptions.

The independent `bench/examples/support/WelfordExecution` and
`OnlineSoftmaxExecution` modules retain the original Triton sources. Welford's
online loop computes the opaque recurrence and writes both bf16 output cells;
the two-pass execution retains both original sum operations. Their proofs
cover symbolic row length and stride, including empty rows, and frame every
untouched cell. OnlineSoftmax computes its recurrence in `m` and `l` and
preserves all memory, matching the original source's lack of an output store.
Source-equality tests compare these copies with the existing correctness
files; the execution proofs do not import those files.

These are execution prerequisites, not new completed FP equivalences. The
Welford recurrence still needs to be related to its two-pass expression using
the accepted scalar theory at the actual arithmetic precision. Count
conversion remains an opaque operation: loop support does not imply
`toReal(i + 1) = toReal(i) + 1`. The OnlineSoftmax comparison still needs the
matching exponential implementation and a scalar-derived invariant relating
the recurrence to the batch expression.

### Welford mean step and integer conversion

`Float/Welford.mean_step` derives
`(m + (x-m)/(n+1)) * (n+1) = m*n + x` from the accepted scalar arithmetic
atoms. Every intermediate finite/nonzero requirement is explicit. The result
is connected to the original loop update by `WelfordExecution.fp32_mean_step`.
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
before the original count-based invariant can close. The existing configured
normal distribution samples floating operands. A separate integer input
distribution and its conversion experiments await user confirmation; no
unmeasured conversion law has been added to the admission table.

### Welford variance step and recentering

Let `q = (x-m)/(n+1)` and `m' = m+q`. `Float/Welford.residual_step`
derives `x-m' = n*q`; `variance_step` then derives
`(x-m)*(x-m') = (x-m')² + n*q²`. The original loop's variance update is
connected to this identity by `WelfordExecution.fp32_variance_step`, retaining
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
equivalence. Relating the full recurrence to the two-pass statistics still
requires the count-conversion binding, the global mean/variance invariants
and the corresponding reduction schedules. The completed example count
remains 13.

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
guard rejects an empty tree. Connecting these row identities to the original
recurrence still requires the conversion binding and global invariant proof.

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
below; count conversion admission and the final public contract remain.
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

`WelfordExecution.onlineIO` and `twopassIO` expose the original kernels through
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

`CountConversion` records exactly two still-unadmitted primitive obligations:
`fromNat(0) = literal(0)` and, for every `i < N`,
`fromNat(i + 1) = fromNat(i) + literal(1)`. From these, `converted_count`
derives the equality with the prefix tree's sum of ones. There is no supplied
reduction-count equality, statistics invariant or whole-kernel equality.
`IterationDomain` contains the finite/nonzero predicates for initialization
and every subsequent append; checking only the final iteration is insufficient.

`WelfordExecution.fp32_recurrence_prefix` identifies this recurrence with the
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

`bench/examples/support/WelfordComparison.original_runs` now compares both
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

The comparison remains conditional rather than a completed public FP example.
The count relations still need admission, and the final `≡[R]` interface must
bind this scheduled execution model plus all loop and rewrite domains. The
existing guarded IO relation runs an arbitrary opaque algebra directly and
cannot silently identify its `reduceSum` field with an addition tree.

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
python3 -m unittest scripts.test_fp_control
lake build TritonBenchSpecExamples
```

The coverage fixture also runs the project axiom audit. None of its
countermodels is installed as a floating-point kernel model or a rule-table
entry.
