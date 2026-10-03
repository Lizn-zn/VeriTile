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
| `SoftmaxStable` | Settle fp32 EXP-SUB, currently WARN_NOT_ACCEPTED, and bind the tested libdevice implementation at the exact precision; accepted bf16-output instances do not cover a tl.exp or fp32-output implementation. | Derive the reduction and division rewrites; exp/max operations cannot be erased. |
| `StableLogSumExp` | The fp32 libdevice EXP-SUB instance is WARN_NOT_ACCEPTED; select accepted EXP-SUB and LOG-EXP precision instances. LOG-MUL has domain events under the configured distribution. | Derive the sum factorization and log transformation from these atoms. |
| `OnlineSoftmax` | Scalar max identities and EXP-NEG-INF-SUB are admitted; fp32 EXP-SUB remains WARN_NOT_ACCEPTED and needs both numerical admission and a compatible libdevice use-site binding. | Prove the loop invariant in the original recurrence scope. The current online source has no output store, so it cannot be presented as a complete stored-output kernel equivalent to the batch kernel. |
| `Welford` | Basic identity, selected inverse-cancellation laws and the fp32 CANCEL instance are admitted under the current profile; integer-count conversion, precision and nonzero-count obligations remain. | Connect explicit reduction trees to the online loop and retain both output windows and memory framing. |
| `FusedLayerNorm` | The Welford prerequisites at the actual arithmetic precision. | Derive the statistics replacement and preserve the common normalization, affine operations and bf16 output conversion. |

This table lists prerequisites, not newly available assumptions. It does not
assert that any proposed numerical experiment will pass.

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
lake build TritonBenchSpecExamples
```

The coverage fixture also runs the project axiom audit. None of its
countermodels is installed as a floating-point kernel model or a rule-table
entry.
