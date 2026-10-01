# Floating-point rule experiments

Edit [profile.py](./profile.py), check atomic expression pairs with Triton kernels on an NVIDIA GPU,
then import the result directory on the development machine (Python + NumPy).
The importer recomputes both gates instead of trusting PASS labels.

**No GPU acceptance measurements are checked into this branch.** Scripts,
offline compilation checks and CPU regressions are implemented; the user will
run the GPU experiments. The complete plan and remaining Lean integration are
in [FloatingPointPrimitives.md](../../documents/FloatingPointPrimitives.md).

## GPU run and result return

Check out `codex/fp-rules` at the same commit on both machines. Use Linux
with an NVIDIA CUDA GPU. For a fresh Python 3.11–3.13 virtual environment:

```bash
python3 -m pip install -r experiments/floating_point/requirements.txt
python3 scripts/fp_experiment.py check
python3 scripts/fp_experiment.py run --smoke --output Logs/fp-smoke
```

The smoke run uses a masked `32×33` shape and four replicates, always marked
`SMOKE_ONLY` and never admitted. Inspect ERROR records before the formal run.
SQRT-RSQRT domain events and fp32 BF16-WIDEN-RETURN unsupported rows are expected.

Start with associativity under all three precision profiles:

```bash
python3 scripts/fp_experiment.py run --rules ADD-ASSOC --output Logs/fp-add-assoc
```

Run the complete table, or select rules and formats:

```bash
python3 scripts/fp_experiment.py run --output Logs/fp-all
python3 scripts/fp_experiment.py run --rules MUL-DISTRIB,FMA-CONTRACT --formats bf16,fp32 --output Logs/fp-mul-fma
```

Default settings: `4096×4096`, independent `Normal(1, 1²)` operands, 4096
**whole-tensor** replicates. Input/compute/output profiles are bf16/bf16/bf16,
bf16/fp32/bf16 and fp32/fp32/fp32. `ACC-WIDEN` compares the two additions
in `(a+b)+c` using input-format versus fp32 intermediates, with the same
output cast. Its widened format is fp32; this is not an entire reduction
or dot-accumulator replacement.
`std` is sigma, not variance. Edit it before running to choose another variance.
These are editable example parameters, not calibrated paper results.

The full run has 42 rows: 14 atomic relations × 3 format profiles, including
one unsupported fp32 BF16-WIDEN-RETURN row. Shape denotes a batch of local
expressions. No run is silently replaced by a smaller shape. Each replicate
generates a fresh tuple.

To resume, repeat the same command with `--resume`:

```bash
python3 scripts/fp_experiment.py run --rules ADD-ASSOC --output Logs/fp-add-assoc --resume
```

Completed records are replay-checked and kept. Interrupted/error instances
restart with the same seed. Domain/unsupported records are kept. Configuration,
source, device or software changes require a new output directory. A compiler
or CUDA ERROR gives a nonzero process exit and a saved error record; a numerical
REJECT/INCONCLUSIVE is an experimental outcome.

Return the **entire output directory**, including observations and PTX:

```bash
tar -czf fp-add-assoc.tar.gz -C Logs fp-add-assoc
```

After copying and unpacking it at the same development revision:

```bash
python3 scripts/fp_experiment.py import Logs/fp-add-assoc --output Logs/fp-accepted.json
```

The output contains every row's status and an `accepted` list. FAIL, WARN under
`pass_only`, incomplete/error/domain/unsupported rows and smoke results never
populate that list. The importer verifies configuration identity, source hashes,
PTX digests, observation shapes and file hashes, and recomputes the statistics.
Existing output files are never overwritten. Returned Python source is **not
executed**, and NPZ is loaded with `allow_pickle=False`.

Per-instance `observations.npz` stores per-replicate bucket delta/ULP and oracle
peak errors/epsilon. It supports CPU statistical replay, not raw-output replay;
frozen seeds and sources support a separate GPU rerun. The manifest includes
profile, implementation hashes, device/capability, driver, software and launch
settings. Keep bundles outside Git (`Logs/` and `results/` are ignored).
Hashes/replay detect mismatches and accidental corruption, not forged execution.
The GPU runner, compiler, hardware and fp64 oracle remain trusted components.

## Concrete coverage

All 14 atomic catalogue IDs have template pairs in [triton_rules.py](./triton_rules.py).
Evidence covers that exact pair/configuration, not all implementations with the
same rule name.

| Family | Implementation choices |
|---|---|
| Commute/associate/distribute/cancel | Explicit per-node bf16 conversions or fp32 arithmetic; ordinary FP fusion disabled |
| FMA | Separate multiply/add versus explicit fp32 `tl.fma`, then compute/output conversion; not native bf16 FMA |
| DIV-RCP; SQRT-RSQRT | `tl.div_rn` versus rounded reciprocal multiplication; reciprocal of `tl.sqrt` versus `tl.rsqrt` |
| ROUND-IDEM; BF16-WIDEN-RETURN | Repeated output-format cast; bf16→fp32→bf16 (requires bf16 input/output) |
| Cast movement/removal | Explicit bf16 quantization around addition; remove intermediate bf16 rounding in `(a+b)*c` |
| ACC-WIDEN | Three-term sum with input-format versus fp32 intermediates, same output format |

Each atom has a fixed number of scalar operations and casts. `CAST-REMOVE`
is specifically `(a+b)*c` with one bf16 intermediate cast removed; it does not
stand for arbitrary F/G. No whole-kernel algorithm is admitted as an atom.

Softmax (including online softmax), Welford, SwiGLU fusion, reduction/scan
reordering and dot/split-K transformations belong to Lean derivations. Layout
and store/load properties require structural and memory proofs. They have been
removed from the catalogue, templates and runner. Explicitly selecting an old
composite ID is rejected before GPU execution; importing such a row is rejected
regardless of its PASS labels. Additional scalar identities, such as an explicit
exponential relation, must be specified and checked separately when needed.

Only the selected distribution is sampled. There are no extra asymmetric probes,
absolute-value transforms, truncation or resampling. Unconditioned normal input
commonly violates SQRT-RSQRT's positive domain: INCONCLUSIVE, not accepted.
A sampled zero divisor is also inconclusive. Associativity's symmetric operands
can cancel mean delta; this does not add a new rejection criterion.

Triton operation contracts: [fma](https://triton-lang.org/main/python-api/generated/triton.language.fma.html),
[div_rn](https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html).

## Fixed statistical protocol

[fp_two_gates.py](../../scripts/fp_two_gates.py) implements
`two-gates-fixed-pwm-v1`; mathematical definitions and interpretation limits are
in [TwoGatesAcceptance.md](../../documents/TwoGatesAcceptance.md).

- Replicate = entire input tuple; both sides and the fp64 oracle use the same
  quantized inputs. fp64 is a reference, not exact real arithmetic.
- Bias buckets = output columns; one-dimensional output has one bucket. Compute
  signed means within each replicate, then sample std across replicates (ddof=1).
  Defaults: z=5, SNR=0.01, ULP floor=1. Zero-variance branches are explicit.
- Bucket ULP = mean over replicates of spacing at that bucket's peak absolute
  reference. Spacing is toward increasing magnitude, including zero/subnormals;
  at maximum finite values use the last binade spacing. Vars epsilon = one
  output-format ULP at the peak absolute fp64 oracle value.
- Vars amplification = `(candidate_error-epsilon)/reference_error`, with explicit
  zero/infinite branches. POT threshold = positive K's 90th percentile; tail
  rate uses **all** replicates. PWM fit, 256 conditional bootstrap fits, minimum
  64 strict exceedances, horizon 625000, upper estimate `return_level+3*SE`,
  cutoffs 2/10. Positive fitted shape is not clipped to zero.
- All observed K=0 has an explicit PASS branch, without a fabricated tail fit.
  Constant positive tails, insufficient excesses or any invalid fit/bootstrap
  give INCONCLUSIVE. Nonfinite oracle/reference is inconclusive; a nonfinite
  candidate with finite reference/oracle rejects the instance.
- Budget is fixed, without adaptive stopping. This and retaining positive shape
  deliberately differ from heuristics in the supplied report. The upper estimate
  is model-based, not a distribution-free guarantee. Default policy `pass_only`;
  `allow_warn` must be selected before running.

## Lean boundary and verification

Correctness stays real-valued. Public equivalence stays `lhs ≡[R] rhs`;
`#print_spec` exposes its atoms. The existing
[TritonBench example](../../bench/examples/TritonBenchVectorAdditionFP.lean)
proves conditional composition under one ADD-COMMUTE assumption.

Imported JSON is the numerical admission table. It **does not yet construct a
Lean `Rules` value or discharge `EvidenceValidated`**. Parameterized rule-fragment
binding and invocation of the existing `scripts/prove.sh` agent/comparator are
the next integration step; no separate proof searcher is planned. The example's
N=98432/block=1024 is not covered by the default 4096×4096 experiment. Atom
admission does not establish whole-kernel gates or distributions of intermediates.

CPU statistics/configuration/replay checks:

```bash
python3 -m unittest scripts.test_fp_two_gates scripts.test_fp_rule_registry -v
python3 -m unittest scripts.test_fp_scalar scripts.test_specification_surface -v
```

With Triton installed, compile without a GPU:

```bash
python3 scripts/check_fp_triton_compile.py
python3 scripts/check_fp_triton_compile.py --rows 4096 --columns 4096
```

Both shapes compile 82 supported atomic specializations for sm_80 using Triton 3.5.1. Optional
CPU indexing/formula/oracle checks use PyTorch and Triton's interpreter:

```bash
TRITON_INTERPRET=1 python3 -m unittest scripts.test_fp_triton_templates -v
```

These checks never create GPU gate records. Existing Lean bit-value modules,
scalar differential tests and six kernel-checked counterexamples remain
software references, not certificates of GPU instruction behavior.
