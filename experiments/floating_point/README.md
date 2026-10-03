# Floating-point rule experiments

For additional basic relations, use the
[supplemental experiment instructions](./supplement/README.md).
Both catalogues use local-ULP mean bias and a ratio of peak absolute oracle errors.

The bias protocol pools IID scalar instances into one mean per replicate in local ULPs.
The magnitude gate compares peak absolute oracle errors without ULP normalization.
The published tables use fresh GPU samples from this protocol and independent CPU
replay. Their recorded source hashes bind the sampling and aggregation semantics.

Edit [config.py](./config.py), check atomic expression pairs with Triton kernels on an NVIDIA GPU,
then import the result directory on the development machine (Python + NumPy).
The importer recomputes both gates instead of trusting PASS labels.

The runner saves paired kernel observations and compiled PTX. CPU replay checks
the bundle and computes the numerical admission table. Lean integration is
described in [FloatingPointPrimitives.md](../../documents/FloatingPointPrimitives.md).

## GPU run and result return

Check out the same experiment revision on both machines. Use Linux
with an NVIDIA CUDA GPU. For a fresh Python 3.11–3.13 virtual environment:

```bash
python3 -m pip install -r experiments/floating_point/requirements.txt
python3 scripts/check_numerics.py check
python3 scripts/check_numerics.py run --smoke --output Logs/fp-smoke
```

The smoke run uses a masked `32×33` shape and four replicates, always marked
`SMOKE_ONLY` and never admitted. Inspect ERROR records before the formal run.
SQRT-RSQRT domain events and fp32 BF16-WIDEN-RETURN unsupported rows are expected.

Start with associativity under all three precision profiles:

```bash
python3 scripts/check_numerics.py run --rules ADD-ASSOC --output Logs/fp-add-assoc
```

Run the complete table, or select rules and formats:

```bash
python3 scripts/check_numerics.py run --output Logs/fp-all
python3 scripts/check_numerics.py run --rules MUL-DISTRIB,FMA-CONTRACT --formats bf16,fp32 --output Logs/fp-mul-fma
```

Default settings: `4096×4096`, independent `Normal(1, 1²)` operands, at least 4096
**whole-tensor** replicates, adaptively extended toward 50000 in batches of 512
(the final full batch can reach 50176). Input/compute/output profiles are bf16/bf16/bf16,
bf16/fp32/bf16 and fp32/fp32/fp32. `ACC-WIDEN` compares the two additions
in `(a+b)+c` using input-format versus fp32 intermediates, with the same
output cast. Its widened format is fp32.
`std` is sigma, not variance. Edit it before running to choose another variance.
These are editable example parameters, not calibrated paper results.

The full run has 42 rows: 14 atomic relations × 3 format profiles, including
one unsupported fp32 BF16-WIDEN-RETURN row. Shape denotes a batch of local
expressions. No run is silently replaced by a smaller shape. Each replicate
generates a fresh tuple.

To resume, repeat the same command with `--resume`:

```bash
python3 scripts/check_numerics.py run --rules ADD-ASSOC --output Logs/fp-add-assoc --resume
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
python3 scripts/check_numerics.py import Logs/fp-add-assoc --output Logs/fp-accepted.json
```

The output contains every row's status and an `accepted` list. FAIL, WARN under
`pass_only`, incomplete/error/domain/unsupported rows and smoke results never
populate that list. The importer verifies configuration identity, source hashes,
PTX digests, observation shapes and file hashes, and recomputes the statistics.
Existing output files are never overwritten. Returned Python source is **not
executed**, and NPZ is loaded with `allow_pickle=False`.

Per-instance `observations.npz` stores one bias mean in local ULPs per replicate
and two peak oracle errors in absolute output units. It supports CPU statistical replay, not raw-output replay;
frozen seeds and sources support a separate GPU rerun. The manifest includes
profile, implementation hashes, device/capability, driver, software and launch
settings. Keep bundles outside Git (`Logs/` and `results/` are ignored).
Hashes/replay detect mismatches and accidental corruption, not forged execution.
The GPU runner, compiler, hardware and fp64 oracle remain trusted components.

## Concrete coverage

All 14 atomic catalogue IDs have template pairs in [kernels.py](./kernels.py).
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
and store/load properties require structural and memory proofs. The catalogue,
templates and runner admit only local atomic relations. Explicitly selecting a
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

## Statistical protocol

[numerical_gates.py](../../scripts/numerical_gates.py) implements the directional-bias
and error-amplification checks. Statistics and bundle replay use NumPy on CPU.

- Same quantized inputs feed reference, candidate and the fp64 oracle. The default
  probe draws independent `Normal(1, 1²)` operands; parameters live in `config.py`.
- For each element, set `s = ULP(abs(golden))` in the output dtype. Round the golden
  value to that dtype before `nextafter`; use the inward spacing at maximum finite
  and minimum subnormal spacing at zero. Both sides share this golden-based scale;
  output outliers cannot enlarge another element's allowance.
- For bias, normalize **before** aggregation: `d = (candidate-reference)/s`.
  Each replicate averages `d` over all IID scalar instances, storing one mean
  (`delta` has shape `[R, 1]`). Array columns have no separate channel semantics.
  Mean and sample std (ddof=1) are computed across these R replicate means, not
  across R times the number of elements. `z = sqrt(R)*abs(mean)/std` is dimensionless.
  With `SE = std/sqrt(R)`, the mean must satisfy `abs(mean) + 5*SE <= 0.05`
  local ULP. A band entirely outside the tolerance FAILs; a boundary-crossing
  band is INCONCLUSIVE. z is diagnostic only. These are engineering SE bands,
  not calibrated simultaneous or optional-stopping confidence guarantees.
  Pooling is scoped to these identical scalar expressions and IID probes; distinct
  channels/heads or probe distributions require their own explicit grouping.
- The magnitude gate uses `Er = max(abs(reference-golden))` and
  `Ec = max(abs(candidate-golden))` within each replicate, followed by `K = Ec/Er`.
  There is no ULP normalization or additive allowance for this gate. Both errors
  zero gives K=0; `Er == 0` with `Ec > 0` gives infinity. These are separate
  absolute-error peaks, which may occur at different positions in the tensor.
  This is the [FlashAttention maximum-error comparison](https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py):
  FA tests use `Ec <= 2*Er`; this project uses a magnitude PASS threshold of 3.
  The local-ULP bias budget and tail extrapolation below are extra criteria,
  so the full two-gate decision is not identical to FA's sample test.
  Nonfinite reference/candidate/golden observations produce K=+inf and FAIL.
- POT targets max(40, round(10% of positive K)) tail samples, using an
  order-statistic threshold and strict exceedances. Fewer than 12 positive K or fewer
  than two strict exceedances return the empirical maximum with `valid=False` and
  `empirical_fallback=True`. This may PASS/WARN/FAIL, but is not a confidence bound.
- PWM retains raw xi for diagnostics and clips xi to <=0 when computing return levels.
  The bootstrap uses seed 0, 1000 resamples and population std (ddof=0); finite bootstrap
  return levels are retained. U = level + NormalDist().inv_cdf(1-alpha)*SE,
  with alpha=1.35e-3, horizon=625000, warn/fail thresholds=3/10:
  U <= 3 is PASS, 3 < U <= 10 is WARN, and U > 10 is FAIL.
- Check after every full batch. Stop immediately for a nonfinite magnitude bound;
  otherwise after the minimum budget, stop for empirical fallback or when the band
  [2*level-U, U] crosses neither threshold. Stop at the full-batch maximum otherwise.
  Replay recomputes **every checkpoint** and rejects a truncated or overrun stream.
- Default admission is `pass_only`; `allow_warn` explicitly admits magnitude warnings,
  but never bias INCONCLUSIVE. The selected tau is a mean-bias budget, not a per-element error bound.
  Smoke is always `SMOKE_ONLY` and never admitted.

Configuration, replay and sampling regression checks:

```bash
python3 -m unittest scripts.test_numerical_gates scripts.test_numerical_sampling scripts.test_numerical_registry scripts.test_numerical_kernels scripts.test_numerical_reporting -v
```

Regression cases cover zero/constant/sparse, exponential/heavy/bounded,
quantized ties and nonfinite errors, plus adaptive stopping and replay integrity.

## Current results table

The checked-in [current table](./report/summary.md) contains all 42 instances,
with full-precision [CSV](./report/summary.csv) and [JSON](./report/summary.json).
The current H200 run (reported as NVIDIA L20X) uses seed 20261003, tau=0.05
local ULP and a five-SE bias band across replicate means. Replicates use independent input tuples at the fixed seed.
Current totals are in the table.
The DLC task is named `traces_kernel_equivalence_testing` (`dlc1q8e8anqbkjgg`).

[Experiment settings](./report/experiment.json) record the input distribution,
precision profiles, gates, device/compiler details and checked source hashes.
The [nonacceptance audit](./report/warning_audit.json) records all warnings and
rejections from the replayed observations, with bias means in local-ULP units.
[Exact counterexamples](./report/exact_counterexamples.json) show why four
transformations are not unconditional floating-point identities; those examples
are independent of the statistical classifications. Lower oracle error does not
cancel a failed or inconclusive bias-budget check.
Statistical acceptance applies only to the tested contract; it is not proof of
strict floating-point equivalence. Raw observations and compiled PTX remain in
the local result bundle and are required to independently replay the table.

Maintain a complete rule/format table while formal bundles are written under one
run directory:

```bash
python3 scripts/report_numerics.py Logs/replicate-mean-verification/original
python3 scripts/report_numerics.py Logs/replicate-mean-verification/original --watch --job-id <dlc-job-id>
```

After both runs complete, publish their independently replayed tables, settings
and audits together. The run directory must contain `original/atomic`,
`original/report`, `supplement/atomic`, `supplement/report`, a DLC `status.json`
with `Status: Succeeded`, and an `exit_code` file containing `0`:

```bash
python3 scripts/publish_numerical_results.py Logs/replicate-mean-verification --hardware-model H200
python3 scripts/export_numerical_rules.py --trust-report
python3 scripts/export_supplemental_rules.py --trust-report
```

The publisher checks both catalogs and compares the CPU tables against the GPU
environment's reports before replacing current files. Raw observation bundles
remain outside Git.

The table includes every rule and precision in `config.py`, even if its run has
not started. Smoke bundles are excluded. `z` is the absolute z across replicate means;
`B` is the bias upper bound in local ULPs, `tau` its budget;
`U` is the magnitude-gate value, with `empirical_max` explicitly distinguished
from a fitted upper estimate. Missing values remain empty rather than becoming
zero. `accept` stays pending until CPU replay verifies the bundle.

`summary.md`, `summary.csv` and `summary.json` are overwritten atomically with the
current table. CSV/JSON retain full numerical precision. The watcher refreshes
every 30 seconds, replays a stage when its `<stage>-report.json` is published,
and performs a final replay when the DLC job ends. Job failure or unfinished
rules remain visible in the table. Without `--watch`, all available bundles are
replayed once. Duplicate rule/format results or incompatible configurations are
rejected instead of silently merging different experiments.

## Lean boundary and verification

Correctness stays real-valued. Public equivalence stays `lhs ≡[R] rhs`;
`#print_spec` exposes its atoms. The existing
[TritonBench example](../../bench/examples/TritonBenchVectorAdditionFPEquiv.lean)
proves conditional composition under one ADD-COMMUTE assumption.

The published report can now be frozen into Lean with
`python3 scripts/export_numerical_rules.py --trust-report`. This explicit mode
trusts the report, checks its source hashes/configuration and exports only its
accepted rows. It does not pretend to replay missing raw observations.

The current report admits fp32 ADD-ASSOC. The row-wise-sum example binds it
alongside fp32 ADD-COMMUTE, retaining explicit evidence premises for both atoms.
The resulting derivation does not claim a whole-reduction IEEE or statistical guarantee.

The [worked example](./EXAMPLE.md) has separate
[real correctness](../../bench/examples/TritonBenchVectorAdditionCorrect.lean) and
[FP equivalence](../../bench/examples/TritonBenchVectorAdditionFPEquiv.lean) files, each
with its own original kernel transcription. The FP file does not import the
correctness example. It binds its fp32 ADD-COMMUTE row to the actual
TritonBench vector_addition fragments with symbolic element count and block size.
The experiment selects the fp32 assumption; the subsequent proof does not match
kernel dimensions against experimental dimensions. `Rules blockSize` instantiates
the typed atom syntax at the kernel's tile size, without a size restriction.
`R.add_comm` is the explicit modeling assumption for that atom,
not a global IEEE axiom. Lean checks the remaining whole-kernel derivation and
`#print_fp_assumptions` shows only the referenced atom name, `add_commute`.
Use `#print_spec ... full` for report identity, provenance and dependency auditing.
Other accepted rows are exported as data; further use-site syntax bindings remain
necessary. Existing `scripts/prove.sh` and the official comparator remain the
proof/checking entry points. Atom admission does not establish whole-kernel gates
or distributions of intermediates.

CPU statistics/configuration/replay checks:

```bash
python3 -m unittest scripts.test_numerical_gates scripts.test_numerical_registry -v
python3 -m unittest scripts.test_fp_scalar scripts.test_specification_surface -v
```

With Triton installed, compile without a GPU:

```bash
python3 scripts/check_numerical_kernels.py
python3 scripts/check_numerical_kernels.py --rows 4096 --columns 4096
```

The compiler check covers 82 supported atomic specializations per shape. Optional
CPU indexing/formula/oracle checks use PyTorch and Triton's interpreter:

```bash
TRITON_INTERPRET=1 python3 -m unittest scripts.test_numerical_kernels -v
```

These checks never create GPU gate records. Existing Lean bit-value modules,
scalar differential tests and six kernel-checked counterexamples remain
software references, not certificates of GPU instruction behavior.
