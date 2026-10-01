# Floating-point rule experiments

Edit [config.py](./config.py), check atomic expression pairs with Triton kernels on an NVIDIA GPU,
then import the result directory on the development machine (Python + NumPy).
The importer recomputes both gates instead of trusting PASS labels.

The runner saves paired kernel observations and compiled PTX. CPU replay checks
the bundle and computes the numerical admission table. Lean integration is
described in [FloatingPointPrimitives.md](../../documents/FloatingPointPrimitives.md).

## GPU run and result return

Check out `codex/fp-rules` at the same commit on both machines. Use Linux
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

Per-instance `observations.npz` stores per-replicate bucket delta/ULP and oracle
peak errors/epsilon. It supports CPU statistical replay, not raw-output replay;
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
- Bias buckets are the output's last axis. For the configured matrix shape, each
  replicate averages signed differences over rows, retaining one bucket per column.
  Sample std across replicates uses ddof=1.
- Bias ULP uses the pooled maximum absolute output over **both** candidate/reference
  and all replicates. We store each replicate's ULP and take their maximum (equivalent
  for finite scales). Cast to comparison dtype before `nextafter`; at maximum finite
  use the inward spacing. Invalid differences or floors FAIL.
- Vars epsilon is one comparison/output-format ULP at the candidate's peak magnitude.
  Nonfinite candidate/reference/golden errors or epsilon produce K=+inf and FAIL.
- POT targets max(40, round(10% of positive K)) tail samples, using an
  order-statistic threshold and strict exceedances. Fewer than 12 positive K or fewer
  than two strict exceedances return the empirical maximum with `valid=False` and
  `empirical_fallback=True`. This may PASS/WARN/FAIL, but is not a confidence bound.
- PWM retains raw xi for diagnostics and clips xi to <=0 when computing return levels.
  The bootstrap uses seed 0, 1000 resamples and population std (ddof=0); finite bootstrap
  return levels are retained. U = level + NormalDist().inv_cdf(1-alpha)*SE,
  with alpha=1.35e-3, horizon=625000, warn/fail thresholds=2/10.
- Check after every full batch. Stop immediately for a nonfinite magnitude bound;
  otherwise after the minimum budget, stop for empirical fallback or when the band
  [2*level-U, U] crosses neither threshold. Stop at the full-batch maximum otherwise.
  Replay recomputes **every checkpoint** and rejects a truncated or overrun stream.
- Default admission is `pass_only`; `allow_warn` explicitly admits warnings.
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
The H200 run (reported by the runtime as NVIDIA L20X) has 30 accepted instances,
6 warnings that remain unaccepted, 2 rejected instances, 3 square-root domain
events and 1 unsupported precision profile. The DLC task is named
`traces_kernel_equivalence_testing`.

[Experiment settings](./report/experiment.json) record the input distribution,
precision profiles, gates, device/compiler details and checked source hashes.
The [warning audit](./report/warning_audit.json) recomputes all six warnings from
the saved observations. [Exact counterexamples](./report/exact_counterexamples.json)
show why the four warned transformations are not unconditional floating-point
identities. Lower oracle error does not cancel a directional-bias warning.
Statistical acceptance applies only to the tested contract; it is not proof of
strict floating-point equivalence. Raw observations and compiled PTX remain in
the local result bundle and are required to independently replay the table.

Maintain a complete rule/format table while formal bundles are written under one
run directory:

```bash
python3 scripts/report_numerics.py Logs/numerics
python3 scripts/report_numerics.py Logs/numerics --watch --job-id <dlc-job-id>
```

After the run completes, refresh the checked-in table from its evidence bundle:

```bash
python3 scripts/report_numerics.py Logs/numerics --report-dir experiments/floating_point/report --job-id <dlc-job-id>
```

Keep the experiment settings, warning audit and counterexamples alongside the
current table consistent with that run. Replace current files when updating them.

The table includes every rule and precision in `config.py`, even if its run has
not started. Smoke bundles are excluded. `z` is the maximum absolute bucket z;
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
[TritonBench example](../../bench/examples/TritonBenchVectorAdditionFP.lean)
proves conditional composition under one ADD-COMMUTE assumption.

The published report can now be frozen into Lean with
`python3 scripts/export_numerical_rules.py --trust-report`. This explicit mode
trusts the report, checks its source hashes/configuration and exports only its
30 accepted rows. It does not pretend to replay missing raw observations.

The [worked example](./EXAMPLE.md) binds its fp32 ADD-COMMUTE row to the actual
TritonBench vector_addition fragments at 4096×4096/block=1024. Its public theorem
still takes only `R : Rules`; the row, precision, shape and PASS results are fixed.
`R.add_comm` is the explicit external numerical assumption for that bound atom,
not a global IEEE axiom. Lean checks the remaining whole-kernel derivation and
`#print_spec` shows the claim, model premise, rule ID, scope and both gates.
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
