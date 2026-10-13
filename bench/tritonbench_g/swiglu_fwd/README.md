# swiglu_fwd

- Source file: `swiglu_fwd.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The migrated layout uses three public entries: the shared source is in `Kernels.lean`, independent real
contracts in `Correct.lean`, complete FP equivalence in `FPEquiv.lean`, and
supporting lemmas in `Proofs/`. All ten legacy declaration headers/docstrings are preserved.

The typed family covers six independently chosen X/Y/OUT storage formats
(fp32, fp16, bf16, fp64, fp8e4nv, fp8e5), or 216 combinations per variant.
Both program axes, all three independent row strides, masks, fp32 load casts,
sigmoid and source multiplication order are retained. The candidate only
commutes the three natural row-address products. Store conversions make the
source's implicit destination conversion explicit. Exact source regressions
cover every load, final store and fp32 formula for every combination.

The independent formula is x*y/(1+exp(-x)), with a proved positive denominator.
The real contract includes successful flat execution, active input/output
bounds and a complete unwritten-memory frame for both variants. Both inputs
are loaded before the store, allowing named input/output aliases. The new
contract adds no positive tile-size or whole-row-within-one-tile premise;
its column program axis handles arbitrary column blocks. Host grid coverage, scheduling and construction of the offset X/Y views from
xy.chunk remain outside the per-program proof.

The complete FP equivalence preserves state and failure without numerical
atoms. Nonempty typed execution uses fp16 X, fp64 Y and bf16 OUT, two nonzero
program axes and distinct row strides. It produces 3*sigmoid(1) and preserves
the masked adjacent value 7. This is an exact-real backend witness with explicit
casts, not a claim of hardware rounding accuracy.

Integer/boolean data, other storage formats, negative strides, fixed-width
address overflow, byte reinterpretation of differently sized aliased views,
target-specific fp8 support and cross-program races are outside this port's
current source profile. Independent comparator replay remains pending.

Independent source/spec review found no correctness defect. All helper and
public file gates pass, and the integrated Lake dependency build passes
(3447 jobs). Four public contracts retain `test-gap` status pending independent
comparator replay.
