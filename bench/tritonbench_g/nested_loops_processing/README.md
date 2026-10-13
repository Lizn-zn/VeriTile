# nested_loops_processing

The new proof follows the complete source: three nested two-iteration loops,
cached `a1` and `a2` loads, eight fresh `a3` loads, and all 24 unmasked 2×2
stores. The candidate only commutes a signed integer multiplication in the
initial input address. This is an address canonicalization, with no claimed
performance improvement.

`Kernels.lean` defines the shared typed original/candidate family. `Correct.lean`
contains the new named real contracts and the unchanged historical public
statements. `FPEquiv.lean` states complete typed state equivalence, including
aliases and failures, with no extra FP assumptions. The historical definitions,
helpers, signatures, and documentation are retained in `Proofs/Legacy.lean`.
The old entry file has been replaced by these three public entry files.

The independent mathematical oracle in `Proofs/Math.lean` folds the source's
ordered writes, retaining cross-tile overlap. It does not assume each output
cell is written once. The region and flat contracts include successful execution,
full raw-cell frames, and every accessed destination cell. Source-facing public
contracts also expose per-tile address injectivity and access safety.

There are 15 homogeneous source profiles: signed and unsigned 8/16/32/64-bit
labels, bool, fp16, bf16, fp32, fp64, and two fp8 formats. Signed and unsigned
width labels share mathematical Int/Nat channels. The generic real oracle uses
an explicit floating storage normalization map; the typed FP backend retains
its own carrier and precision operations. These are not machine-width or IEEE
rounding proofs. Mixed source/destination conversions remain outside this family.

For nonnegative strides `sm` and `sn`, access bounds are `sm + 27*sn` for input
and `sm + 47*sn` for output, both strictly below their allocation extents.
Snapshot IO requires distinct input/output regions. The flat bridge additionally
requires the library's explicit zero-`undef` condition; region IO does not.
Negative pointer increments and fixed-width overflow are not certified.

The wrapper's 4×4 case is out of bounds, and its 2×2 case launches zero programs.
Per-program contracts do not establish grid coverage or cross-program race
freedom. A valid 8×8 witness records the actual overwrite `output[8] = 2`, final
cells `output[47] = 27`, `output[55] = 35`, and untouched sentinel `output[56] = -1`.

Sequential single-worker Lean checks passed for all 59 Lean modules, including
the full real proof chain, all 15 typed successful executions, concrete
int32/fp64 source and candidate witnesses, preserved legacy helpers, and both
public specification files. All eight public headlines use only `propext`,
`Classical.choice`, and `Quot.sound`; the FP assumption list is empty. Each
proof module also has a standard-axiom audit. The coordinated dependency-aware
build and independent comparator replay remain separate integration gates.
See `source_scope.md` for the precise source boundaries.
