# Local completion audit

The local proof migration is complete for the 15 homogeneous storage profiles
and the explicit address/placement scope described in `source_scope.md`.
Repository integration and independent comparator replay are separate gates.

- Actual source: all three two-iteration loops, cached loads, pointer updates,
  and 24 stores remain in `Kernels.lean`. Exact load/store AST checks cover every
  profile and both versions; lane enumeration is pinned to the source model.
- Independent mathematics: `CopyMath.expected` folds the exact ordered write
  schedule. `PayloadReal.CopyInvariant` connects every actual intermediate state
  to its prefix; it does not assume kernel output or successful execution.
- New real contracts: four original/candidate region/flat headlines have complete
  successful execution, ordered oracle, raw memory frame, native lane injectivity,
  and actual load/store access safety. Floating `storedCell` normalization,
  nonnegative strides, input/output separation, allocation bounds, and the flat
  bridge's zero-undef requirement are explicit.
- Typed FP: full optional state equality holds for all 15 profiles, arbitrary
  backend states and aliases, with an empty extra-assumption list. The candidate
  commutes only an initial signed address multiplication.
- Nonvacuity: all 15 profiles have a successful typed execution theorem. Actual
  int32 and fp64 executions in a nonempty WithBot Real backend return cells
  8=2, 47=27, 55=35, and retain cell 56=-1, for both original and candidate.
- Legacy: all 52 historical declaration headers/bodies and 37 declaration
  docstrings are unchanged across `Proofs/Legacy.lean` and `Correct.lean`.
  Their three public headlines retain their original partial/copy-slice scope.
- Trust: all 59 modules passed sequential single-worker file gates and module
  axiom audits. The eight public-head audit passed with only propext,
  Classical.choice, Quot.sound; `source_equivalence` prints no FP assumptions.

Final local evidence:

- `/private/tmp/nested-correct1.log`: public real/legacy entry, exit 0.
- `/private/tmp/nested-fpequiv1.log`: public FP entry, exit 0; assumptions none.
- `/private/tmp/nested-public-audit2.log`: all eight headlines, exit 0.
- `/private/tmp/nested-fpexecutionwitness-batch.log`: int32/fp64 typed numeric
  source/candidate witnesses, exit 0.
- `/private/tmp/nested-contracts-batch.log`: complete named real contracts, exit 0.

Remaining coverage boundaries are explicit, not discharged by these checks:
native lowering and fixed-width overflow, mixed/discrete conversion families,
negative pointer increments, host grid coverage, and cross-program race freedom.
The wrapper's unsafe 4×4 and empty-launch 2×2 cases are not certified witnesses.
