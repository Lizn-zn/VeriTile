# Local verification audit

Status: all implementation leaf gates and the exhaustive theorem audit passed.
Root dependency-aware integration and independent replay remain pending.
Implementation leaf gates were run before the final shared negative-literal DSL
refresh; the root integration rebuild must replay those dependencies.

- Typed source: `Kernels.lean` passed the single-thread compiler on the revised
  quotation form (`/private/tmp/llama-kernels2.log`). Both complete source loops
  and all 216 floating storage triples are present.
- Exact memory/cast/precision AST checks: passed for all 216 profiles and both
  variants (`/private/tmp/llama-proofs-precision.log`), with a standard-axiom module audit.
- Exact typed erasure and the independent logical-row mathematics, source
  decomposition, variance, normalization and writeback modules passed their
  sequential single-thread gates. The actual variance body has no extra `where`.
  Complete successful execution, actual trace safety, candidate transport, and
  full flat IO/frame also passed (`llama-proofs-fullexecution.log` through
  `llama-proofs-fullcontracts.log`).
- Whole typed FP equality passed (`/private/tmp/llama-fpequiv.log`); printed
  assumptions: none. Complete state, aliases and failure behavior are preserved.
- Eight nonempty numerical single/multiblock witnesses passed
  (`/private/tmp/llama-proofs-execution.log`), with actual sum/sqrt and stored values.
- Historical signatures/docstrings: all 33 declaration headers and 26 docstrings
  compared equal across Legacy/Correct. The obsolete top-level entry was removed;
  no external Lean imports referenced it.
- Legacy and Correct passed, including both historical headlines and both new
  named Spec.Real full-row contracts. Five public heads, source definitions,
  precision checks, actual trace safety and numerical readback passed their
  explicit axiom audit: only the standard three axioms. The final exhaustive
  audit (`/private/tmp/llama-proofs-audit.log`, exit 0 after the DSL refresh)
  checked all 829 discovered theorem declarations, including private/generated
  helpers, and found only the standard base. No local compiler remains live.

See `source_scope.md` for floating-family, native-width, discrete-conversion,
allocation, initial snapshot and per-program boundaries. These are not silently
replaced by a single-block source or by a precomputed output model.
