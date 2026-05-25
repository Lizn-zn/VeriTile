# TritonBench-G Completion Audit

Objective: check every `bench/tritonbench_g` problem against
`review_criteria.md`, keep Python and Lean kernel bodies faithful, and ensure
each completed port has a standard `ComputeCorrect.Realizes` correctness
surface.

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Current status |
|---|---|---|
| Check every `bench/tritonbench_g` problem. | 184 work directories are present; 141 currently have `.py` / `.lean` port pairs, and 43 are README-only scaffolds. | Covered for completed port pairs; scaffolds are not counted as completed ports. |
| Ensure every completed Python port has a Lean port. | Python/Lean file counts both report 141; `bench/audit_tritonbench_g.sh` enforces the count match. | Passing. |
| Ensure Lean ports compile. | `bench/check_ports.sh` reports `TritonBench-G ports: 141 ok, 0 fail`; the audit script reruns this gate. | Passing. |
| Apply `review_criteria.md` faithful-translation rules. | Mechanical gates check dtype-load additions, `keep_dims` substitutions, `+=` coverage, normalized pointer-update lhs, `rsqrt` preservation, Lean-only `tl.where`, `tl.*(...)` call set/order, kernel control-flow counts, statement lhs order, and documented translation-surface blockers. | Mechanically covered for the listed must-fix patterns; still not a substitute for human line review of arbitrary arithmetic structure. |
| Fix Python/Lean mismatches found by the sweep. | Recent fixes restored faithful loop/tuple/helper-call/statement surfaces and moved policy checks into `bench/audit_tritonbench_g.sh`; current audit passes. | No current unannotated mechanical mismatch and no documented translation-surface blocker remains. |
| Ensure completed ports expose a standard correctness surface. | Audit scans every `.lean` for `ComputeCorrect.Realizes`, `ComputeRefine.Realizes`, `ComputeCorrect.General`, or a named `correct_target`. | Passing. |
| Classify stronger proof gaps from #146. | `bench/check_proof_gap_manifest.py` extracts every `output_summary` declaration and checks it against `proof_gap_manifest.tsv`. | Passing; 180 summaries are classified across 76 files. |
| Do not count placeholder proofs as complete. | Placeholder scan for `True := by`, `trivial`, `sorry`, and `admit` reports no matches. | Passing. |
| Do not close while algorithm-layer proof obligations remain. | Audit now checks that there are no explicit `hAlg` blockers and no stale translation-surface blocker entries. | Passing; no algorithm-layer or translation-surface blocker remains. |

## Evidence Checked

- Directory coverage: `find bench/tritonbench_g -mindepth 1 -maxdepth 1 -type d`
  reports 184 work directories. Of these, 141 currently contain a `.py` /
  `.lean` port pair; the remaining 43 are README-only scaffolds and are not
  counted as completed ports by this audit.
- Python/Lean file coverage: `find bench/tritonbench_g -maxdepth 2 -name '*.py'`
  and the matching Lean query both report 141 files.
- Build gate: `lake build` succeeds.
- Per-port source elaboration gate: `bench/check_ports.sh` reports
  `TritonBench-G ports: 141 ok, 0 fail`.
- Mechanical audit gate: `bench/audit_tritonbench_g.sh` reports
  `TritonBench-G audit gates passed`, covering Python/Lean count matching,
  port elaboration, placeholder-proof scanning, and correctness-surface
  scanning. It also checks that compiled ports do not still advertise README
  `TODO` status and that Python `.to(tl.float32)` casts missing from Lean are
  covered by an explicit documented slice/scope note. The same gate rejects
  any algorithm-layer `hAlg` blockers and requires every remaining
  translation-surface marker to have a corresponding `proof_blockers.md`
  entry. It rejects Lean-only `tl.load(..., dtype=...)`
  annotations and `keep_dims` reduction substitutions, both of which are
  must-fix deviations under
  `review_criteria.md`. It also flags Python `+=` statements missing from Lean
  unless the Lean port documents that the update is outside a proof slice or
  branch/surface specialization; this includes a normalized left-hand-side
  check that treats names like `a_ptr` and `A` as the same pointer. It checks
  that upstream `rsqrt` calls are preserved rather than rewritten as reciprocal
  square roots. It also rejects Lean-only `tl.where` statements, another
  must-fix "extra statement" pattern in `review_criteria.md`. Finally, it
  compares the Python and Lean `tl.*(...)` call surfaces and requires any
  missing or extra call to be covered by an explicit slice/specialization note.
  It also compares `for` / `while` / `if` counts inside the Python
  `@triton.jit` kernel body and Lean `triton { ... }` body to catch
  unannotated control-flow rewrites, and compares the ordered `tl.*(...)` call
  sequence to catch unannotated call reordering. A statement left-hand-side
  sequence scan covers top-level assignments, `+=` updates, annotated
  assignments, and tuple assignments, so added/removed/reordered non-call
  statements are also checked mechanically. The audit also compares the
  explicit `completion_audit.md` Remaining Blockers list against the active
  Lean preamble marker set, so stale or missing blocker entries fail the gate.
- Placeholder scan:
  `rg -n "True := by|trivial|sorry|admit" bench/tritonbench_g -g '*.lean'`
  currently reports no matches.
- Correctness-surface scan:
  every `bench/tritonbench_g/*/*.lean` file now contains a
  `ComputeCorrect.Realizes` target or theorem.
- Proof-gap manifest scan:
  `bench/check_proof_gap_manifest.py` reports 180 `output_summary`
  declarations across 76 files. It classifies 58 as conservative
  `full_value_candidate`, 109 as `public_summary_with_proof_gap`, and 13 as
  `blocked_summary`. Every non-full candidate is linked to a specific follow-up
  issue and blocker family in `proof_gap_manifest.tsv`; no row remains linked
  to the former broad #147 quantization bucket, the former broad #148
  matmul/dot bucket, the former broad #153 rotary/cache bucket, or the broad
  #152 `semantic-blocker` bucket. The #148 rows now split into GEMV K-loop,
  BMM final-store, dequantized cross-kernel, IV-dependent output-store, plain
  matmul output-store, activation-tail, and TMA output-store accumulator
  blockers. The #152 blocked summaries now migrate to the open #154
  `fixed-width-int8-cast-semantics` issue because their concrete blocker is
  CUDA `llrint` / int8-cast semantics; the #153 rows split into
  `rope-head-slice-lift` for RoPE Q/K head-slice value proofs and
  `rotary-2d-tile-value-lift` for row-level rotary `o0`/`o1` proofs that still
  need the full `[BLOCK_M, BLOCK_HALF]` 2D tile lift. The #167 context-attention
  rows split into Mistral sliding-window accumulator-to-store and nopad
  variable-length accumulator-to-store obligations. The #172 Phi flash-decode
  row now splits into running-max, masked scaled-accumulator, and final
  normalization/store obligations linked to #175, #176, and #177.

## Remaining Blockers

No explicit TritonBench-G `hAlg` blocker remains, and no documented
translation-surface blocker remains. If a future Lean port reintroduces a
translation-scope marker, it must be covered by `proof_blockers.md`, and
`bench/audit_tritonbench_g.sh` enforces that coverage.
The current documented blocker set is:

Passing `lake build` alone is still not sufficient evidence for future changes;
this audit must continue to run the translation-consistency gates above and the
#146 proof-gap manifest check.
