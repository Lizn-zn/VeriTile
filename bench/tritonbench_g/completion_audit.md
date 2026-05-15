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
| Fix Python/Lean mismatches found by the sweep. | Recent fixes restored faithful loop/tuple/statement surfaces and moved policy checks into `bench/audit_tritonbench_g.sh`; current audit passes. | No current unannotated mechanical mismatch; 20 documented translation-surface blockers remain. |
| Ensure completed ports expose a standard correctness surface. | Audit scans every `.lean` for `ComputeCorrect.Realizes`, `ComputeRefine.Realizes`, `ComputeCorrect.General`, or a named `correct_target`. | Passing. |
| Do not count placeholder proofs as complete. | Placeholder scan for `True := by`, `trivial`, `sorry`, and `admit` reports no matches. | Passing. |
| Do not close while algorithm-layer proof obligations remain. | Audit now checks that there are no explicit `hAlg` blockers. | Passing for algorithm-layer blockers. Translation-surface blockers remain and are documented below. |

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

## Remaining Blockers

No explicit TritonBench-G `hAlg` blocker remains.

There are still 20 documented translation-surface blockers. These are not
silent green ports: each remaining marker must be covered by
`proof_blockers.md`, and `bench/audit_tritonbench_g.sh` enforces that coverage.
The current documented blocker set is:

- `attention_forward_triton`
- `attention_fwd_triton1`
- `attention_fwd_triton2`
- `attention_fwd_triton3`
- `attention_kernel`
- `attention_kernel_aligned`
- `attention_score`
- `attn_fwd_causal`
- `attn_fwd_triton`
- `context_attn_bloom`
- `context_attn_fwd`
- `context_attn_llama`
- `context_attn_mistral`
- `context_attn_nopad`
- `flash_attn`
- `iv_dependent_matmul`
- `mixed_sparse_attention`
- `rotary_transform`
- `rotary_transform_ops`
- `triton_attention`

Passing `lake build` alone is still not sufficient evidence for future changes;
this audit must continue to run the translation-consistency gates above.
